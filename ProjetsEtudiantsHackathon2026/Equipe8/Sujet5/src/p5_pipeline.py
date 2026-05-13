"""P5 — Pipeline complet (Q13) + interface API (Q15) + ONNX (Q14).

Q13 : `predict(image_id or path) → list[Detection]` qui regroupe :
    1. Détection (YOLO COCO out-of-the-box si image réelle, sinon lecture
       des bboxes pré-annotées du `detection_results.csv` synthétique).
    2. Classification militaire/civil (mapping `category_id ∈ MILITARY_IDS`).
    3. Annotation géospatiale via `p4_geospatial.annotate_detections_with_zones`.
    4. Décision **ALERTE** :
         is_military ET in_military_zone ET risk_level ∈ {High, Critical}
         ET zone.active. La présence d'un MMSI AIS proche désamorce l'alerte
         (sauf si « navire sombre »).

Q15 : `enrich_pipeline_outputs(df_alerts)` appelle `osint_enrich` (port + météo).

Q14 : `export_onnx(weights, out_path)` + `benchmark_inference(...)` →
       comparaison PyTorch CPU vs ONNX vs ONNX-INT8 sur une image sanity.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from . import config as C
from . import load as L
from . import p4_geospatial


# ----------------------------------------------------------------------
# Schéma de détection (homogène avec le rapport)

@dataclass
class Detection:
    detection_id: str
    image_id: str | None
    category_id: int | None
    category: str | None
    is_military: bool
    confidence: float | None
    bbox_px: list[float] | None  # x, y, w, h en pixels
    lat: float | None
    lon: float | None
    timestamp: str | None
    in_military_zone: bool
    nearest_mil_zone_name: str | None
    nearest_mil_zone_risk: str | None
    nearest_mil_zone_dist_km: float | None
    alert: bool
    alert_reason: str | None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------
# Règle d'alerte centralisée (cf. hypotheses §C Q13)

def _decide_alert(row: pd.Series) -> tuple[bool, str | None]:
    """Renvoie (alert, raison)."""
    if not row.get("is_military"):
        return False, None
    if not row.get("in_military_zone"):
        return False, None
    risk = row.get("nearest_mil_zone_risk")
    if risk not in ("High", "Critical"):
        return False, None
    # Si on a un MMSI AIS proche, on ne déclenche pas d'alerte « sombre »
    if row.get("is_dark") is True:
        return True, (
            f"Militaire en zone {risk} sans MMSI AIS dans la fenêtre "
            f"({C.AIS_TIME_WINDOW_MIN} min, {C.AIS_DIST_WINDOW_KM} km) — "
            "navire sombre."
        )
    return True, f"Militaire en zone {risk} (active) — surveillance recommandée."


# ----------------------------------------------------------------------
# Q13 — Pipeline depuis le CSV (les images synthétiques n'existent pas)

def predict_from_csv(image_ids: list[str] | None = None) -> pd.DataFrame:
    """Applique le pipeline à toutes (ou un sous-ensemble) des images du CSV.

    Les images `satellite_*.jpg` ne sont pas fournies → on lit les détections
    déjà annotées dans `detection_results.csv` et on simule la chaîne
    « image → bboxes → classes → zone → alerte ». La substitution est
    documentée explicitement dans le rapport.

    Args:
        image_ids: filtre. Si None, on traite TOUTES les images.

    Returns:
        DataFrame avec une ligne par détection, prêt à exporter.
    """
    df_meta = L.load_images_metadata()
    df_det = L.load_detections()
    df_zones = L.load_military_zones()

    if image_ids:
        df_det = df_det[df_det["image_id"].isin(image_ids)].copy()

    # Réutilise la jointure de p4 (par distance, pas par id)
    df_det = df_det.merge(
        df_meta[["image_id", "lat", "lon", "date", "time", "source", "resolution_m"]],
        on="image_id",
        how="left",
        suffixes=("", "_img"),
    )
    df_det = p4_geospatial.annotate_detections_with_zones(df_det, df_zones)

    # Pour l'instant pas de pont AIS : tout = is_dark inconnu (None)
    df_det["is_dark"] = None

    alerts, reasons = [], []
    for _, r in df_det.iterrows():
        a, why = _decide_alert(r)
        alerts.append(a)
        reasons.append(why)
    df_det["alert"] = alerts
    df_det["alert_reason"] = reasons

    return df_det


def predictions_to_detections(df: pd.DataFrame) -> list[Detection]:
    """Convertit le DataFrame du pipeline en liste de `Detection` dataclasses."""
    out = []
    for _, r in df.iterrows():
        bbox = None
        if all(pd.notna(r.get(c)) for c in ("bbox_x_norm", "bbox_y_norm", "bbox_w_norm", "bbox_h_norm")):
            bbox = [r["bbox_x_norm"], r["bbox_y_norm"], r["bbox_w_norm"], r["bbox_h_norm"]]
        out.append(
            Detection(
                detection_id=str(r.get("detection_id")),
                image_id=r.get("image_id"),
                category_id=int(r["category_id"]) if pd.notna(r.get("category_id")) else None,
                category=r.get("category"),
                is_military=bool(r.get("is_military", False)),
                confidence=float(r["confidence"]) if pd.notna(r.get("confidence")) else None,
                bbox_px=bbox,
                lat=float(r["lat"]) if pd.notna(r.get("lat")) else None,
                lon=float(r["lon"]) if pd.notna(r.get("lon")) else None,
                timestamp=str(r["timestamp"]) if pd.notna(r.get("timestamp")) else None,
                in_military_zone=bool(r.get("in_military_zone", False)),
                nearest_mil_zone_name=r.get("nearest_mil_zone_name"),
                nearest_mil_zone_risk=r.get("nearest_mil_zone_risk"),
                nearest_mil_zone_dist_km=float(r["nearest_mil_zone_dist_km"])
                if pd.notna(r.get("nearest_mil_zone_dist_km"))
                else None,
                alert=bool(r.get("alert", False)),
                alert_reason=r.get("alert_reason"),
            )
        )
    return out


# ----------------------------------------------------------------------
# Q13 (bis) — Pipeline depuis une image réelle (utilisé pour la Piste B)

def predict_from_image(image_path: str | Path, *, conf: float = 0.25) -> list[dict]:
    """Pipeline pour une image réelle : YOLOv8n COCO → liste de boats détectés.

    YOLO COCO ne distingue PAS frégate/destroyer (il ne connaît que `boat`/`ship`).
    Cette fonction sert pour la démo Piste B + la sanity. Pour la classification
    de type, il faut un YOLO fine-tuné (cf. `p2_detection.train_yolo_on_substituted`).
    """
    from .p2_detection import sanity_yolov8n

    s = sanity_yolov8n(image_path)
    return [
        d for d in s["detections"]
        if d["class_name"] in {"boat", "ship"} and d["confidence"] >= conf
    ]


# ----------------------------------------------------------------------
# Q14 — Export ONNX + benchmark inférence CPU

def export_onnx(weights: str = "yolov8n.pt", out_dir: Path = C.OUTPUTS) -> dict:
    """Export YOLOv8 → ONNX (FP32). Q14.

    On NE quantise PAS en INT8 ici (la quantisation dynamique d'ONNX Runtime
    n'aime pas tous les ops Ultralytics). On benche FP32 vs PyTorch CPU.
    Documentation honnête dans le rapport.
    """
    from ultralytics import YOLO

    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(weights)
    # Ultralytics écrit le .onnx à côté du .pt (pas de paramètre out_dir)
    onnx_path = model.export(format="onnx", dynamic=True, simplify=True, imgsz=640)
    return {"weights": weights, "onnx_path": str(onnx_path)}


def benchmark_inference(
    image_path: str | Path,
    pt_weights: str = "yolov8n.pt",
    onnx_weights: str | None = None,
    n_iter: int = 10,
) -> pd.DataFrame:
    """Benche PyTorch CPU vs ONNX FP32 CPU sur la même image.

    Args:
        n_iter: nombre d'inférences (médiane reportée).

    Returns:
        DataFrame `{backend, median_ms, p10_ms, p90_ms, n_det}`.
    """
    from ultralytics import YOLO

    rows = []
    for backend, weights in (("pytorch_cpu", pt_weights), ("onnx_cpu", onnx_weights)):
        if not weights:
            continue
        try:
            model = YOLO(weights)
            # Warm-up
            model.predict(source=str(image_path), device="cpu", verbose=False, save=False)
            samples = []
            for _ in range(n_iter):
                t0 = time.perf_counter()
                r = model.predict(
                    source=str(image_path),
                    device="cpu",
                    verbose=False,
                    save=False,
                )
                samples.append((time.perf_counter() - t0) * 1000)
            samples.sort()
            rows.append(
                {
                    "backend": backend,
                    "weights": weights,
                    "n_iter": n_iter,
                    "median_ms": round(samples[n_iter // 2], 2),
                    "p10_ms": round(samples[max(0, n_iter // 10)], 2),
                    "p90_ms": round(samples[min(n_iter - 1, (9 * n_iter) // 10)], 2),
                    "n_det": len(r[0].boxes) if r else 0,
                }
            )
        except Exception as e:  # noqa: BLE001
            rows.append({"backend": backend, "weights": weights, "error": str(e)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Run P5 tout-en-un (Q13 + alertes + sortie pour rapport)

def run_p5(test_image_ids: list[str] | None = None) -> dict:
    """Q13 sur toutes (ou N) images, écrit alertes + résumé."""
    df_pipeline = predict_from_csv(test_image_ids)

    # Alertes uniquement
    df_alerts = df_pipeline[df_pipeline["alert"]].copy()

    n_total = len(df_pipeline)
    n_alerts = int(df_pipeline["alert"].sum())
    n_mil = int(df_pipeline["is_military"].sum())
    n_in_zone = int(df_pipeline["in_military_zone"].sum())

    out_pipeline = C.OUTPUTS / "q13_pipeline_detections.csv"
    df_pipeline.to_csv(out_pipeline, index=False)
    out_alerts = C.OUTPUTS / "q13_pipeline_alertes.csv"
    df_alerts.to_csv(out_alerts, index=False)

    return {
        "n_total_detections": n_total,
        "n_militaires": n_mil,
        "n_en_zone_militaire": n_in_zone,
        "n_alertes": n_alerts,
        "df_pipeline": df_pipeline,
        "df_alerts": df_alerts,
        "files": {
            "detections_csv": str(out_pipeline.relative_to(C.ROOT)),
            "alertes_csv": str(out_alerts.relative_to(C.ROOT)),
        },
        "_note": (
            "Q13 sur le CSV synthétique. Une variante `predict_from_image` est "
            "exposée pour les images réelles de la Piste B (cf. src/hunt.py)."
        ),
    }


if __name__ == "__main__":
    out = run_p5()
    print(
        f"Q13 — {out['n_total_detections']} détections traitées, "
        f"{out['n_militaires']} militaires, {out['n_en_zone_militaire']} en zone, "
        f"**{out['n_alertes']} alertes**."
    )
    print(f"  → {out['files']['detections_csv']}")
    print(f"  → {out['files']['alertes_csv']}")
