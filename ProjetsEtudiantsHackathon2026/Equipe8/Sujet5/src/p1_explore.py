"""P1 — Exploration des données fournies (Q1, Q2, Q3).

Toutes les fonctions renvoient un `dict` structuré : utilisé directement
par `reponses_generalisation_detection_navires.py` pour les sorties chiffrées.
"""

from __future__ import annotations

import pandas as pd

from . import config as C
from . import load as L


def q1_metadata(df_meta: pd.DataFrame) -> dict:
    """Top 3 sources, résolution moyenne, % images avec cloud_cover > 30 %."""
    top3_sources = df_meta["source"].value_counts().head(3).to_dict()
    mean_resolution_m = df_meta["resolution_m"].mean()
    pct_cloud_gt_30 = (df_meta["cloud_cover"] > 30).mean() * 100
    return {
        "top3_sources": top3_sources,
        "mean_resolution_m": float(mean_resolution_m),
        "n_cloud_gt_30": int((df_meta["cloud_cover"] > 30).sum()),
        "pct_cloud_gt_30": float(pct_cloud_gt_30),
        "_warning_source_resolution_incoherence": (
            "Sentinel-2 réel = 10 m toujours (jamais 1 m). Landsat-8 = 30 m. "
            "Le couple (source, resolution) du CSV fourni est aléatoire / synthétique."
        ),
    }


def q2_annotations(df_ann: pd.DataFrame) -> dict:
    """Nb militaires, répartition % par type, taille moyenne bbox par type (en pixels)."""
    n_mil = int(df_ann["is_military"].sum())
    n_total = len(df_ann)
    repartition_pct = (
        df_ann["category_name"].value_counts(normalize=True) * 100
    ).round(2).to_dict()
    bbox_by_type = (
        df_ann.groupby("category_name")
        .agg(
            n=("annotation_id", "count"),
            mean_w_px=("bbox_w_px", "mean"),
            mean_h_px=("bbox_h_px", "mean"),
            mean_area_px=("area_px", "mean"),
        )
        .round(2)
        .sort_values("n", ascending=False)
    )
    return {
        "n_militaires": n_mil,
        "n_total_annotations": n_total,
        "pct_militaires": round(n_mil / n_total * 100, 2) if n_total else 0,
        "repartition_pct": repartition_pct,
        "bbox_by_type": bbox_by_type,
        "_note": (
            "Hypothèse militaire ⟺ category_id ∈ {2,3,4,5,9,10,11,12,13}. "
            "Bbox reconverti en pixels avec width/height de l'image (le `area` "
            "du JSON est incohérent avec la bbox normalisée → recalculé)."
        ),
    }


def q3_fusion(df_meta: pd.DataFrame, df_det: pd.DataFrame) -> dict:
    """Fusion images_metadata × detection_results, nb détections/image, image max."""
    merged = df_meta.merge(
        df_det, on=["image_id", "file_name"], suffixes=("_img", "_det"), how="left"
    )
    n_det_per_image = df_det.groupby("image_id").size()
    image_max = n_det_per_image.idxmax()
    n_max = int(n_det_per_image.max())
    return {
        "n_rows_merged": len(merged),
        "mean_det_per_image": float(n_det_per_image.mean()),
        "median_det_per_image": float(n_det_per_image.median()),
        "max_det_image_id": image_max,
        "max_det_count": n_max,
        "df_merged": merged,
        "_note": (
            "`detection_results.csv` est ≈ une recopie de `annotations_large.json` "
            "enrichie de colonnes zone → fusion partiellement triviale (mêmes bbox)."
        ),
    }


# ----------------------------------------------------------------------
# Run tout-en-un (utilisé par reponses_generalisation_detection_navires.py)

def run_p1() -> dict:
    df_meta = L.load_images_metadata()
    coco = L.load_annotations()
    df_ann = coco["df_annotations"]
    df_det = L.load_detections()
    return {
        "Q1": q1_metadata(df_meta),
        "Q2": q2_annotations(df_ann),
        "Q3": q3_fusion(df_meta, df_det),
        "_data": {
            "df_meta": df_meta,
            "df_ann": df_ann,
            "df_det": df_det,
            "n_images_in_coco": len(coco["images"]),
            "n_categories": len(coco["categories"]),
        },
    }
