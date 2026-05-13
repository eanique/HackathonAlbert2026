"""Chargement & normalisation des données fournies.

- `images_metadata_large.csv` : 100 lignes de métadonnées synthétiques.
- `annotations_large.json` : 100 images × 256 annotations × 13 catégories, format COCO.
  ⚠️ bbox en **normalisé [0,1]** — on les reconvertit toujours en pixels.
- `detection_results.csv` : 256 détections (≈ recopie de annotations + colonnes zone).
- `military_zones.csv` : 20 zones, `zone_id = MIL-XXX` (NB : ne se joint PAS à
  `images_metadata.zone_id = ZONE-XXX` — joindre par distance, cf. config.MIL_ZONE_RADIUS_KM).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from . import config as C


# ----------------------------------------------------------------------
# Parsing helpers

_RESOLUTION_RE = re.compile(r"([\d.]+)\s*m", re.IGNORECASE)


def parse_resolution(s: str | float) -> float:
    """`"10m"` -> `10.0`, `"0.5m"` -> `0.5`. NaN-tolerant."""
    if pd.isna(s):
        return float("nan")
    m = _RESOLUTION_RE.match(str(s).strip())
    return float(m.group(1)) if m else float("nan")


def parse_latlon(s: str | float) -> tuple[float, float] | None:
    """`"36.9,-76.3"` -> `(36.9, -76.3)`. Renvoie `None` si invalide."""
    if pd.isna(s):
        return None
    parts = str(s).split(",")
    if len(parts) != 2:
        return None
    try:
        return float(parts[0].strip()), float(parts[1].strip())
    except ValueError:
        return None


def parse_bbox_str(s: str | float) -> list[float] | None:
    """`"0.37,0.60,0.27,0.06"` (4 floats normalisés) -> `[0.37, 0.60, 0.27, 0.06]`."""
    if pd.isna(s):
        return None
    parts = [p.strip() for p in str(s).split(",")]
    if len(parts) != 4:
        return None
    try:
        return [float(p) for p in parts]
    except ValueError:
        return None


def parse_bool(s: Any) -> bool:
    if isinstance(s, bool):
        return s
    return str(s).strip().lower() == "true"


# ----------------------------------------------------------------------
# Loaders

def load_images_metadata(path: Path = C.F_IMAGES_META) -> pd.DataFrame:
    """100 images avec métadonnées. Ajoute `lat`, `lon`, `resolution_m`."""
    df = pd.read_csv(path)
    df["resolution_m"] = df["resolution"].map(parse_resolution)
    latlon = df["coordinates"].map(parse_latlon)
    df["lat"] = latlon.map(lambda p: p[0] if p else None)
    df["lon"] = latlon.map(lambda p: p[1] if p else None)
    df["cloud_cover"] = pd.to_numeric(df["cloud_cover"], errors="coerce")
    df["num_ships"] = pd.to_numeric(df["num_ships"], errors="coerce").astype("Int64")
    return df


def load_annotations(path: Path = C.F_ANNOTATIONS) -> dict:
    """Charge le JSON COCO et **annote chaque annotation** avec bbox px et area px.

    Retourne le dict COCO + une clé `df_annotations` (pandas) prête à l'analyse.
    """
    with path.open() as f:
        coco = json.load(f)

    images = {im["id"]: im for im in coco["images"]}
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    rows = []
    for a in coco["annotations"]:
        im = images.get(a["image_id"], {})
        W, H = im.get("width"), im.get("height")
        x_n, y_n, w_n, h_n = a["bbox"]
        if W and H:
            x_px = x_n * W
            y_px = y_n * H
            w_px = w_n * W
            h_px = h_n * H
            area_px = w_px * h_px
        else:
            x_px = y_px = w_px = h_px = area_px = float("nan")
        rows.append(
            {
                "annotation_id": a["id"],
                "image_id": a["image_id"],
                "file_name": im.get("file_name"),
                "image_width": W,
                "image_height": H,
                "category_id": a["category_id"],
                "category_name": cats.get(a["category_id"]),
                "is_military": a["category_id"] in C.MILITARY_IDS,
                "bbox_x_norm": x_n,
                "bbox_y_norm": y_n,
                "bbox_w_norm": w_n,
                "bbox_h_norm": h_n,
                "bbox_x_px": x_px,
                "bbox_y_px": y_px,
                "bbox_w_px": w_px,
                "bbox_h_px": h_px,
                "area_px": area_px,
                "area_coco_raw": a.get("area"),
                "bbox_ratio_wh": w_n / h_n if h_n > 0 else float("nan"),
                "confidence": a.get("confidence"),
            }
        )
    coco["df_annotations"] = pd.DataFrame(rows)
    return coco


def load_detections(path: Path = C.F_DETECTIONS) -> pd.DataFrame:
    """256 détections du CSV. Ajoute bbox parsé."""
    df = pd.read_csv(path)
    bbox = df["bbox"].map(parse_bbox_str)
    df["bbox_x_norm"] = bbox.map(lambda b: b[0] if b else None)
    df["bbox_y_norm"] = bbox.map(lambda b: b[1] if b else None)
    df["bbox_w_norm"] = bbox.map(lambda b: b[2] if b else None)
    df["bbox_h_norm"] = bbox.map(lambda b: b[3] if b else None)
    df["is_military"] = df["is_military"].map(parse_bool)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def load_military_zones(path: Path = C.F_MIL_ZONES) -> pd.DataFrame:
    """20 zones militaires. Ajoute `lat`, `lon`, `active_bool`.

    ⚠️ `zone_id` ici est de la forme `MIL-XXX` — NE correspond PAS au `zone_id =
    ZONE-XXX` de `images_metadata_large.csv`. Joindre par distance, pas par id.
    """
    df = pd.read_csv(path)
    latlon = df["coordinates"].map(parse_latlon)
    df["lat"] = latlon.map(lambda p: p[0] if p else None)
    df["lon"] = latlon.map(lambda p: p[1] if p else None)
    df["active_bool"] = df["active"].map(parse_bool)
    return df
