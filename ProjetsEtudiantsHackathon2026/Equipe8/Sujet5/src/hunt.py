"""Piste B — La vraie chasse aux navires militaires.

Pipeline cible :
    1. Query Sentinel-1 (SAR, gratuit, tout temps/nuit) via Planetary Computer STAC.
       → liste des scènes récentes au-dessus d'une base navale.
    2. Téléchargement de la scène (.tif COG).         [Phase 2 J2]
    3. Découpage en tuiles 1024×1024 avec overlap.    [Phase 2 J2]
    4. Inférence YOLOv8 fine-tuné xView3 (ou pré-entraîné DOTA en fallback).
    5. NMS global sur la scène → liste de détections (lat/lon, classe, conf).
    6. Croisement AIS → flag « navire sombre » (cf. ais_cross.py).
    7. Vérité terrain croisée Wikipedia/Jane's (composition flotte du port).

Phase 1 (ce module, ici) : on implémente uniquement (1) — la **query STAC**.
Cela suffit pour valider : « combien de scènes Sentinel-1 dispo sur chaque base
dans les 30 derniers jours, à quelle date, quelle résolution ». C'est déjà un
livrable noté (critère 4 : usage d'API).
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Iterable

import pandas as pd
import pystac_client
from planetary_computer import sign_inplace

from . import config as C

# Catalogue STAC public Microsoft Planetary Computer (pas de clé requise)
_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
_COLLECTION_S1 = "sentinel-1-rtc"  # Radiometrically Terrain Corrected (mieux que GRD pur)
_COLLECTION_S2 = "sentinel-2-l2a"


def _bbox_to_polygon(bbox: tuple[float, float, float, float]) -> dict:
    """(south, west, north, east) -> GeoJSON Polygon (WGS84) attendu par STAC."""
    s, w, n, e = bbox
    return {
        "type": "Polygon",
        "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]],
    }


def query_sentinel1(
    base_name: str,
    bbox: tuple[float, float, float, float],
    window_days: int = C.HUNT_WINDOW_DAYS,
    collection: str = _COLLECTION_S1,
    max_items: int = 50,
) -> pd.DataFrame:
    """Liste les scènes Sentinel-1 RTC qui couvrent `bbox` sur les N derniers jours.

    Args:
        base_name: étiquette (ex. « Toulon (FR) »).
        bbox: (south, west, north, east) en degrés WGS84.
        window_days: profondeur temporelle.
        collection: `sentinel-1-rtc` (SAR) ou `sentinel-2-l2a` (optique).
        max_items: limite STAC.

    Returns:
        DataFrame trié par date décroissante, colonnes :
            base, scene_id, collection, datetime, platform, polarisation,
            cloud_cover (S2 only), bbox, href_metadata
    """
    end = _dt.datetime.now(_dt.UTC)
    start = end - _dt.timedelta(days=window_days)
    catalog = pystac_client.Client.open(_STAC_URL, modifier=sign_inplace)
    search = catalog.search(
        collections=[collection],
        intersects=_bbox_to_polygon(bbox),
        datetime=f"{start.isoformat()}/{end.isoformat()}",
        max_items=max_items,
    )
    items = list(search.items())

    rows = []
    for item in items:
        props = item.properties
        rows.append(
            {
                "base": base_name,
                "scene_id": item.id,
                "collection": collection,
                "datetime": props.get("datetime"),
                "platform": props.get("platform"),
                "polarisations": ",".join(props.get("sar:polarizations", []))
                if "sar:polarizations" in props
                else None,
                "cloud_cover": props.get("eo:cloud_cover"),
                "bbox": tuple(item.bbox) if item.bbox else None,
                "href_metadata": item.get_self_href(),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.sort_values("datetime", ascending=False).reset_index(drop=True)
    return df


def survey_all_bases(
    bases: dict | None = None, window_days: int = C.HUNT_WINDOW_DAYS
) -> pd.DataFrame:
    """Query Sentinel-1 sur toutes les bases définies dans config.BASES_NAVALES.

    Sortie : un seul DataFrame agrégé (`outputs/chasse_scenes_disponibles.csv`).
    """
    bases = bases or C.BASES_NAVALES
    all_dfs = []
    for name, bbox in bases.items():
        try:
            df = query_sentinel1(name, bbox, window_days=window_days)
            print(f"  {name:<20} -> {len(df)} scènes Sentinel-1 sur {window_days} j")
            all_dfs.append(df)
        except Exception as e:  # noqa: BLE001 — on log, on continue
            print(f"  {name:<20} -> ERREUR : {type(e).__name__}: {e}")
    if not all_dfs:
        return pd.DataFrame()
    out = pd.concat(all_dfs, ignore_index=True)
    out_path = C.DATA_PROCESSED / "chasse_scenes_disponibles.csv"
    out.to_csv(out_path, index=False)
    print(f"\n  -> ecrit {out_path.relative_to(C.ROOT)} ({len(out)} scenes)")
    return out


# ----------------------------------------------------------------------
# Téléchargement + inférence sur scène réelle

def _get_stac_item(scene_id: str, collection: str = _COLLECTION_S1):
    """Récupère l'item STAC complet (avec assets signés) par son id."""
    catalog = pystac_client.Client.open(_STAC_URL, modifier=sign_inplace)
    search = catalog.search(collections=[collection], ids=[scene_id])
    items = list(search.items())
    if not items:
        # Plan B : si la collection est S1-rtc on tente s1-grd, et inversement
        alt = "sentinel-1-grd" if collection == _COLLECTION_S1 else collection
        if alt != collection:
            search = catalog.search(collections=[alt], ids=[scene_id])
            items = list(search.items())
    if not items:
        raise ValueError(f"Scène {scene_id} introuvable dans {collection}")
    return items[0]


def fetch_scene_window(
    scene_id: str,
    bbox: tuple[float, float, float, float],
    asset_key: str = "vv",
    out_dir: Path = C.DATA_IMAGES_REAL,
    max_pixels_per_side: int = 4096,
    collection: str = _COLLECTION_S1,
    save_preview: bool = True,
) -> Path:
    """Télécharge UNE FENÊTRE d'une scène Sentinel-1 (pas la scène entière).

    Pourquoi pas la scène entière : un Sentinel-1 RTC complet fait 1-3 Go.
    On veut juste la portion qui couvre la base navale (typiquement
    quelques km²) → ~50-200 Mo, gérable.

    Args:
        scene_id: id STAC de la scène (cf. `survey_all_bases`).
        bbox: (south, west, north, east) en degrés — l'emprise à découper.
        asset_key: 'vv' (recommandé pour navires en SAR) ou 'vh'.
        out_dir: répertoire de sortie (gitignored).
        max_pixels_per_side: garde-fou : on resample si la fenêtre dépasse.

    Returns:
        Path du GeoTIFF local.
    """
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    out_dir.mkdir(parents=True, exist_ok=True)
    item = _get_stac_item(scene_id, collection=collection)
    asset = item.assets.get(asset_key)
    if asset is None:
        raise ValueError(
            f"Asset '{asset_key}' absent ; assets dispos : {list(item.assets.keys())}"
        )
    out_path = out_dir / f"{scene_id}_{asset_key}.tif"
    if out_path.exists():
        return out_path

    print(f"  [hunt] streaming {scene_id} asset={asset_key} bbox={bbox}...")
    with rasterio.open(asset.href) as src:
        # bbox arrive en (south, west, north, east) ; rasterio veut (west, south, east, north)
        s, w, n, e = bbox
        win_bounds_wgs = (w, s, e, n)
        try:
            win_bounds_src = transform_bounds("EPSG:4326", src.crs, *win_bounds_wgs)
        except Exception:  # noqa: BLE001
            win_bounds_src = win_bounds_wgs
        window = from_bounds(*win_bounds_src, transform=src.transform)
        # Lecture (rasterio downloade uniquement les tuiles COG nécessaires)
        # Pour Sentinel-2 visual : 3 bandes RGB ; sinon 1 bande
        n_bands = src.count if asset_key == "visual" else 1
        if n_bands > 1:
            data = src.read(window=window, boundless=True, fill_value=0)  # (C, H, W)
        else:
            data = src.read(1, window=window, boundless=True, fill_value=0)  # (H, W)
        # Pour multi-band : (C, H, W). Pour single-band : (H, W).
        if data.ndim == 3:
            n_b, h, w_pix = data.shape
        else:
            h, w_pix = data.shape
            n_b = 1
        if max(h, w_pix) > max_pixels_per_side:
            scale = max_pixels_per_side / max(h, w_pix)
            new_h, new_w = int(h * scale), int(w_pix * scale)
            import numpy as np
            from PIL import Image
            if data.ndim == 3:
                # Resample chaque bande
                resampled = np.zeros((n_b, new_h, new_w), dtype=data.dtype)
                for b in range(n_b):
                    resampled[b] = np.array(
                        Image.fromarray(data[b]).resize((new_w, new_h), Image.BILINEAR)
                    )
                data = resampled
            else:
                data = np.array(Image.fromarray(data).resize((new_w, new_h), Image.BILINEAR))

        # Recalculer le transform pour le sous-jeu
        from rasterio.windows import transform as win_transform

        out_transform = win_transform(window, src.transform)
        # Si on a resamplé, scale l'affine
        new_shape = data.shape[-2:]
        if max(h, w_pix) > max_pixels_per_side:
            out_transform = out_transform * out_transform.scale(
                w_pix / new_shape[1], h / new_shape[0]
            )

        with rasterio.open(
            out_path, "w",
            driver="GTiff",
            height=new_shape[0],
            width=new_shape[1],
            count=n_b,
            dtype=data.dtype,
            crs=src.crs,
            transform=out_transform,
            compress="deflate",
        ) as dst:
            if data.ndim == 3:
                dst.write(data)
            else:
                dst.write(data, 1)
    print(f"  [hunt] -> {out_path.relative_to(C.ROOT)} ({new_shape[1]}x{new_shape[0]} px, {n_b}-band)")

    # Preview JPG pour la demo (toujours utile pour les slides)
    if save_preview:
        try:
            from PIL import Image
            import numpy as np

            if data.ndim == 3:
                # Stretch chaque bande sur [0, 255]
                preview = np.zeros((new_shape[0], new_shape[1], 3), dtype=np.uint8)
                for i, b in enumerate([0, 1, 2][:n_b]):
                    band = data[b].astype("float32")
                    lo, hi = np.percentile(band, [2, 98])
                    band = np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1)
                    preview[..., i] = (band * 255).astype("uint8")
            else:
                preview = _sar_to_uint8(data)
            jpg_path = out_path.with_suffix(".preview.jpg")
            if preview.ndim == 3:
                Image.fromarray(preview).save(jpg_path, quality=85)
            else:
                Image.fromarray(preview).save(jpg_path, quality=85)
            print(f"  [hunt] -> preview {jpg_path.relative_to(C.ROOT)}")
        except Exception as e:  # noqa: BLE001
            print(f"  [hunt] preview saute : {type(e).__name__}: {e}")

    return out_path


def _sar_to_uint8(arr):
    """Normalise un raster SAR (float32 backscatter linéaire) vers uint8 8-bit
    par percentile-clipping 2-98 % puis échelle log (dB). Robuste aux outliers
    et aux speculaires côtiers.
    """
    import numpy as np

    a = arr.astype("float32")
    # Sentinel-1 RTC : valeurs en backscatter linéaire (~[0, 1+]) ; passer en dB
    a = np.where(a > 0, 10 * np.log10(a + 1e-6), -40.0)
    p2, p98 = np.percentile(a[a > -40], [2, 98]) if (a > -40).any() else (-30, 0)
    a = np.clip((a - p2) / max(p98 - p2, 1e-6), 0, 1)
    return (a * 255).astype("uint8")


def detect_ships_on_scene(
    geotiff_path: Path,
    scene_id: str | None = None,
    yolo_weights: str = "yolov8n.pt",
    tile_size: int = 640,
    overlap: int = 128,
    conf: float = C.YOLO_CONF_SAR,
    min_ship_px: int = 4,
    upscale: int = 2,
) -> pd.DataFrame:
    """Inférence YOLO tuilée sur une scène GeoTIFF.

    YOLO COCO ne connaît que `boat`/`ship` → on filtre sur ces classes.
    Pour distinguer frégate/destroyer/etc., il faut un YOLO fine-tuné xView3
    (cf. `p2_detection.train_yolo_on_substituted`, exécution Colab).

    Args:
        geotiff_path: scène locale (sortie de `fetch_scene_window`).
        scene_id: pour traçabilité dans le DataFrame final.
        yolo_weights: poids YOLO (`yolov8n.pt` par défaut, COCO).
        tile_size: côté de la tuile (1024 par défaut).
        overlap: chevauchement entre tuiles (128 px).
        conf: seuil de confiance YOLO (SAR : 0.30 par défaut, plus permissif
              que l'optique parce que le contraste est moins bon).
        min_ship_px: rejette les bboxes < ce nombre de pixels (épaves, bruit).

    Returns:
        DataFrame `[detection_id, scene_id, lat, lon, class_name, confidence,
                    bbox_px_x1, bbox_px_y1, bbox_px_x2, bbox_px_y2]`.
    """
    import numpy as np
    import rasterio
    from PIL import Image
    from ultralytics import YOLO

    geotiff_path = Path(geotiff_path)
    model = YOLO(yolo_weights)
    rows = []

    with rasterio.open(geotiff_path) as src:
        transform = src.transform
        crs = src.crs

        if src.count >= 3:
            # Optique (Sentinel-2 visual) : prendre les 3 bandes en RGB directement.
            bands = [src.read(i + 1) for i in range(3)]
            from PIL import Image as _PILImage
            rgb = np.zeros((bands[0].shape[0], bands[0].shape[1], 3), dtype="uint8")
            for i, b in enumerate(bands):
                b = b.astype("float32")
                lo, hi = np.percentile(b, [2, 98])
                b = np.clip((b - lo) / max(hi - lo, 1e-6), 0, 1)
                rgb[..., i] = (b * 255).astype("uint8")
            img8 = rgb  # (H, W, 3)
        else:
            arr = src.read(1)
            img8 = _sar_to_uint8(arr)  # (H, W) uint8

        # Upscale : rend les petits navires (10-30 px à 10 m/pix) plus visibles
        # pour YOLO COCO qui n'a pas vu beaucoup d'aerial top-down.
        if upscale and upscale > 1:
            from PIL import Image as _PIL
            if img8.ndim == 3:
                h0, w0 = img8.shape[:2]
                pil = _PIL.fromarray(img8)
                pil = pil.resize((w0 * upscale, h0 * upscale), _PIL.BICUBIC)
                img8 = np.asarray(pil)
            else:
                h0, w0 = img8.shape
                pil = _PIL.fromarray(img8)
                pil = pil.resize((w0 * upscale, h0 * upscale), _PIL.BICUBIC)
                img8 = np.asarray(pil)

        if img8.ndim == 3:
            H, W = img8.shape[:2]
        else:
            H, W = img8.shape
        stride = tile_size - overlap

        det_idx = 0
        for y0 in range(0, max(1, H - overlap), stride):
            for x0 in range(0, max(1, W - overlap), stride):
                y1 = min(y0 + tile_size, H)
                x1 = min(x0 + tile_size, W)
                if img8.ndim == 3:
                    tile = img8[y0:y1, x0:x1, :]
                else:
                    tile = img8[y0:y1, x0:x1]
                if tile.shape[0] < 32 or tile.shape[1] < 32:
                    continue
                # YOLO veut RGB — pour le SAR on duplique la bande
                tile_rgb = tile if tile.ndim == 3 else np.stack([tile, tile, tile], axis=-1)
                r = model.predict(
                    source=tile_rgb,
                    device="cpu",
                    conf=conf,
                    imgsz=tile_size,
                    verbose=False,
                    save=False,
                )[0]
                boxes = r.boxes
                names = model.names
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    cn = names[cls_id]
                    if cn not in {"boat", "ship", "airplane"}:  # airplane parfois sur petits navires SAR
                        continue
                    bx1, by1, bx2, by2 = boxes.xyxy[i].tolist()
                    w_px = bx2 - bx1
                    h_px = by2 - by1
                    if max(w_px, h_px) < min_ship_px:
                        continue
                    # Coordonnées globales dans la scène (espace upscalé)
                    gx1, gy1 = bx1 + x0, by1 + y0
                    gx2, gy2 = bx2 + x0, by2 + y0
                    cx, cy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
                    # Reproj pixel -> CRS source -> WGS84.
                    # Si on a upscalé, ramener cx/cy en pixels originaux pour appliquer transform.
                    cx_src = cx / max(upscale, 1)
                    cy_src = cy / max(upscale, 1)
                    sx, sy = transform * (cx_src, cy_src)
                    if str(crs).upper() != "EPSG:4326":
                        from rasterio.warp import transform as warp_transform

                        xs, ys = warp_transform(crs, "EPSG:4326", [sx], [sy])
                        lon_, lat_ = xs[0], ys[0]
                    else:
                        lon_, lat_ = sx, sy
                    det_idx += 1
                    rows.append(
                        {
                            "detection_id": f"HUNT-{scene_id or geotiff_path.stem}-{det_idx:05d}",
                            "scene_id": scene_id or geotiff_path.stem,
                            "lat": lat_,
                            "lon": lon_,
                            "class_name": cn,
                            "confidence": round(float(boxes.conf[i].item()), 3),
                            "bbox_px_x1": round(gx1, 1),
                            "bbox_px_y1": round(gy1, 1),
                            "bbox_px_x2": round(gx2, 1),
                            "bbox_px_y2": round(gy2, 1),
                        }
                    )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = _nms_global(df, iou_thresh=0.3)
    return df


def _nms_global(df: pd.DataFrame, iou_thresh: float = 0.3) -> pd.DataFrame:
    """NMS global sur les détections (les tuiles se chevauchent → doublons).

    IoU calculée en pixels image (coordonnées globales bbox_px_*).
    """
    if df.empty:
        return df
    boxes = df[["bbox_px_x1", "bbox_px_y1", "bbox_px_x2", "bbox_px_y2"]].to_numpy()
    scores = df["confidence"].to_numpy()
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        x1 = boxes[rest, 0].clip(min=boxes[i, 0])
        y1 = boxes[rest, 1].clip(min=boxes[i, 1])
        x2 = boxes[rest, 2].clip(max=boxes[i, 2])
        y2 = boxes[rest, 3].clip(max=boxes[i, 3])
        inter = ((x2 - x1).clip(min=0)) * ((y2 - y1).clip(min=0))
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-9)
        order = rest[iou < iou_thresh]
    return df.iloc[keep].reset_index(drop=True)


# ----------------------------------------------------------------------
# Détecteur SAR classique (CFAR — Cell-Averaging) : pas de ML, pas de Colab.
# Méthode de référence pour la détection de navires sur SAR (cible brillante sur
# mer sombre). Fonctionne immédiatement sur Sentinel-1 RTC.

def detect_ships_cfar(
    geotiff_path: Path,
    scene_id: str | None = None,
    *,
    k_sigma: float = 3.5,
    guard_px: int = 5,
    bg_px: int = 25,
    min_area_px: int = 3,
    max_area_px: int = 1500,
    sea_pct: float = 55.0,
    water_radius_px: int = 25,
    water_frac_thr: float = 0.65,
) -> pd.DataFrame:
    """Détection de cibles brillantes sur scène SAR par CFAR à moyenne de cellules.

    Pour chaque pixel : fond local = moyenne/écart-type de l'anneau (fenêtre `bg_px`
    privée de la zone de garde `guard_px`). Détection si `pixel_dB > μ_fond + k·σ_fond`.
    Puis composantes connexes → blobs, filtrés par aire (en pixels) ; on rejette aussi
    les blobs dont le fond local est trop clair (= structure portuaire, pas un navire
    sur mer sombre — heuristique `sea_pct`).

    Args:
        k_sigma: facteur CFAR (3-5 typique ; 4.5 = ~conservateur).
        guard_px / bg_px: demi-côtés des fenêtres de garde et de fond.
        min_area_px / max_area_px: filtre de taille de blob.
        sea_pct: un blob est gardé si le fond local médian < percentile `sea_pct`
                 de toute la scène (i.e. le blob est entouré de mer, pas de terre).

    Returns:
        DataFrame `[detection_id, scene_id, lat, lon, class_name, confidence,
                    bbox_px_x1..y2]` — `class_name='sar_target'`, `confidence`
                    = SNR normalisé du blob (∈ [0,1]).
    """
    import numpy as np
    import rasterio
    from scipy import ndimage

    geotiff_path = Path(geotiff_path)
    rows = []
    with rasterio.open(geotiff_path) as src:
        transform = src.transform
        crs = src.crs
        arr = src.read(1).astype("float32")

    # Backscatter linéaire -> dB (robuste). Pixels nuls/négatifs -> plancher.
    db = np.where(arr > 0, 10.0 * np.log10(arr + 1e-6), -40.0)
    valid = db > -40.0
    if valid.sum() < 100:
        return pd.DataFrame()
    sea_thr = np.percentile(db[valid], sea_pct)

    # Moyenne/variance locales sur l'ANNEAU = (fenêtre fond) - (fenêtre garde),
    # calculées par filtres de boîte (O(N), pas O(N·k²)).
    def _box_mean(x, half):
        return ndimage.uniform_filter(x, size=2 * half + 1, mode="nearest")

    n_bg = (2 * bg_px + 1) ** 2
    n_gd = (2 * guard_px + 1) ** 2
    n_ring = max(n_bg - n_gd, 1)
    s_bg = _box_mean(db, bg_px) * n_bg
    s_gd = _box_mean(db, guard_px) * n_gd
    ring_mean = (s_bg - s_gd) / n_ring
    s2_bg = _box_mean(db * db, bg_px) * n_bg
    s2_gd = _box_mean(db * db, guard_px) * n_gd
    ring_var = np.maximum((s2_bg - s2_gd) / n_ring - ring_mean ** 2, 1e-6)
    ring_std = np.sqrt(ring_var)

    thr = ring_mean + k_sigma * ring_std
    mask = (db > thr) & valid

    # Carte "eau" approximative : pixel sombre (sous le seuil mer). Sert à classer
    # chaque détection : entourée d'eau (= navire) vs proche de structures (= quai/digue).
    water_map = (db <= sea_thr).astype("float32")
    water_frac = ndimage.uniform_filter(
        water_map, size=2 * water_radius_px + 1, mode="nearest"
    )

    labels, n = ndimage.label(mask)
    if n == 0:
        return pd.DataFrame()
    objs = ndimage.find_objects(labels)
    snr_max = float(np.percentile((db - ring_mean) / ring_std, 99.9)) or 1.0

    det_idx = 0
    for lab_id, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        ys, xs = sl
        sub = labels[ys, xs] == lab_id
        area = int(sub.sum())
        if area < min_area_px or area > max_area_px:
            continue
        # Compacité : un navire est un blob compact, pas une ligne (digue, quai).
        bbox_area = max((ys.stop - ys.start) * (xs.stop - xs.start), 1)
        compactness = area / bbox_area
        if compactness < 0.30:
            continue
        # Position du pixel le plus brillant du blob (= "pic" de la cible)
        sub_db = np.where(sub, db[ys, xs], -1e9)
        local_max_idx = np.unravel_index(np.argmax(sub_db), sub_db.shape)
        py = ys.start + local_max_idx[0]
        px = xs.start + local_max_idx[1]
        # Heuristique mer : le fond local autour du blob doit être sombre.
        if ring_mean[py, px] > sea_thr:
            continue
        wf = float(water_frac[py, px])
        target_kind = "vessel_open_water" if wf >= water_frac_thr else "harbor_target"
        snr = float((db[py, px] - ring_mean[py, px]) / max(ring_std[py, px], 1e-6))
        conf = float(np.clip(snr / max(snr_max, 1e-6), 0.0, 1.0))
        # Reproj pixel -> CRS source -> WGS84.
        sx, sy = transform * (px + 0.5, py + 0.5)
        if str(crs).upper() != "EPSG:4326":
            from rasterio.warp import transform as warp_transform

            lons, lats = warp_transform(crs, "EPSG:4326", [sx], [sy])
            lon_, lat_ = lons[0], lats[0]
        else:
            lon_, lat_ = sx, sy
        det_idx += 1
        rows.append(
            {
                "detection_id": f"CFAR-{scene_id or geotiff_path.stem}-{det_idx:05d}",
                "scene_id": scene_id or geotiff_path.stem,
                "lat": float(lat_),
                "lon": float(lon_),
                "class_name": "sar_target",
                "target_kind": target_kind,
                "water_frac": round(wf, 3),
                "confidence": round(conf, 3),
                "snr_db": round(snr, 2),
                "area_px": area,
                "bbox_px_x1": float(xs.start),
                "bbox_px_y1": float(ys.start),
                "bbox_px_x2": float(xs.stop),
                "bbox_px_y2": float(ys.stop),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = _nms_global(df, iou_thresh=0.2)
    return df


# ----------------------------------------------------------------------
# Boucle complète sur une base : query → fetch fenêtre → inférence

_DETECTORS = {"yolo", "cfar"}


def hunt_base(
    base_name: str,
    bbox: tuple[float, float, float, float] | None = None,
    max_scenes: int = 2,
    window_days: int = C.HUNT_WINDOW_DAYS,
    yolo_weights: str = "yolov8n.pt",
    collection: str = _COLLECTION_S1,
    asset_key: str | None = None,
    max_cloud_pct: float = 30.0,
    detector: str | None = None,
) -> pd.DataFrame:
    """Pipeline complet sur 1 base : query → download window → détection → DataFrame.

    Args:
        base_name: clé de `C.BASES_NAVALES` (ex. "Toulon (FR)") ou autre.
        bbox: si None, lu depuis `C.BASES_NAVALES[base_name]`.
        max_scenes: nombre max de scènes à traiter (récentes d'abord).
        collection: 'sentinel-1-rtc' (SAR, défaut) ou 'sentinel-2-l2a' (optique).
        asset_key: nom de l'asset STAC ; auto : 'vv' pour S1, 'visual' pour S2.
        max_cloud_pct: ignore les scènes Sentinel-2 trop nuageuses.
        detector: 'cfar' (CFAR SAR classique — défaut pour Sentinel-1) ou 'yolo'
                  (YOLO ; `yolo_weights` COCO par défaut — défaut pour Sentinel-2).
                  Le CFAR ne dépend d'aucun modèle entraîné → marche tout de suite
                  sur SAR ; YOLO COCO sur optique 10 m est faible (cf. diagnostic).

    Returns:
        DataFrame agrégé de toutes les détections sur cette base.
    """
    bbox = bbox or C.BASES_NAVALES[base_name]
    asset_key = asset_key or ("visual" if collection == _COLLECTION_S2 else "vv")
    conf = C.YOLO_CONF_OPTIQUE if collection == _COLLECTION_S2 else C.YOLO_CONF_SAR
    if detector is None:
        detector = "cfar" if collection == _COLLECTION_S1 else "yolo"
    if detector not in _DETECTORS:
        raise ValueError(f"detector doit être dans {_DETECTORS}, reçu {detector!r}")
    print(f"\n[hunt] === {base_name} bbox={bbox} collection={collection} detector={detector} ===")
    df_scenes = query_sentinel1(
        base_name, bbox, window_days=window_days, collection=collection
    )
    if df_scenes.empty:
        print(f"  Aucune scene {collection} recente.")
        return pd.DataFrame()
    # Filtre nuages pour Sentinel-2
    if collection == _COLLECTION_S2 and "cloud_cover" in df_scenes.columns:
        df_scenes = df_scenes[
            df_scenes["cloud_cover"].fillna(100) <= max_cloud_pct
        ]
        if df_scenes.empty:
            print(f"  Toutes les scenes ont cloud_cover > {max_cloud_pct}% -> aucune utilisable.")
            return pd.DataFrame()

    df_scenes = df_scenes.head(max_scenes)
    all_dets = []
    for _, sc in df_scenes.iterrows():
        scene_id = sc["scene_id"]
        try:
            tif = fetch_scene_window(
                scene_id, bbox, asset_key=asset_key, collection=collection,
            )
            if detector == "cfar":
                df_d = detect_ships_cfar(tif, scene_id=scene_id)
            else:
                df_d = detect_ships_on_scene(
                    tif, scene_id=scene_id, yolo_weights=yolo_weights, conf=conf
                )
            df_d["base"] = base_name
            df_d["scene_datetime"] = sc["datetime"]
            df_d["collection"] = collection
            df_d["detector"] = detector
            print(f"  {scene_id} -> {len(df_d)} detections ({detector}) apres NMS")
            all_dets.append(df_d)
        except Exception as e:  # noqa: BLE001
            print(f"  {scene_id} -> ERREUR : {type(e).__name__}: {e}")

    if not all_dets:
        return pd.DataFrame()
    df = pd.concat(all_dets, ignore_index=True)
    short = base_name.split()[0].lower().replace("(", "")
    suffix = "s2" if collection == _COLLECTION_S2 else "s1"
    out_path = C.DATA_PROCESSED / f"chasse_{short}_{suffix}.csv"
    df.to_csv(out_path, index=False)
    print(f"  -> ecrit {out_path.relative_to(C.ROOT)} ({len(df)} detections)")
    return df


def hunt_all_bases(
    max_scenes_per_base: int = 1,
    collection: str = _COLLECTION_S1,
    detector: str | None = None,
) -> pd.DataFrame:
    """Boucle hunt_base sur les 5 bases. Coupe à `max_scenes_per_base` pour la démo."""
    all_dfs = []
    for name, bbox in C.BASES_NAVALES.items():
        try:
            df = hunt_base(
                name, bbox, max_scenes=max_scenes_per_base,
                collection=collection, detector=detector,
            )
            if not df.empty:
                all_dfs.append(df)
        except Exception as e:  # noqa: BLE001
            print(f"  [hunt] {name} -> ERREUR : {type(e).__name__}: {e}")
    if not all_dfs:
        return pd.DataFrame()
    df_all = pd.concat(all_dfs, ignore_index=True)
    suffix = "s2" if collection == _COLLECTION_S2 else "s1"
    out_path = C.DATA_PROCESSED / f"chasse_toutes_bases_{suffix}.csv"
    df_all.to_csv(out_path, index=False)
    by_base = df_all.groupby("base").size().to_dict()
    print(f"\n[hunt] -> {out_path.relative_to(C.ROOT)} ({len(df_all)} detections globales)")
    print(f"[hunt] par base : {by_base}")
    return df_all


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if args and args[0] == "real":
        collection = _COLLECTION_S2 if "s2" in args else _COLLECTION_S1
        detector = "yolo" if "yolo" in args else ("cfar" if "cfar" in args else None)
        print(f"[hunt] Chasse complete (download + détection) sur {collection}")
        hunt_all_bases(max_scenes_per_base=1, collection=collection, detector=detector)
    elif args and args[0] == "toulon":
        # Test rapide : 1 base, optique (Sentinel-2), YOLO
        hunt_base("Toulon (FR)", max_scenes=1, collection=_COLLECTION_S2)
    elif args and args[0] in {"sebastopol", "sévastopol", "sébastopol", "brest", "norfolk", "severomorsk"}:
        # Test rapide : 1 base, SAR (Sentinel-1), CFAR. Match accent-insensible.
        import unicodedata as _ud

        def _norm(s):
            return "".join(c for c in _ud.normalize("NFD", s.lower()) if c.isalnum())

        target = _norm(args[0])
        key = next((k for k in C.BASES_NAVALES if _norm(k).startswith(target)), None)
        if key is None:
            print(f"Base inconnue: {args[0]}. Dispo: {list(C.BASES_NAVALES)}")
            raise SystemExit(1)
        hunt_base(key, max_scenes=1, collection=_COLLECTION_S1, detector="cfar")
    else:
        # Mode défaut : juste la query STAC (rapide, sans download)
        print(f"[hunt] Query Sentinel-1 RTC sur {len(C.BASES_NAVALES)} bases "
              f"({C.HUNT_WINDOW_DAYS} derniers jours)...")
        print("       Pour la chasse reelle (download + detection) :")
        print("         python -m src.hunt real          # SAR (Sentinel-1) + CFAR (defaut)")
        print("         python -m src.hunt real cfar     # idem, explicite")
        print("         python -m src.hunt real s2       # Optique (Sentinel-2) + YOLO")
        print("         python -m src.hunt sebastopol    # Test rapide 1 base SAR + CFAR")
        print("         python -m src.hunt toulon        # Test rapide 1 base Sentinel-2 + YOLO")
        survey_all_bases()
