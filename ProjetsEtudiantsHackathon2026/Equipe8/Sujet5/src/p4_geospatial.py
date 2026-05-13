"""P4 — Analyse géospatiale & temporelle (Q10, Q11, Q12).

⚠️ Le piège central : `military_zones.zone_id = MIL-XXX` ne correspond PAS
à `images_metadata.zone_id = ZONE-XXX`. → JOINTURE PAR DISTANCE, pas par id.
Voir hypotheses.md §B Q10. Rayon par défaut : `config.MIL_ZONE_RADIUS_KM`.
"""

from __future__ import annotations

import pandas as pd
from geopy.distance import geodesic

from . import config as C
from . import load as L


# ----------------------------------------------------------------------
# Q10 — Jointure détections × zones militaires par distance géodésique

def annotate_detections_with_zones(
    df_det_with_coords: pd.DataFrame,
    df_zones: pd.DataFrame,
    radius_km: float = C.MIL_ZONE_RADIUS_KM,
) -> pd.DataFrame:
    """Pour chaque détection, calcule la zone militaire active la plus proche.

    Args:
        df_det_with_coords: DataFrame de détections avec colonnes `lat`, `lon`.
        df_zones: `military_zones.csv` chargé avec `lat`, `lon`, `active_bool`.
        radius_km: seuil de proximité.

    Renvoie le DataFrame d'entrée enrichi des colonnes :
        - `nearest_mil_zone_id`, `nearest_mil_zone_name`, `nearest_mil_zone_risk`
        - `nearest_mil_zone_dist_km`
        - `in_military_zone` (bool) = active ET distance < radius_km
    """
    zones_active = df_zones[df_zones["active_bool"]].copy()
    out = df_det_with_coords.copy()

    nearest_id, nearest_name, nearest_risk, nearest_dist, in_zone = [], [], [], [], []
    for lat, lon in zip(out["lat"], out["lon"], strict=False):
        if pd.isna(lat) or pd.isna(lon):
            nearest_id.append(None); nearest_name.append(None)
            nearest_risk.append(None); nearest_dist.append(None)
            in_zone.append(False)
            continue
        best_d, best_row = float("inf"), None
        for _, z in zones_active.iterrows():
            d = geodesic((lat, lon), (z["lat"], z["lon"])).km
            if d < best_d:
                best_d = d
                best_row = z
        nearest_id.append(best_row["zone_id"] if best_row is not None else None)
        nearest_name.append(best_row["name"] if best_row is not None else None)
        nearest_risk.append(best_row["risk_level"] if best_row is not None else None)
        nearest_dist.append(best_d if best_row is not None else None)
        in_zone.append(best_d < radius_km if best_row is not None else False)

    out["nearest_mil_zone_id"] = nearest_id
    out["nearest_mil_zone_name"] = nearest_name
    out["nearest_mil_zone_risk"] = nearest_risk
    out["nearest_mil_zone_dist_km"] = nearest_dist
    out["in_military_zone"] = in_zone
    return out


def q10(df_det_in_zones: pd.DataFrame) -> dict:
    """Combien de militaires détectés en zone militaire ?"""
    mask_mil = df_det_in_zones["is_military"] & df_det_in_zones["in_military_zone"]
    return {
        "n_militaires_en_zone": int(mask_mil.sum()),
        "n_militaires_total": int(df_det_in_zones["is_military"].sum()),
        "pct_militaires_en_zone": round(
            mask_mil.sum() / max(df_det_in_zones["is_military"].sum(), 1) * 100, 2
        ),
        "_note": (
            f"Jointure par distance géodésique (rayon {C.MIL_ZONE_RADIUS_KM} km) "
            "et NON par zone_id : les deux référentiels (`MIL-XXX` vs `ZONE-XXX`) "
            "ne se joignent pas."
        ),
    }


# ----------------------------------------------------------------------
# Q11 — Temporel par zone

def q11_temporal(df_det_in_zones: pd.DataFrame) -> dict:
    """Évolution du nombre de détections par zone, détection de pics (> µ + 2σ)."""
    df = df_det_in_zones.dropna(subset=["nearest_mil_zone_name", "timestamp"]).copy()
    if df.empty:
        return {"series": {}, "peaks": {}, "_warning": "Aucune détection datée."}
    df["date"] = df["timestamp"].dt.date
    series = (
        df.groupby(["nearest_mil_zone_name", "date"])
        .size()
        .unstack(fill_value=0)
        .sort_index(axis=1)
    )
    peaks = {}
    for zone in series.index:
        s = series.loc[zone]
        mu, sigma = s.mean(), s.std()
        threshold = mu + 2 * sigma if sigma > 0 else mu + 1
        peak_dates = s[s > threshold].index.tolist()
        peaks[zone] = [str(d) for d in peak_dates]
    return {"series": series, "peaks": peaks}


# ----------------------------------------------------------------------
# Q12 — Anomalies géospatiales

def q12_anomalies(df_det_in_zones: pd.DataFrame) -> dict:
    """
    - Type A : militaire détecté HORS zone militaire (transit légitime, mais à surveiller).
    - Type B : civil détecté EN zone militaire à risque High/Critical (suspect).
    """
    type_a = df_det_in_zones[
        df_det_in_zones["is_military"] & ~df_det_in_zones["in_military_zone"]
    ].copy()
    type_b = df_det_in_zones[
        ~df_det_in_zones["is_military"]
        & df_det_in_zones["in_military_zone"]
        & df_det_in_zones["nearest_mil_zone_risk"].isin(["High", "Critical"])
    ].copy()
    return {
        "n_type_a_mil_hors_zone": len(type_a),
        "n_type_b_civil_en_zone_critique": len(type_b),
        "df_type_a": type_a,
        "df_type_b": type_b,
    }


# ----------------------------------------------------------------------
# Q12-bis — Anomalie type C : proximité d'une infrastructure critique
# sous-marine (câbles télécom/électriques, gazoducs/oléoducs).
# Précédents : Nord Stream (2022), Eagle S × EstLink (2024).

def annotate_detections_with_critical_infra(
    df_det_with_coords: pd.DataFrame,
    radius_km: float = C.INFRA_SEARCH_RADIUS_KM,
    *,
    enrich: bool = True,
) -> pd.DataFrame:
    """Pour chaque détection, calcule l'infra critique sous-marine la plus proche.

    Ajoute les colonnes :
        - `nearest_infra_kind` (str | None) — type lisible (Gazoduc, Câble télécom…)
        - `nearest_infra_name` (str | None)
        - `nearest_infra_dist_km` (float | None)

    Args:
        enrich: si False, ne fait aucun appel Overpass (utile pour les tests
                ou les runs offline) → toutes les colonnes restent à None.

    Note : Overpass est appelé une fois par détection mais avec cache SQLite
    persistant (`data/cache.sqlite`), donc les détections groupées par scène
    sont essentiellement gratuites au second passage.
    """
    out = df_det_with_coords.copy()
    out["nearest_infra_kind"] = None
    out["nearest_infra_name"] = None
    out["nearest_infra_dist_km"] = None
    if not enrich:
        return out

    from . import osint_enrich

    for i, row in out.iterrows():
        lat, lon = row.get("lat"), row.get("lon")
        if pd.isna(lat) or pd.isna(lon):
            continue
        try:
            inf = osint_enrich.nearest_submarine_infra(
                float(lat), float(lon), radius_km=radius_km
            )
        except Exception:  # noqa: BLE001 — un échec API ne casse pas le pipeline
            continue
        if inf is None:
            continue
        out.at[i, "nearest_infra_kind"] = inf["kind"]
        out.at[i, "nearest_infra_name"] = inf["name"]
        out.at[i, "nearest_infra_dist_km"] = inf["distance_km"]
    return out


def q12_anomalies_type_c(
    df_det_with_infra: pd.DataFrame,
    threshold_km: float = C.INFRA_DIST_KM,
) -> pd.DataFrame:
    """Type C : détection à `< threshold_km` d'une infrastructure critique
    sous-marine. Hypothèse opérationnelle : tout navire qui s'approche autant
    d'un câble ou d'un pipeline mérite enquête (cf. précédents Nord Stream,
    Eagle S). On n'exige PAS qu'il soit militaire — le risque vient justement
    des navires civils utilisés en proxy.

    Args:
        df_det_with_infra: DataFrame déjà passé dans
            `annotate_detections_with_critical_infra`.
        threshold_km: distance d'alerte (défaut `C.INFRA_DIST_KM`).

    Returns:
        Sous-DataFrame trié par distance croissante.
    """
    if "nearest_infra_dist_km" not in df_det_with_infra.columns:
        return df_det_with_infra.iloc[0:0].copy()
    mask = df_det_with_infra["nearest_infra_dist_km"].notna() & (
        df_det_with_infra["nearest_infra_dist_km"] < threshold_km
    )
    return df_det_with_infra[mask].copy().sort_values("nearest_infra_dist_km")


# ----------------------------------------------------------------------
# Run P4 tout-en-un

def run_p4(df_det: pd.DataFrame | None = None) -> dict:
    """Charge les détections + zones et calcule Q10-Q12 + carte.

    Les détections n'ont pas de `lat/lon` explicite — on hérite de la zone via
    `images_metadata` (un détection est dans la même `coordinates` que son image).
    """
    df_meta = L.load_images_metadata()
    df_zones = L.load_military_zones()
    if df_det is None:
        df_det = L.load_detections()

    # Attribution lat/lon à chaque détection via l'image
    img_coords = df_meta[["image_id", "lat", "lon", "date", "time"]].rename(
        columns={"lat": "img_lat", "lon": "img_lon"}
    )
    df_det = df_det.merge(img_coords, on="image_id", how="left")
    df_det["lat"] = df_det["img_lat"]
    df_det["lon"] = df_det["img_lon"]

    df_det_z = annotate_detections_with_zones(df_det, df_zones)

    return {
        "df_det_with_zones": df_det_z,
        "df_zones": df_zones,
        "Q10": q10(df_det_z),
        "Q11": q11_temporal(df_det_z),
        "Q12": q12_anomalies(df_det_z),
    }
