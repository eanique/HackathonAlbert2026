"""Helpers folium — couches navires / zones / sombres / alertes.

Convention de couleur :
- Militaires en zone militaire (légitime) : bleu marine
- Militaires hors zone (transit) : orange
- Civils en zone militaire critique (suspect) : rouge
- Civils hors zone : vert clair
- Zones militaires actives : cercle violet (rayon = MIL_ZONE_RADIUS_KM)
- Navires "sombres" (sat oui, AIS non) : rouge pulsant
"""

from __future__ import annotations

from pathlib import Path

import folium
import pandas as pd
from folium.plugins import MarkerCluster

from . import config as C


def _color_for(row: pd.Series) -> str:
    if row["is_military"] and row["in_military_zone"]:
        return "darkblue"
    if row["is_military"] and not row["in_military_zone"]:
        return "orange"
    if (
        not row["is_military"]
        and row["in_military_zone"]
        and row.get("nearest_mil_zone_risk") in {"High", "Critical"}
    ):
        return "red"
    return "lightgreen"


def detections_map(
    df_det_with_zones: pd.DataFrame,
    df_zones: pd.DataFrame,
    out_path: Path | None = None,
    center: tuple[float, float] | None = None,
) -> folium.Map:
    """Carte folium des détections + zones militaires actives."""
    df = df_det_with_zones.dropna(subset=["lat", "lon"]).copy()

    if center is None:
        if not df.empty:
            center = (df["lat"].mean(), df["lon"].mean())
        else:
            center = (20.0, 0.0)

    fmap = folium.Map(location=center, zoom_start=2, tiles="cartodbpositron")

    # 1) Zones militaires actives (cercles)
    zones_layer = folium.FeatureGroup(name="Zones militaires (actives)").add_to(fmap)
    for _, z in df_zones[df_zones["active_bool"]].iterrows():
        if pd.isna(z["lat"]) or pd.isna(z["lon"]):
            continue
        folium.Circle(
            location=(z["lat"], z["lon"]),
            radius=C.MIL_ZONE_RADIUS_KM * 1000,  # mètres
            color="purple",
            weight=1,
            fill=True,
            fill_opacity=0.05,
            tooltip=(
                f"{z['name']} · {z['country']} · risk={z['risk_level']}"
            ),
        ).add_to(zones_layer)

    # 2) Détections clusterisées
    cluster = MarkerCluster(name="Détections").add_to(fmap)
    for _, r in df.iterrows():
        color = _color_for(r)
        popup_html = (
            f"<b>{r.get('category', '?')}</b><br>"
            f"Image : {r.get('file_name', '?')}<br>"
            f"Confidence : {r.get('confidence', '?')}<br>"
            f"Militaire : {r.get('is_military')}<br>"
            f"En zone militaire : {r.get('in_military_zone')}<br>"
            f"Zone proche : {r.get('nearest_mil_zone_name', '?')} "
            f"({r.get('nearest_mil_zone_dist_km', '?'):.1f} km, "
            f"risk={r.get('nearest_mil_zone_risk', '?')})"
        )
        folium.CircleMarker(
            location=(r["lat"], r["lon"]),
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=320),
        ).add_to(cluster)

    folium.LayerControl(collapsed=False).add_to(fmap)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fmap.save(str(out_path))
    return fmap


def anomalies_map(
    df_type_a: pd.DataFrame,
    df_type_b: pd.DataFrame,
    df_zones: pd.DataFrame,
    out_path: Path | None = None,
) -> folium.Map:
    """Carte folium des anomalies Q12 uniquement (militaires hors zone + civils en zone critique)."""
    all_pts = pd.concat([df_type_a, df_type_b], ignore_index=True).dropna(
        subset=["lat", "lon"]
    )
    center = (
        (all_pts["lat"].mean(), all_pts["lon"].mean())
        if not all_pts.empty
        else (20.0, 0.0)
    )
    fmap = folium.Map(location=center, zoom_start=2, tiles="cartodbpositron")

    for _, z in df_zones[df_zones["active_bool"]].iterrows():
        if pd.isna(z["lat"]) or pd.isna(z["lon"]):
            continue
        folium.Circle(
            location=(z["lat"], z["lon"]),
            radius=C.MIL_ZONE_RADIUS_KM * 1000,
            color="purple",
            weight=1,
            fill=False,
            tooltip=z["name"],
        ).add_to(fmap)

    a_layer = folium.FeatureGroup(name="Type A · militaire hors zone").add_to(fmap)
    for _, r in df_type_a.iterrows():
        if pd.isna(r["lat"]):
            continue
        folium.Marker(
            location=(r["lat"], r["lon"]),
            icon=folium.Icon(color="orange", icon="ship", prefix="fa"),
            popup=f"{r.get('category', '?')} militaire hors zone",
        ).add_to(a_layer)

    b_layer = folium.FeatureGroup(name="Type B · civil en zone critique").add_to(fmap)
    for _, r in df_type_b.iterrows():
        if pd.isna(r["lat"]):
            continue
        folium.Marker(
            location=(r["lat"], r["lon"]),
            icon=folium.Icon(color="red", icon="warning", prefix="fa"),
            popup=(
                f"{r.get('category', '?')} civil en zone "
                f"{r.get('nearest_mil_zone_name', '?')} "
                f"(risk={r.get('nearest_mil_zone_risk', '?')})"
            ),
        ).add_to(b_layer)

    folium.LayerControl(collapsed=False).add_to(fmap)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fmap.save(str(out_path))
    return fmap
