"""Helpers folium pour la carte de démo (Sujet 3, Q8 + démo finale)."""

from __future__ import annotations

from pathlib import Path

import folium
import pandas as pd

from .config import OUTPUTS


# Centre par défaut : Atlantique nord — vue large.
DEFAULT_CENTER = (45.0, -10.0)
DEFAULT_ZOOM = 4


def base_map(center: tuple[float, float] = DEFAULT_CENTER,
             zoom_start: int = DEFAULT_ZOOM) -> folium.Map:
    """Carte de base avec OpenStreetMap + couche océanique."""
    m = folium.Map(location=center, zoom_start=zoom_start, tiles="OpenStreetMap")
    folium.TileLayer("CartoDB Positron").add_to(m)
    return m


def layer_position_mismatches(m: folium.Map, df: pd.DataFrame) -> folium.FeatureGroup:
    """Couche `lines` : trait AIS↔radio + marqueurs aux deux extrémités."""
    fg = folium.FeatureGroup(name="Écarts AIS↔radio (Q8)", show=True)
    for _, r in df.iterrows():
        folium.PolyLine(
            [(r["lat_ais"], r["lon_ais"]), (r["lat_radio"], r["lon_radio"])],
            color="red", weight=2, opacity=0.6,
            tooltip=f"MMSI {r['mmsi']} — écart {r['distance_km']:.1f} km",
        ).add_to(fg)
        folium.CircleMarker((r["lat_ais"], r["lon_ais"]), radius=4,
                            color="blue", fill=True,
                            tooltip=f"AIS {r['mmsi']}").add_to(fg)
        folium.CircleMarker((r["lat_radio"], r["lon_radio"]), radius=4,
                            color="orange", fill=True,
                            tooltip=f"Radio {r['mmsi']}").add_to(fg)
    fg.add_to(m)
    return fg


def layer_suspect_ships(m: folium.Map, scores: pd.DataFrame,
                        ships: pd.DataFrame, ais: pd.DataFrame,
                        threshold: float = 0.5) -> folium.FeatureGroup:
    """Couche marqueurs : derniers points AIS des navires au score ≥ threshold."""
    fg = folium.FeatureGroup(name="Navires suspects", show=True)
    suspects = scores[scores["score"] >= threshold].merge(ships, on="mmsi", how="left")
    last_pos = ais.sort_values("timestamp").groupby("mmsi").tail(1)[
        ["mmsi", "latitude", "longitude"]
    ].set_index("mmsi")

    for _, r in suspects.iterrows():
        if r["mmsi"] not in last_pos.index:
            continue
        lat, lon = last_pos.loc[r["mmsi"], ["latitude", "longitude"]]
        popup = folium.Popup(
            f"<b>{r.get('name','?')}</b> (MMSI {r['mmsi']})<br>"
            f"Pavillon : {r.get('flag','?')}<br>"
            f"<b>Score : {r['score']:.2f}</b><br>"
            f"Top raisons : {r['top_reasons']}",
            max_width=380,
        )
        color = "red" if r["score"] >= 0.7 else "orange"
        folium.Marker((lat, lon), popup=popup,
                      icon=folium.Icon(color=color, icon="warning-sign")).add_to(fg)
    fg.add_to(m)
    return fg


def save(m: folium.Map, name: str = "carte_anomalies.html") -> Path:
    folium.LayerControl(collapsed=False).add_to(m)
    out = OUTPUTS / name
    m.save(str(out))
    return out
