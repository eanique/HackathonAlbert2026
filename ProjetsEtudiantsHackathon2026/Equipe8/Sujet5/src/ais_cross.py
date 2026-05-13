"""Croisement détection satellite × AIS — détection des « navires sombres »
(Levier L2 du cadrage — pont narratif avec le Sujet 3).

Principe :
    Pour chaque détection issue de la Piste B (`hunt.py`), interroger les sources
    AIS dans la fenêtre `(timestamp ± AIS_TIME_WINDOW_MIN, bbox ± AIS_DIST_WINDOW_KM)`.
    Une détection sans MMSI associé = **« navire sombre »** : visible par satellite,
    invisible (volontairement ou non) à l'AIS.

Sources AIS supportées (gratuit, OSINT) :
    - **Danish Maritime Authority** (`web.ais.dk/aisdata`) : CSV journaliers,
      principalement Baltique mais archive mondiale. Fichiers ~1 Go/jour.
    - **AISStream.io** : websocket temps réel, key gratuite. Idéal démo live.
    - **Global Fishing Watch API** : **gap events pré-calculés** (≥ 12 h AIS off
      en haute mer), bien plus rentable que de filtrer du AIS brut. Token gratuit
      non commercial.

Phase 1 (ce module) : interfaces + fonction de croisement déterministe.
Phase 2 J2 : implémentation effective des clients AIS (selon dispo des tokens).
"""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import pandas as pd
import requests
from geopy.distance import geodesic

from . import config as C


# ----------------------------------------------------------------------
# 1) Global Fishing Watch — gap events (les plus utiles pour le pitch)

_GFW_BASE = "https://gateway.api.globalfishingwatch.org/v3"


def gfw_gap_events(
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    min_gap_hours: int = 12,
    token: str | None = None,
) -> pd.DataFrame:
    """Récupère les `gap_events` GFW dans une bbox + plage temporelle.

    Un *gap event* = un navire dont l'AIS s'éteint plus de N heures en haute mer.
    Parfait pour le récit « navire sombre » : on lit les MMSI qui ont une coupure
    AIS au moment où on a une détection satellite dans la même zone.

    Args:
        bbox: (south, west, north, east) en degrés.
        start_date: ISO 'YYYY-MM-DD'.
        end_date:   ISO 'YYYY-MM-DD'.
        min_gap_hours: filtre min (défaut 12 h).
        token: GFW API token (sinon lu depuis `GFW_API_TOKEN`).

    Returns:
        DataFrame des gap events. Vide si pas de token ou API down.
    """
    token = token or os.getenv("GFW_API_TOKEN")
    if not token:
        return pd.DataFrame(
            columns=[
                "vessel_id",
                "mmsi",
                "start",
                "end",
                "gap_hours",
                "start_lat",
                "start_lon",
                "end_lat",
                "end_lon",
            ]
        )

    import json as _json
    south, west, north, east = bbox
    url = f"{_GFW_BASE}/events"
    # GFW v3 API : refuse tous les filtres spatiaux côté serveur (region/bbox/geometry).
    # On pagine + filtre côté client par bbox. Limit cap = 200/page.
    # Empiriquement : sur 2000 events worldwide ~1% tombent dans nos 5 zones stratégiques.
    # 50 pages = 10k events scannés → ~30-100 events par zone sur 1 an.
    PAGE = 200
    MAX_PAGES = 50
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    events = []
    try:
        for offset in range(0, PAGE * MAX_PAGES, PAGE):
            params = {
                "datasets[0]": "public-global-gaps-events:latest",
                "start-date": start_date,
                "end-date": end_date,
                "limit": PAGE,
                "offset": offset,
            }
            r = requests.get(url, params=params, headers=headers, timeout=30)
            r.raise_for_status()
            body = r.json()
            page = body.get("entries") or body.get("events") or []
            if not page:
                break
            events.extend(page)
            if len(page) < PAGE:  # plus de pages
                break
    except Exception as e:  # noqa: BLE001
        try:
            print(f"  [GFW] {type(e).__name__}: {e} · body: {_json.dumps(r.json())[:300]}")
        except Exception:
            print(f"  [GFW] {type(e).__name__}: {e}")
        if not events:
            return pd.DataFrame()

    # Filtrage spatial côté client (bbox)
    in_bbox = []
    for ev in events:
        pos = ev.get("position") or {}
        lat, lon = pos.get("lat"), pos.get("lon")
        if lat is None or lon is None:
            continue
        if south <= lat <= north and west <= lon <= east:
            in_bbox.append(ev)
    events = in_bbox

    rows = []
    for ev in events:
        # v3 retourne `durationHours` (float) ou `durationS` (en secondes)
        gap_h = ev.get("durationHours")
        if gap_h is None:
            ds = ev.get("durationS")
            if ds:
                gap_h = ds / 3600
        if gap_h is None:
            # Fallback : calcul depuis start/end
            try:
                from datetime import datetime as _dt
                t0 = _dt.fromisoformat(ev["start"].replace("Z", "+00:00"))
                t1 = _dt.fromisoformat(ev["end"].replace("Z", "+00:00"))
                gap_h = (t1 - t0).total_seconds() / 3600
            except Exception:  # noqa: BLE001
                continue
        if gap_h < min_gap_hours:
            continue
        v = ev.get("vessel", {}) or {}
        # v3 : position = centre de l'event ; pas de startCoordinates/endCoordinates
        pos = ev.get("position", {}) or {}
        rows.append(
            {
                "event_id": ev.get("id"),
                "vessel_id": v.get("id"),
                "mmsi": v.get("ssvid"),
                "vessel_name": v.get("name"),
                "flag": v.get("flag"),
                "ship_type": v.get("type") or v.get("geartype"),
                "start": ev.get("start"),
                "end": ev.get("end"),
                "gap_hours": round(float(gap_h), 1),
                "lat": pos.get("lat"),
                "lon": pos.get("lon"),
                # Compat avec l'ancien schéma
                "start_lat": pos.get("lat"),
                "start_lon": pos.get("lon"),
                "end_lat": pos.get("lat"),
                "end_lon": pos.get("lon"),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 2) AIS dans une bbox/fenêtre (stub interface — Phase 2 J2)

def ais_in_bbox(
    bbox: tuple[float, float, float, float],
    t_min,
    t_max,
    source: str = "auto",
) -> pd.DataFrame:
    """Liste les positions AIS dans `bbox` × `[t_min, t_max]`.

    Args:
        source: 'dma' (Danish Maritime Authority CSV), 'aisstream' (websocket
                temps réel), 'gfw' (positions API GFW), ou 'auto' (cascade
                selon les tokens dispos).

    Returns:
        DataFrame : `mmsi, timestamp, lat, lon, speed, course, ship_type, name`.
    """
    raise NotImplementedError(
        "Phase 2 J2 : implémenter le client AIS (DMA CSV, AISStream WS, ou GFW REST). "
        "Pour la démo, GFW (gap events pré-calculés) suffit — cf. gfw_gap_events()."
    )


# ----------------------------------------------------------------------
# 3) Croisement déterministe : détections × AIS → navires sombres

def flag_dark_ships(
    df_detections: pd.DataFrame,
    df_ais: pd.DataFrame,
    time_window_min: int = C.AIS_TIME_WINDOW_MIN,
    dist_window_km: float = C.AIS_DIST_WINDOW_KM,
) -> pd.DataFrame:
    """Annoter chaque détection : a-t-elle un MMSI AIS au même `(t, position)` ?

    Args:
        df_detections: colonnes `detection_id, timestamp (datetime UTC), lat, lon`.
        df_ais: colonnes `mmsi, timestamp, lat, lon`.
        time_window_min: fenêtre temporelle de matching.
        dist_window_km: fenêtre spatiale.

    Returns:
        df_detections enrichi de :
            - `n_ais_in_window`
            - `nearest_mmsi`, `nearest_mmsi_dist_km`
            - `is_dark` (bool) = aucun MMSI dans la fenêtre
    """
    df = df_detections.copy()
    df["n_ais_in_window"] = 0
    df["nearest_mmsi"] = None
    df["nearest_mmsi_dist_km"] = None

    if df_ais.empty:
        df["is_dark"] = True
        return df

    delta = pd.Timedelta(minutes=time_window_min)
    for i, det in df.iterrows():
        if pd.isna(det.get("lat")) or pd.isna(det.get("lon")):
            continue
        window = df_ais[
            (df_ais["timestamp"] >= det["timestamp"] - delta)
            & (df_ais["timestamp"] <= det["timestamp"] + delta)
        ]
        if window.empty:
            continue
        # Filtrage spatial
        best_d, best_mmsi = float("inf"), None
        n_in = 0
        for _, p in window.iterrows():
            try:
                d = geodesic((det["lat"], det["lon"]), (p["lat"], p["lon"])).km
            except (ValueError, TypeError):
                continue
            if d <= dist_window_km:
                n_in += 1
                if d < best_d:
                    best_d, best_mmsi = d, p["mmsi"]
        df.at[i, "n_ais_in_window"] = n_in
        if best_mmsi is not None:
            df.at[i, "nearest_mmsi"] = best_mmsi
            df.at[i, "nearest_mmsi_dist_km"] = round(best_d, 2)

    df["is_dark"] = df["n_ais_in_window"] == 0
    return df


# ----------------------------------------------------------------------
# 4) Démo standalone — gap events sur les bases navales

def survey_gap_events_all_bases(window_days: int = 30) -> pd.DataFrame:
    """Récupère les gap events GFW sur les 5 bases navales (preview Levier L2)."""
    from datetime import datetime, timezone

    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=window_days)
    all_dfs = []
    for name, bbox in C.BASES_NAVALES.items():
        df = gfw_gap_events(bbox, str(start), str(end))
        if df.empty:
            print(f"  {name:<20} -> 0 gap events (pas de token GFW ou rien à signaler)")
        else:
            df["base"] = name
            print(f"  {name:<20} -> {len(df)} gap events ≥ 12 h")
            all_dfs.append(df)
    if not all_dfs:
        return pd.DataFrame()
    out = pd.concat(all_dfs, ignore_index=True)
    out_path = C.DATA_PROCESSED / "chasse_gap_events.csv"
    out.to_csv(out_path, index=False)
    print(f"\n  → écrit {out_path.relative_to(C.ROOT)} ({len(out)} events)")
    return out


if __name__ == "__main__":
    print("[ais_cross] GFW gap events sur les 5 bases navales...")
    survey_gap_events_all_bases()
