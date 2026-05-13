"""Règles de spoofing par-dessus l'AIS (Levier 3).

Référentiel : arXiv 2603.11055 (« Wide-Area GNSS Spoofing & Jamming Detection »)
              + 2602.16257 (« SeaSpoofFinder »).

Chaque règle renvoie un DataFrame [mmsi, ts, type, confidence, description].
Toutes sont déterministes et rapides — pas de ML — donc idéales pour un
**pré-filtre** robuste avant les modèles statistiques (cours 4 §2.1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from geopy.distance import geodesic

from .config import SPEED_IMPLICIT_KMH_MAX


def rule_duplicate_mmsi(ais: pd.DataFrame, dt_max_min: int = 5,
                        d_min_km: float = 10.0) -> pd.DataFrame:
    """Même MMSI émettant depuis 2 positions incompatibles au même instant.

    Détecté en cherchant, dans une fenêtre Δt ≤ dt_max_min minutes, des paires
    de messages dont la distance géodésique > d_min_km — physiquement
    impossible pour un même bateau.
    """
    rows: list[dict] = []
    for mmsi, g in ais.sort_values("timestamp").groupby("mmsi"):
        if len(g) < 2:
            continue
        ts = g["timestamp"].to_numpy()
        lats = g["latitude"].to_numpy()
        lons = g["longitude"].to_numpy()
        for i in range(len(g) - 1):
            dt_min = (ts[i + 1] - ts[i]) / np.timedelta64(1, "m")
            if dt_min > dt_max_min:
                continue
            try:
                d_km = geodesic((lats[i], lons[i]), (lats[i + 1], lons[i + 1])).km
            except Exception:
                continue
            if d_km > d_min_km:
                conf = min(1.0, d_km / (dt_min * 200 + 1))
                rows.append({
                    "mmsi": mmsi, "ts": pd.Timestamp(ts[i + 1]),
                    "type": "Spoofing",
                    "confidence": float(conf),
                    "description": f"MMSI dupliqué : {d_km:.1f} km en {dt_min:.1f} min.",
                })
    return pd.DataFrame(rows)


def rule_timestamp_inconsistent(ais: pd.DataFrame) -> pd.DataFrame:
    """Timestamp antérieur au précédent du même MMSI (régression temporelle).

    Note : on n'utilise PAS `now + 1j` comme borne — le dataset est historique
    et peut couvrir une période future (synthétique). On flague uniquement les
    *régressions temporelles* dans la séquence d'un même MMSI, qui sont
    physiquement impossibles.
    """
    rows: list[dict] = []
    for mmsi, g in ais.sort_values(["mmsi", "timestamp"]).groupby("mmsi"):
        # On doit reposer le tri à l'intérieur du groupe par sécurité
        ts = g["timestamp"].sort_values()
        regressed = ts.diff() < pd.Timedelta(0)
        for t in ts[regressed]:
            rows.append({
                "mmsi": mmsi, "ts": t, "type": "Spoofing",
                "confidence": 0.9,
                "description": "Régression temporelle dans la séquence AIS.",
            })
    return pd.DataFrame(rows)


def rule_implausible_speed(ais: pd.DataFrame,
                           kmh_max: float = SPEED_IMPLICIT_KMH_MAX) -> pd.DataFrame:
    """Vitesse implicite (distance géodésique entre 2 points consécutifs / Δt)
    au-dessus du seuil → spoofing GNSS probable ou tracking faussé.

    Le `speed` déclaré peut être faussé ; la vitesse implicite est plus robuste.
    """
    rows: list[dict] = []
    for mmsi, g in ais.sort_values("timestamp").groupby("mmsi"):
        if len(g) < 2:
            continue
        g = g.reset_index(drop=True)
        for i in range(1, len(g)):
            dt_h = (g.loc[i, "timestamp"] - g.loc[i - 1, "timestamp"]).total_seconds() / 3600
            if dt_h <= 0:
                continue
            try:
                d_km = geodesic(
                    (g.loc[i - 1, "latitude"], g.loc[i - 1, "longitude"]),
                    (g.loc[i, "latitude"], g.loc[i, "longitude"]),
                ).km
            except Exception:
                continue
            v = d_km / dt_h
            if v > kmh_max:
                conf = min(1.0, (v - kmh_max) / kmh_max)
                rows.append({
                    "mmsi": mmsi, "ts": g.loc[i, "timestamp"],
                    "type": "Speed Anomaly",
                    "confidence": float(conf),
                    "description": f"Vitesse implicite : {v:.1f} km/h (seuil {kmh_max} km/h).",
                })
    return pd.DataFrame(rows)


def rule_transmission_interval(ais: pd.DataFrame,
                               z_threshold: float = 4.0) -> pd.DataFrame:
    """Intervalles de transmission anormaux par rapport au profil du navire.

    Pour chaque MMSI : on calcule la médiane et le MAD des Δt entre messages ;
    on flag les Δt dont le z-score robuste dépasse `z_threshold` (en plus ou
    en moins — émissions trop espacées OU au contraire trop rapprochées,
    typique d'un retransmission spoofée).
    """
    rows: list[dict] = []
    for mmsi, g in ais.sort_values("timestamp").groupby("mmsi"):
        if len(g) < 5:
            continue
        dt_min = g["timestamp"].diff().dt.total_seconds().div(60).dropna().to_numpy()
        if len(dt_min) < 4:
            continue
        med = np.median(dt_min)
        mad = np.median(np.abs(dt_min - med)) * 1.4826 + 1e-9
        z = (dt_min - med) / mad
        idx = np.where(np.abs(z) > z_threshold)[0]
        for j in idx:
            rows.append({
                "mmsi": mmsi, "ts": g["timestamp"].iloc[j + 1],
                "type": "Spoofing",
                "confidence": float(min(1.0, (abs(z[j]) - z_threshold) / z_threshold)),
                "description": f"Intervalle anormal : Δt={dt_min[j]:.1f} min (médiane {med:.1f}, |z|={abs(z[j]):.1f}).",
            })
    return pd.DataFrame(rows)


def rule_course_anomaly(ais: pd.DataFrame, delta_deg: float = 60.0) -> pd.DataFrame:
    """Changement de cap brutal entre 2 messages AIS consécutifs (gère le wraparound 0/360).

    Un navire normal vire avec une accélération angulaire limitée — Δcap > 60°
    entre deux positions consécutives en mer ouverte trahit soit une manœuvre
    militaire, soit un cap falsifié. Sert à récupérer les `Course Anomaly` dont
    la valeur n'apparaît pas dans la table `ais.course` (description-only).
    """
    rows: list[dict] = []
    if "course" not in ais.columns:
        return pd.DataFrame(rows)
    for mmsi, g in ais.sort_values(["mmsi", "timestamp"]).groupby("mmsi"):
        if len(g) < 2:
            continue
        c = g["course"].to_numpy(dtype=float)
        ts = g["timestamp"].to_numpy()
        for i in range(1, len(c)):
            if np.isnan(c[i]) or np.isnan(c[i - 1]):
                continue
            d = abs(c[i] - c[i - 1]) % 360.0
            d = min(d, 360.0 - d)  # plus court arc
            if d > delta_deg:
                conf = float(min(1.0, (d - delta_deg) / (180.0 - delta_deg + 1e-9)))
                rows.append({
                    "mmsi": mmsi, "ts": pd.Timestamp(ts[i]),
                    "type": "Course Anomaly",
                    "confidence": conf,
                    "description": f"Δcap = {d:.1f}° entre 2 positions consécutives (seuil {delta_deg:.0f}°).",
                })
    return pd.DataFrame(rows)


def run_all(ais: pd.DataFrame) -> pd.DataFrame:
    """Concatène les règles → DataFrame unique [mmsi, ts, type, confidence, description, rule]."""
    parts = []
    for name, fn in [
        ("duplicate_mmsi", rule_duplicate_mmsi),
        ("timestamp_inconsistent", rule_timestamp_inconsistent),
        ("implausible_speed", rule_implausible_speed),
        ("transmission_interval", rule_transmission_interval),
        ("course_anomaly", rule_course_anomaly),
    ]:
        out = fn(ais)
        if not out.empty:
            out["rule"] = name
            parts.append(out)
    return (pd.concat(parts, ignore_index=True)
            if parts else pd.DataFrame(columns=["mmsi", "ts", "type", "confidence", "description", "rule"]))
