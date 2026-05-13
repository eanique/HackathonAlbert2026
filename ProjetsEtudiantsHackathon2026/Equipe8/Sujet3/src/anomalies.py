"""Détecteurs d'anomalies — Q4, Q5, Q6, Q7, Q8 (Sujet 3).

Référentiel ML : cours d'A. Bogroff, cours 4 « Détection d'anomalies ».

Q4 — Anomalies de pavillon (`Fake Flag`)
    Pour chaque `flag`, fitter `EllipticEnvelope (MCD)` (cours 4 §4) sur le
    cœur dense des navires *non suspects* de ce pavillon → flaguer les écarts.
    Contourne le **masking effect** de la distance de Mahalanobis naïve.

Q5 — Changements de nom (`Name Change`)
    `historical_names` parsé ; flag si > 2 noms.

Q6 — Signatures orphelines (`Spoofing`)
    `radio.mmsi ∉ ships.mmsi` → orphelines. Attribution par k-NN dans
    l'espace standardisé (frequency, bandwidth, power).

Q7 — AIS désactivé > 24 h consécutives (`AIS Disabled`)
    Plages contiguës `ais_active=False`, exclusion `Moored/At Anchor`.
    (Algo inspiré de pipe-gaps / Global Fishing Watch.)

Q8 — Écart position AIS ↔ radio > 1 km (`Position Mismatch`)
    `merge_asof` par mmsi (tolérance ±10 min) + `geopy.distance.geodesic`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .config import (
    AIS_OFF_EXCLUDE_STATUS,
    AIS_OFF_HOURS,
    ASOF_TOLERANCE_MIN,
    FLAG_CONTAMINATION,
    FLAG_MIN_SHIPS,
    POS_MISMATCH_KM,
)


@dataclass
class AnomalyFlag:
    """Résultat unifié d'un détecteur. Réutilisé par anomaly_score.py."""
    mmsi: str
    type: str           # "Fake Flag" | "Name Change" | "Spoofing" | "AIS Disabled" | "Position Mismatch"
    confidence: float   # ∈ [0, 1]
    description: str
    timestamp: pd.Timestamp | None = None


# ----------------------------------------------------------------------------
# Q4 — Faux pavillon (EllipticEnvelope / MCD par flag)
# ----------------------------------------------------------------------------


def detect_fake_flag(profiles: pd.DataFrame,
                     features: tuple[str, ...] = ("freq_mean", "bandwidth_mean", "power_mean"),
                     min_ships: int = FLAG_MIN_SHIPS,
                     contamination: float = FLAG_CONTAMINATION) -> pd.DataFrame:
    """Cours 4 §4 — EllipticEnvelope (MCD) par pavillon.

    Pour chaque flag ≥ `min_ships` navires :
        1. fitter `EllipticEnvelope` sur les navires `is_suspicious=False`,
        2. scorer **tous** les navires de ce pavillon (`mahalanobis()`),
        3. label `-1` (anormal) ou `+1` (normal) via `predict()`.

    Pour les pavillons trop petits, fallback : z-score robuste sur médiane ± MAD.

    Returns: DataFrame [mmsi, flag, mahalanobis, predicted, confidence]
        - confidence ∈ [0, 1] : `min(1, (d - threshold) / threshold)` puis clip.
    """
    rows: list[dict] = []
    for flag, sub in profiles.dropna(subset=list(features) + ["flag"]).groupby("flag"):
        X = sub[list(features)].to_numpy()
        is_susp = sub.get("is_suspicious", pd.Series(False, index=sub.index)).fillna(False).astype(bool).to_numpy()
        if len(sub) < min_ships or (~is_susp).sum() < min_ships // 2:
            # Fallback : z-score robuste (médiane ± MAD)
            med = np.median(X[~is_susp] if (~is_susp).any() else X, axis=0)
            mad = np.median(np.abs(X - med), axis=0) * 1.4826 + 1e-9
            z = np.max(np.abs((X - med) / mad), axis=1)
            conf = np.clip((z - 3) / 3, 0, 1)
            for mmsi, zi, ci in zip(sub["mmsi"], z, conf):
                rows.append({"mmsi": mmsi, "flag": flag, "mahalanobis": float(zi),
                             "predicted": -1 if zi > 3 else 1, "confidence": float(ci),
                             "method": "MAD-z (fallback)"})
            continue

        try:
            envelope = EllipticEnvelope(contamination=contamination,
                                        support_fraction=None, random_state=42)
            envelope.fit(X[~is_susp])
            d = envelope.mahalanobis(X)
            pred = envelope.predict(X)
            threshold = float(np.quantile(d[~is_susp], 1 - contamination))
            conf = np.clip((d - threshold) / max(threshold, 1e-9), 0, 1)
            for mmsi, di, pi, ci in zip(sub["mmsi"], d, pred, conf):
                rows.append({"mmsi": mmsi, "flag": flag, "mahalanobis": float(di),
                             "predicted": int(pi), "confidence": float(ci),
                             "method": "EllipticEnvelope/MCD"})
        except Exception as exc:
            # Si MCD échoue (singularité, etc.), on tombe en MAD
            med = np.median(X[~is_susp], axis=0)
            mad = np.median(np.abs(X - med), axis=0) * 1.4826 + 1e-9
            z = np.max(np.abs((X - med) / mad), axis=1)
            conf = np.clip((z - 3) / 3, 0, 1)
            for mmsi, zi, ci in zip(sub["mmsi"], z, conf):
                rows.append({"mmsi": mmsi, "flag": flag, "mahalanobis": float(zi),
                             "predicted": -1 if zi > 3 else 1, "confidence": float(ci),
                             "method": f"MAD-z (MCD failed: {type(exc).__name__})"})

    return pd.DataFrame(rows)


def top_n_fake_flag(report: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Q4 — top N anomalies de pavillon triées par `confidence`."""
    return (report[report["predicted"] == -1]
            .sort_values("confidence", ascending=False)
            .head(n)
            .reset_index(drop=True))


# ----------------------------------------------------------------------------
# Q5 — Changements de nom
# ----------------------------------------------------------------------------


def detect_name_change(ships: pd.DataFrame, threshold: int = 2) -> pd.DataFrame:
    """Navires avec `n_names_historical > threshold`.

    Returns: DataFrame [mmsi, name, n_names_historical, historical_names_list].
    """
    if "n_names_historical" not in ships.columns:
        return ships.iloc[0:0]
    mask = ships["n_names_historical"] > threshold
    cols = [c for c in ("mmsi", "name", "flag", "n_names_historical",
                        "historical_names_list", "is_suspicious") if c in ships.columns]
    return ships.loc[mask, cols].reset_index(drop=True)


# ----------------------------------------------------------------------------
# Q6 — Signatures orphelines (`Spoofing`)
# ----------------------------------------------------------------------------


def detect_orphan_signatures(radio: pd.DataFrame, ships: pd.DataFrame) -> pd.DataFrame:
    """Signatures dont le `mmsi` n'est pas dans `ships`."""
    return radio[~radio["mmsi"].isin(ships["mmsi"])].copy()


def detect_silent_ships(ships: pd.DataFrame, radio: pd.DataFrame) -> pd.DataFrame:
    """Navires du registre **sans aucune signature radio captée**.

    Découvert à l'EDA : 6 navires sur 1000 n'ont jamais été « entendus » sur
    les ondes — dont 4 marqués `is_suspicious`. Un navire électromagnétiquement
    invisible est en soi un indice (cf. rapport, section qualité des données).
    """
    silent = ships[~ships["mmsi"].isin(set(radio["mmsi"]))]
    cols = [c for c in ("mmsi", "name", "flag", "type", "is_suspicious")
            if c in ships.columns]
    return silent[cols].reset_index(drop=True)


def attribute_orphans(orphans: pd.DataFrame,
                      profiles: pd.DataFrame,
                      k: int = 5,
                      features: tuple[str, ...] = ("freq_mean", "bandwidth_mean", "power_mean")
                      ) -> pd.DataFrame:
    """k-NN dans l'espace standardisé → MMSI candidat le plus proche.

    Returns: DataFrame avec colonnes ajoutées :
        candidate_mmsi : MMSI le plus proche dans `profiles`
        candidate_distance : distance dans l'espace standardisé
        candidate_confidence : `1 / (1 + distance)` (heuristique)
    """
    feat_radio = ("frequency", "bandwidth", "power")
    if profiles.empty or orphans.empty:
        out = orphans.copy()
        out["candidate_mmsi"] = None
        out["candidate_distance"] = np.nan
        out["candidate_confidence"] = 0.0
        return out

    X_ref = profiles[list(features)].to_numpy()
    scaler = StandardScaler().fit(X_ref)
    X_ref_std = scaler.transform(X_ref)
    nn = NearestNeighbors(n_neighbors=min(k, len(X_ref))).fit(X_ref_std)

    X_q = scaler.transform(orphans[list(feat_radio)].to_numpy())
    dist, idx = nn.kneighbors(X_q)
    candidate = profiles["mmsi"].iloc[idx[:, 0]].to_numpy()

    out = orphans.copy()
    out["candidate_mmsi"] = candidate
    out["candidate_distance"] = dist[:, 0]
    out["candidate_confidence"] = 1.0 / (1.0 + dist[:, 0])
    return out


# ----------------------------------------------------------------------------
# Q7 — AIS off > 24 h consécutives
# ----------------------------------------------------------------------------


def detect_ais_disabled(ais: pd.DataFrame,
                        min_hours: int = AIS_OFF_HOURS,
                        exclude_status: set[str] = AIS_OFF_EXCLUDE_STATUS) -> pd.DataFrame:
    """Plages contiguës `ais_active=False`, durée > `min_hours`,
    en excluant les plages où le navire est `Moored/At Anchor` (légitime).

    Inspiré de l'algo `pipe-gaps` de Global Fishing Watch.

    Returns: DataFrame [mmsi, start, end, duration_hours, n_messages,
                       status_majority, suspect]
    """
    df = ais.sort_values(["mmsi", "timestamp"]).copy()
    df["ais_off"] = ~df["ais_active"].astype(bool)
    # nouveau bloc à chaque changement de mmsi ou de ais_off
    # (cast en int : pandas-pyarrow ne supporte pas cumsum sur bool)
    block_change = (
        (df["mmsi"].astype("string") != df["mmsi"].astype("string").shift())
        | (df["ais_off"].astype(bool) != df["ais_off"].astype(bool).shift())
    ).astype("int64")
    df["block_id"] = block_change.cumsum()

    blocks = (df[df["ais_off"]]
              .groupby(["block_id", "mmsi"])
              .agg(start=("timestamp", "min"),
                   end=("timestamp", "max"),
                   n_messages=("timestamp", "size"),
                   status_majority=("status", lambda s: s.mode().iloc[0] if len(s.mode()) else ""))
              .reset_index())
    blocks["duration_hours"] = (blocks["end"] - blocks["start"]).dt.total_seconds() / 3600
    blocks["suspect"] = (
        (blocks["duration_hours"] > min_hours)
        & (~blocks["status_majority"].isin(exclude_status))
    )
    return blocks


# ----------------------------------------------------------------------------
# Q8 — Écart position AIS ↔ radio > 1 km
# ----------------------------------------------------------------------------


def detect_position_mismatch(ais: pd.DataFrame, radio: pd.DataFrame,
                             tolerance_min: int = ASOF_TOLERANCE_MIN,
                             threshold_km: float = POS_MISMATCH_KM) -> pd.DataFrame:
    """Apparie chaque point radio au point AIS le plus proche dans le temps
    (`merge_asof` par mmsi avec tolérance ±N minutes), puis calcule la
    distance géodésique. Renvoie les paires avec distance > threshold_km.

    Returns: DataFrame [mmsi, timestamp_radio, timestamp_ais,
                       lat_ais, lon_ais, lat_radio, lon_radio,
                       distance_km]
    """
    radio_s = radio.sort_values("timestamp").rename(
        columns={"timestamp": "timestamp_radio",
                 "location_lat": "lat_radio",
                 "location_lon": "lon_radio"})
    ais_s = (ais.sort_values("timestamp")
             .rename(columns={"timestamp": "timestamp_ais",
                              "latitude": "lat_ais",
                              "longitude": "lon_ais"})
             [["mmsi", "timestamp_ais", "lat_ais", "lon_ais"]])
    merged = pd.merge_asof(
        radio_s.sort_values("timestamp_radio"),
        ais_s.sort_values("timestamp_ais"),
        left_on="timestamp_radio", right_on="timestamp_ais",
        by="mmsi", tolerance=pd.Timedelta(minutes=tolerance_min),
        direction="nearest",
    ).dropna(subset=["lat_ais", "lon_ais"])

    def _d(row) -> float:
        try:
            return geodesic((row["lat_ais"], row["lon_ais"]),
                            (row["lat_radio"], row["lon_radio"])).km
        except Exception:
            return np.nan

    merged["distance_km"] = merged.apply(_d, axis=1)
    return merged[merged["distance_km"] > threshold_km][
        ["mmsi", "timestamp_radio", "timestamp_ais",
         "lat_ais", "lon_ais", "lat_radio", "lon_radio", "distance_km"]
    ].reset_index(drop=True)
