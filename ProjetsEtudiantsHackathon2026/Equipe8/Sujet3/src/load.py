"""Chargement et normalisation des CSV fournis (Sujet 3).

Conventions (cf. docs/hypotheses.md §A) :
- `mmsi`, `imo` traités comme **chaînes** (jamais en int).
- Tous les `timestamp` → `datetime` UTC.
- AIS **trié par (mmsi, timestamp)**.
- Dédup ; manquants loggés.
- Validation MMSI : 9 chiffres + MID UIT connu.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import DATA_RAW, mmsi_country

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Lecture brute
# ----------------------------------------------------------------------------


def _read_csv_str_mmsi(path: Path) -> pd.DataFrame:
    """Lecture pandas avec `mmsi`/`imo` forcés en string."""
    dtype = {"mmsi": "string", "imo": "string"}
    return pd.read_csv(path, dtype={k: v for k, v in dtype.items()})


def _to_utc(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


# ----------------------------------------------------------------------------
# Loaders publics
# ----------------------------------------------------------------------------


def load_ships(small: bool = False) -> pd.DataFrame:
    """`ships_large.csv` (1 000) ou `ships_small.csv` (20)."""
    path = DATA_RAW / ("ships_small.csv" if small else "ships_large.csv")
    df = _read_csv_str_mmsi(path)
    df = _to_utc(df, "last_ais_update")
    # historical_names → list[str]
    if "historical_names" in df.columns:
        df["historical_names_list"] = (
            df["historical_names"].fillna("").astype(str)
            .str.split(",").apply(lambda lst: [s.strip() for s in lst if s.strip()])
        )
        df["n_names_historical"] = df["historical_names_list"].str.len()
    df = df.drop_duplicates(subset=["mmsi"], keep="last")
    log.info("ships : %d navires (dont %d suspects)",
             len(df), int(df.get("is_suspicious", pd.Series(dtype=bool)).sum()))
    return df


def load_radio(small: bool = False) -> pd.DataFrame:
    """`radio_signatures_large.csv` (5 000) ou `..._small.csv` (20)."""
    path = DATA_RAW / ("radio_signatures_small.csv" if small else "radio_signatures_large.csv")
    df = _read_csv_str_mmsi(path)
    df = _to_utc(df)
    df = df.drop_duplicates().reset_index(drop=True)
    log.info("radio : %d signatures, %d MMSI distincts",
             len(df), df["mmsi"].nunique())
    return df


def load_ais(small: bool = False) -> pd.DataFrame:
    """`ais_data_large.csv` (10 000) ou `..._small.csv` (20).

    Trié par (mmsi, timestamp) — préalable à tout calcul de gap / vitesse implicite.
    """
    path = DATA_RAW / ("ais_data_small.csv" if small else "ais_data_large.csv")
    df = _read_csv_str_mmsi(path)
    df = _to_utc(df)
    df = df.drop_duplicates(subset=["mmsi", "timestamp"], keep="last")
    df = df.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)
    log.info("ais : %d lignes, %d MMSI, AIS_OFF=%d",
             len(df), df["mmsi"].nunique(),
             int((~df["ais_active"].astype(bool)).sum())
             if "ais_active" in df.columns else 0)
    return df


def load_anomalies() -> pd.DataFrame:
    """`anomalies_large.csv` (100, vérité terrain). Pas de version small."""
    path = DATA_RAW / "anomalies_large.csv"
    df = _read_csv_str_mmsi(path)
    df = _to_utc(df)
    return df


# ----------------------------------------------------------------------------
# Validation MMSI / MID UIT
# ----------------------------------------------------------------------------


def annotate_mmsi_validity(df: pd.DataFrame, col: str = "mmsi") -> pd.DataFrame:
    """Ajoute `mmsi_is_valid` (bool) et `mmsi_country` (pays MID UIT ou None)."""
    df = df.copy()
    s = df[col].astype("string").fillna("")
    df["mmsi_is_valid"] = s.str.match(r"^\d{9}$")
    df["mmsi_country"] = s.apply(mmsi_country)
    return df


# ----------------------------------------------------------------------------
# Snapshot de qualité (loggué dans le rapport)
# ----------------------------------------------------------------------------


def data_quality_report(ships: pd.DataFrame, radio: pd.DataFrame,
                        ais: pd.DataFrame, anomalies: pd.DataFrame) -> dict:
    """Renvoie un dict de stats à inclure dans le rapport (manquants, dédups…)."""
    return {
        "n_ships": len(ships),
        "n_radio_signatures": len(radio),
        "n_ais_rows": len(ais),
        "n_anomalies": len(anomalies),
        "ships_with_historical_names>2": int(
            ships.get("n_names_historical", pd.Series(dtype=int)).gt(2).sum()
        ),
        "radio_orphan_mmsi": int(
            (~radio["mmsi"].isin(ships["mmsi"])).sum()
        ),
        "ais_off_rows": int((~ais["ais_active"].astype(bool)).sum())
            if "ais_active" in ais.columns else 0,
        "anomaly_types": (
            anomalies["type"].value_counts().to_dict() if "type" in anomalies else {}
        ),
    }
