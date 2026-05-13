"""Construction de la base de profils radio (Q1 — Sujet 3).

Pour chaque navire, on agrège ses signatures radio (cours 1 §2 → moyennes des
features quantitatives, mode pour le catégoriel). On produit le fichier
`ship_radio_profiles.csv` demandé par l'énoncé.

Variables d'entrée (cf. docs/hypotheses.md §B-Q1) :
    radio : DataFrame (radio_signatures_large.csv)
    ships : DataFrame (ships_large.csv) — optionnel, pour join du flag

Variables produites :
    ship_radio_profiles.csv : 1 ligne par MMSI distinct dans `radio`,
        colonnes :
            mmsi, n_signatures,
            freq_mean, freq_std,
            bandwidth_mean, bandwidth_std,
            power_mean, power_std,
            signal_strength_mean, snr_mean,
            modulation_mode, pulse_pattern_mode,
            modulation_entropy, pulse_pattern_entropy
        + (si ships fourni) flag, name, type, is_suspicious
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_PROCESSED


# ----------------------------------------------------------------------------
# Utilitaires
# ----------------------------------------------------------------------------


def _entropy(series: pd.Series) -> float:
    """Entropie de Shannon (base 2) d'une série catégorielle (ignore NaN)."""
    s = series.dropna()
    if len(s) == 0:
        return 0.0
    p = s.value_counts(normalize=True).to_numpy()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _mode_or_none(series: pd.Series):
    s = series.dropna()
    return s.mode().iloc[0] if len(s) and len(s.mode()) else None


# ----------------------------------------------------------------------------
# Construction des profils
# ----------------------------------------------------------------------------


def build_profiles(radio: pd.DataFrame,
                   ships: pd.DataFrame | None = None,
                   out_path: Path | None = None) -> pd.DataFrame:
    """Agrège les signatures radio par MMSI (cf. docstring du module).

    Note : l'énoncé impose la moyenne des features numériques. On ajoute
    l'écart-type, le mode catégoriel et l'entropie catégorielle pour enrichir
    le profil sans casser l'attendu.
    """
    num_cols = [c for c in ("frequency", "bandwidth", "power",
                            "signal_strength", "signal_to_noise_ratio")
                if c in radio.columns]

    agg: dict[str, list] = {c: ["mean", "std"] for c in num_cols}
    grouped = radio.groupby("mmsi").agg(agg)
    grouped.columns = [f"{c}_{stat}" for c, stat in grouped.columns]
    grouped["n_signatures"] = radio.groupby("mmsi").size()

    # Catégoriel : mode + entropie
    for cat in ("modulation", "pulse_pattern"):
        if cat in radio.columns:
            grouped[f"{cat}_mode"] = radio.groupby("mmsi")[cat].agg(_mode_or_none)
            grouped[f"{cat}_entropy"] = radio.groupby("mmsi")[cat].agg(_entropy)

    grouped = grouped.reset_index()

    # Renommage attendu par l'énoncé (alias plus courts)
    rename = {
        "frequency_mean": "freq_mean", "frequency_std": "freq_std",
        "signal_strength_mean": "signal_strength_mean",
        "signal_to_noise_ratio_mean": "snr_mean",
        "signal_to_noise_ratio_std": "snr_std",
    }
    grouped = grouped.rename(columns=rename)

    if ships is not None:
        keep = [c for c in ("mmsi", "name", "type", "flag", "is_suspicious",
                            "historical_names", "n_names_historical")
                if c in ships.columns]
        grouped = grouped.merge(ships[keep], on="mmsi", how="left")

    out_path = out_path or DATA_PROCESSED / "ship_radio_profiles.csv"
    grouped.to_csv(out_path, index=False)
    return grouped


# ----------------------------------------------------------------------------
# Helpers requêtes (Q1, Q2)
# ----------------------------------------------------------------------------


def top_n_frequency(profiles: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Q1 : top N navires par fréquence moyenne."""
    cols = [c for c in ("mmsi", "name", "flag", "freq_mean", "n_signatures")
            if c in profiles.columns]
    return profiles.sort_values("freq_mean", ascending=False).head(n)[cols].reset_index(drop=True)


def unique_pulse_patterns(radio: pd.DataFrame) -> dict:
    """Q2 : `pulse_pattern` apparaissant 1 seule fois + combinaisons uniques.

    L'énoncé prend `pulse_pattern` seul (6 valeurs possibles → peu discriminant).
    On rapporte aussi la combinaison `pulse_pattern × modulation × bande` où la
    bande = `frequency` arrondie à 0.1 MHz — bien plus informatif.
    """
    counts = radio["pulse_pattern"].value_counts()
    unique_patterns = counts[counts == 1].index.tolist()
    n_distinct = radio["pulse_pattern"].nunique()

    radio = radio.copy()
    radio["freq_bin"] = (radio["frequency"] * 10).round() / 10
    combo = radio.groupby(["pulse_pattern", "modulation", "freq_bin"]).size()
    unique_combo = combo[combo == 1].reset_index().rename(columns={0: "count"})

    return {
        "n_distinct_patterns": int(n_distinct),
        "patterns_with_count_1": unique_patterns,
        "n_unique_combinations": int(len(unique_combo)),
        "unique_combinations_sample": unique_combo.head(20).to_dict(orient="records"),
    }
