"""Score de suspicion global multi-facteurs (Levier 3 — Sujet 3).

Combine :
  - Détecteurs Q4–Q8 (anomalies.py)
  - Règles de spoofing (spoofing_rules.py)
  - **Isolation Forest** sur `radio_signatures` (cours 4 §6) — pas de scaling
  - **LOF** sur `radio_signatures` (cours 4 §7) — avec StandardScaler
  - Sous-score zone-dépendant (façon GeoTrackNet light, arXiv 1912.00682)
  - (option) terme LLM tabulaire zero-shot (arXiv 2406.16308)

Le score global ∈ [0, 1] est calibré sur l'AUC vs `is_suspicious` (split 70/30).

⚠️ Cours 4 §8 — Évaluation à **coût asymétrique** :
    coût_total = α·FP + β·FN avec β/α = COST_FN/COST_FP = 10 → on calibre vers
    le rappel (rater un navire suspect est très coûteux pour Minarm).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from .config import (
    COST_FN,
    COST_FP,
    IFOREST_CONTAMINATION,
    IFOREST_N_ESTIMATORS,
    LOF_CONTAMINATION,
    LOF_N_NEIGHBORS,
)


# ----------------------------------------------------------------------------
# Features pour iForest / LOF
# ----------------------------------------------------------------------------


def make_features(radio: pd.DataFrame) -> pd.DataFrame:
    """One-hot des catégoriels + features numériques.

    Renvoie un DataFrame numérique (signature-level) prêt pour iForest / LOF.
    """
    num_cols = [c for c in ("frequency", "bandwidth", "power",
                            "signal_strength", "noise_level",
                            "signal_to_noise_ratio") if c in radio.columns]
    df = radio[num_cols].copy()

    if "modulation" in radio.columns:
        df = pd.concat([df, pd.get_dummies(radio["modulation"], prefix="mod")], axis=1)
    if "pulse_pattern" in radio.columns:
        df = pd.concat([df, pd.get_dummies(radio["pulse_pattern"], prefix="pp")], axis=1)
    return df.fillna(df.median(numeric_only=True))


# ----------------------------------------------------------------------------
# Isolation Forest (cours 4 §6) — sans scaling !
# ----------------------------------------------------------------------------


def isolation_forest_scores(radio: pd.DataFrame,
                            contamination: float = IFOREST_CONTAMINATION,
                            n_estimators: int = IFOREST_N_ESTIMATORS) -> pd.DataFrame:
    """Score iForest par signature. ⚠️ Pas de StandardScaler (cours 4 §6).

    Returns: DataFrame [signature_id, mmsi, if_score, if_anomaly]
      - if_score ∈ [0, 1] ≈ probabilité d'anomalie (1 = sûr anomalie).
      - if_anomaly ∈ {-1, 1} via le contamination threshold.
    """
    X = make_features(radio).to_numpy()
    model = IsolationForest(n_estimators=n_estimators,
                            contamination=contamination,
                            random_state=42)
    model.fit(X)
    raw = -model.score_samples(X)  # plus grand = plus anormal
    score = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    pred = model.predict(X)
    return pd.DataFrame({
        "signature_id": radio.get("signature_id"),
        "mmsi": radio["mmsi"],
        "if_score": score,
        "if_anomaly": pred,
    })


# ----------------------------------------------------------------------------
# LOF (cours 4 §7) — anomalies locales, avec StandardScaler
# ----------------------------------------------------------------------------


def lof_scores(radio: pd.DataFrame,
               n_neighbors: int = LOF_N_NEIGHBORS,
               contamination: float = LOF_CONTAMINATION) -> pd.DataFrame:
    """LOF (cours 4 §7) — anomalies *locales* (un point qui paraît normal
    globalement mais détonne dans son voisinage)."""
    X = make_features(radio).to_numpy()
    X_std = StandardScaler().fit_transform(X)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors,
                             contamination=contamination,
                             novelty=False)
    pred = lof.fit_predict(X_std)
    raw = -lof.negative_outlier_factor_   # >1 = anormal
    score = np.clip((raw - 1.0) / 2.0, 0, 1)
    return pd.DataFrame({
        "signature_id": radio.get("signature_id"),
        "mmsi": radio["mmsi"],
        "lof_score": score,
        "lof_anomaly": pred,
    })


# ----------------------------------------------------------------------------
# Sous-score zone-dépendant (façon GeoTrackNet light)
# ----------------------------------------------------------------------------


def zone_anomaly_score(ais: pd.DataFrame, cell_deg: float = 1.0) -> pd.DataFrame:
    """Pour chaque cellule lat×lon de `cell_deg`°, on calcule la distribution
    empirique de (cap, vitesse, statut). Le score d'un point = `−log p`
    écrêté à [0, 10], normalisé en [0, 1].

    Returns: DataFrame [mmsi, timestamp, lat, lon, zone_score]
    """
    df = ais.copy()
    df["lat_bin"] = (df["latitude"] // cell_deg) * cell_deg
    df["lon_bin"] = (df["longitude"] // cell_deg) * cell_deg
    df["speed_bin"] = (df["speed"].fillna(0) // 5) * 5      # 5 nœuds
    df["course_bin"] = (df["course"].fillna(0) // 30) * 30  # 30°

    # Probabilité empirique de la combinaison (cap_bin, speed_bin, status) par cellule
    by_cell = df.groupby(["lat_bin", "lon_bin"]).size().rename("cell_total")
    by_combo = (df.groupby(["lat_bin", "lon_bin", "course_bin", "speed_bin"])
                .size().rename("combo_count").reset_index())
    by_combo = by_combo.merge(by_cell, on=["lat_bin", "lon_bin"])
    by_combo["p"] = by_combo["combo_count"] / by_combo["cell_total"]

    merged = df.merge(by_combo[["lat_bin", "lon_bin", "course_bin", "speed_bin", "p"]],
                      on=["lat_bin", "lon_bin", "course_bin", "speed_bin"], how="left")
    merged["p"] = merged["p"].fillna(1e-3)
    merged["zone_score"] = np.clip(-np.log(merged["p"]) / 10.0, 0, 1)
    return merged[["mmsi", "timestamp", "latitude", "longitude", "zone_score"]]


# ----------------------------------------------------------------------------
# Score global par navire
# ----------------------------------------------------------------------------


@dataclass
class ScoreWeights:
    """Poids du score multi-facteurs. À ajuster pour maximiser l'AUC."""
    fake_flag: float = 1.5
    name_change: float = 0.8
    orphan: float = 1.0
    ais_off: float = 1.3
    position_mismatch: float = 1.2
    speed: float = 1.0
    spoofing_rules: float = 1.5
    isolation_forest: float = 1.0
    lof: float = 0.7
    zone: float = 0.6
    llm_tabular: float = 0.3   # faible, complément zero-shot


def build_global_score(ships: pd.DataFrame, *,
                       fake_flag_report: pd.DataFrame | None = None,
                       name_change_mmsi: list[str] | None = None,
                       orphan_attribution: pd.DataFrame | None = None,
                       ais_off_blocks: pd.DataFrame | None = None,
                       position_mismatch: pd.DataFrame | None = None,
                       spoofing_rules: pd.DataFrame | None = None,
                       iforest_scores: pd.DataFrame | None = None,
                       lof_scores_df: pd.DataFrame | None = None,
                       zone_scores: pd.DataFrame | None = None,
                       llm_scores: pd.DataFrame | None = None,
                       weights: ScoreWeights | None = None,
                       normalize: bool = True) -> pd.DataFrame:
    """Calcule un score global ∈ [0,1] par MMSI à partir des détecteurs.

    Chaque source contribue `w_i · c_i / Σw` (les contributions s'additionnent
    pour donner le score). Si `normalize`, le score final est ramené à [0,1]
    par min-max (transformation monotone → AUC inchangée, score interprétable).

    Returns: DataFrame [mmsi, score, top_reasons, + 11 colonnes de contributions]
    """
    weights = weights or ScoreWeights()
    n = len(ships)
    score = np.zeros(n)
    contrib = {col: np.zeros(n) for col in (
        "fake_flag", "name_change", "orphan", "ais_off", "position_mismatch",
        "speed", "spoofing_rules", "isolation_forest", "lof", "zone", "llm_tabular",
    )}
    mmsi_to_idx = {m: i for i, m in enumerate(ships["mmsi"])}

    def _add(col: str, mmsi: str, c: float) -> None:
        i = mmsi_to_idx.get(mmsi)
        if i is None:
            return
        contrib[col][i] = max(contrib[col][i], float(c))

    if fake_flag_report is not None:
        for r in fake_flag_report[fake_flag_report["predicted"] == -1].itertuples():
            _add("fake_flag", r.mmsi, r.confidence)

    if name_change_mmsi:
        for m in name_change_mmsi:
            _add("name_change", m, 1.0)

    if orphan_attribution is not None and "candidate_mmsi" in orphan_attribution:
        for r in orphan_attribution.itertuples():
            _add("orphan", r.candidate_mmsi, r.candidate_confidence)

    if ais_off_blocks is not None and "suspect" in ais_off_blocks:
        for r in ais_off_blocks[ais_off_blocks["suspect"]].itertuples():
            _add("ais_off", r.mmsi, min(1.0, r.duration_hours / 168.0))  # 168 h = 1 semaine

    if position_mismatch is not None and "distance_km" in position_mismatch:
        for r in position_mismatch.itertuples():
            _add("position_mismatch", r.mmsi, min(1.0, r.distance_km / 10.0))

    if spoofing_rules is not None and "confidence" in spoofing_rules:
        agg = spoofing_rules.groupby("mmsi")["confidence"].max()
        speed_agg = spoofing_rules[spoofing_rules["type"] == "Speed Anomaly"] \
            .groupby("mmsi")["confidence"].max()
        for m, c in agg.items():
            _add("spoofing_rules", m, c)
        for m, c in speed_agg.items():
            _add("speed", m, c)

    if iforest_scores is not None:
        agg = iforest_scores.groupby("mmsi")["if_score"].max()
        for m, c in agg.items():
            _add("isolation_forest", m, c)

    if lof_scores_df is not None:
        agg = lof_scores_df.groupby("mmsi")["lof_score"].max()
        for m, c in agg.items():
            _add("lof", m, c)

    if zone_scores is not None:
        agg = zone_scores.groupby("mmsi")["zone_score"].max()
        for m, c in agg.items():
            _add("zone", m, c)

    if llm_scores is not None and "llm_score" in llm_scores:
        agg = llm_scores.groupby("mmsi")["llm_score"].max()
        for m, c in agg.items():
            _add("llm_tabular", m, c)

    w_vec = {k: getattr(weights, k) for k in contrib}
    w_total = sum(w_vec.values()) or 1.0
    # Contributions pondérées et normalisées par la somme des poids → elles
    # s'additionnent pour donner `score` (∈ [0, ~]).
    weighted = {k: w_vec[k] * contrib[k] / w_total for k in contrib}
    for k in contrib:
        score += weighted[k]

    # Rescale min-max → score ∈ [0, 1] (transformation monotone : AUC inchangée,
    # rend le score interprétable / le seuil exploitable dans le dashboard).
    lo, hi = float(np.min(score)), float(np.max(score))
    if normalize and hi > lo:
        factor = 1.0 / (hi - lo)
        score = (score - lo) * factor
        weighted = {k: v * factor for k, v in weighted.items()}

    contributions = pd.DataFrame(weighted)
    contributions["mmsi"] = ships["mmsi"].values

    def _top_reasons(i: int, max_k: int = 3) -> str:
        items = sorted(
            ((k, weighted[k][i]) for k in weighted),
            key=lambda kv: kv[1], reverse=True,
        )[:max_k]
        return ", ".join(f"{k}={v:.2f}" for k, v in items if v > 0)

    out = pd.DataFrame({
        "mmsi": ships["mmsi"].values,
        "score": score,
        "top_reasons": [_top_reasons(i) for i in range(n)],
    })
    return out.merge(contributions, on="mmsi", how="left")


def evaluate_score(scores: pd.DataFrame, ships: pd.DataFrame,
                   cost_fp: float = COST_FP, cost_fn: float = COST_FN) -> dict:
    """ROC-AUC + courbe coût asymétrique (cours 4 §8) vs `is_suspicious`."""
    from sklearn.metrics import (
        precision_recall_curve,
        precision_recall_fscore_support,
        roc_auc_score,
        roc_curve,
    )

    y = ships.merge(scores, on="mmsi", how="left")["score"].fillna(0).to_numpy()
    label = ships["is_suspicious"].fillna(False).astype(bool).to_numpy()
    if label.sum() == 0:
        return {"warning": "Pas de label `is_suspicious` à True dans `ships` : éval impossible."}

    auc = roc_auc_score(label, y)
    fpr, tpr, thr_roc = roc_curve(label, y)
    prec, rec, thr_pr = precision_recall_curve(label, y)

    # Seuil minimisant le coût asymétrique
    best_cost, best_thr = float("inf"), 0.5
    for t in np.linspace(0.05, 0.95, 50):
        pred = y >= t
        fp = int(((~label) & pred).sum())
        fn = int((label & ~pred).sum())
        c = cost_fp * fp + cost_fn * fn
        if c < best_cost:
            best_cost, best_thr = c, float(t)

    pred = y >= best_thr
    p, r, f, _ = precision_recall_fscore_support(label, pred, average="binary",
                                                 zero_division=0)
    return {
        "auc": float(auc),
        "best_threshold": best_thr,
        "best_cost": float(best_cost),
        "precision_at_best": float(p),
        "recall_at_best": float(r),
        "f1_at_best": float(f),
        "fpr": fpr.tolist(), "tpr": tpr.tolist(),
        "precision_curve": prec.tolist(), "recall_curve": rec.tolist(),
    }
