"""Classifieur supervisé XGBoost pour la prédiction de suspicion.

Justification (vs le LogReg L2 du module `tuning`) :
  - La consigne valorise « force de proposition sur les méthodes ».
  - On a **96 labels positifs** (`anomalies_large.csv`) — autant en profiter
    en supervisé plutôt qu'en linéaire seul.
  - GBM gère nativement les interactions non linéaires (« LOF haut ET AIS off
    ET dans une zone à risque ») que le LogReg rate par construction.
  - GBM gère le déséquilibre de classes via `scale_pos_weight` et est
    robuste à des features hétérogènes (numériques + binaires).

Entrée : `global_score` (1 ligne / MMSI avec les 11 contributions) +
         (optionnel) `radio_profile` (features brutes agrégées par navire).
Cible  : `mmsi ∈ anomalies_large.mmsi`.
Sortie : AUC OOF + precision@k + importance des features + prédictions OOF.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

CONTRIB_COLS = [
    "fake_flag", "name_change", "orphan", "ais_off", "position_mismatch",
    "speed", "spoofing_rules", "isolation_forest", "lof", "zone", "llm_tabular",
]


def _features_with_profiles(global_score: pd.DataFrame,
                             profiles: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str]]:
    """Joint les contributions au profil radio brut du navire (features additionnelles)."""
    df = global_score.copy()
    if profiles is not None:
        prof_num = [c for c in ("freq_mean", "freq_std", "bandwidth_mean", "bandwidth_std",
                                "power_mean", "power_std", "signal_strength_mean",
                                "snr_mean", "snr_std", "n_signatures",
                                "modulation_entropy", "pulse_pattern_entropy")
                    if c in profiles.columns]
        df = df.merge(profiles[["mmsi", *prof_num]], on="mmsi", how="left")
    feat_cols = [c for c in CONTRIB_COLS if c in df.columns]
    feat_cols += [c for c in df.columns if c.startswith(("freq_", "bandwidth_",
                                                          "power_", "signal_strength",
                                                          "snr_", "modulation_entropy",
                                                          "pulse_pattern_entropy",
                                                          "n_signatures"))]
    feat_cols = list(dict.fromkeys(feat_cols))   # déduplique en gardant l'ordre
    return df, feat_cols


def train_eval(global_score: pd.DataFrame,
               ships: pd.DataFrame,
               anomalies: pd.DataFrame,
               profiles: pd.DataFrame | None = None,
               n_folds: int = 5,
               random_state: int = 42,
               plot_path: Path | None = None) -> dict:
    """XGBoost + 5-fold CV stratifié, AUC out-of-fold + feature importance."""
    truth = set(anomalies["mmsi"])
    df = ships[["mmsi"]].merge(global_score, on="mmsi", how="left").fillna(0)
    df, feat_cols = _features_with_profiles(df, profiles)
    y = df["mmsi"].isin(truth).astype(int).to_numpy()
    X = df[feat_cols].fillna(0).to_numpy(dtype=float)

    pos = max(y.sum(), 1)
    neg = max((1 - y).sum(), 1)
    scale_pos_weight = neg / pos     # compense le déséquilibre 96/904

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                          random_state=random_state)

    oof = np.zeros(len(y))
    importance_sum = np.zeros(len(feat_cols))
    aucs_train: list[float] = []

    for tr, te in skf.split(X, y):
        model = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.5,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric="logloss",
            tree_method="hist",
            verbosity=0,
        )
        model.fit(X[tr], y[tr])
        oof[te] = model.predict_proba(X[te])[:, 1]
        aucs_train.append(roc_auc_score(y[tr], model.predict_proba(X[tr])[:, 1]))
        importance_sum += model.feature_importances_

    auc_oof = float(roc_auc_score(y, oof))
    k = int(y.sum())
    prec_at_k = float(y[np.argsort(-oof)[:k]].sum() / max(k, 1))
    importance = (importance_sum / n_folds).tolist()
    fpr, tpr, _ = roc_curve(y, oof)

    if plot_path is not None:
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        ax[0].plot(fpr, tpr, label=f"XGBoost (AUC OOF = {auc_oof:.3f})",
                   linewidth=2, color="#0a6")
        ax[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax[0].set_xlabel("Taux de faux positifs")
        ax[0].set_ylabel("Taux de vrais positifs")
        ax[0].set_title("ROC — XGBoost vs anomalies_large.csv")
        ax[0].grid(alpha=0.3)
        ax[0].legend(loc="lower right")
        # Importance
        order = np.argsort(importance)[-15:]
        ax[1].barh([feat_cols[i] for i in order],
                   [importance[i] for i in order], color="#0a6")
        ax[1].set_title("Top-15 feature importance (moyenne 5 folds)")
        ax[1].grid(alpha=0.3, axis="x")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)

    return {
        "auc_oof": auc_oof,
        "auc_train_mean": float(np.mean(aucs_train)),
        "precision_at_k_oof": prec_at_k,
        "k": k,
        "n_features": len(feat_cols),
        "feature_cols": feat_cols,
        "feature_importance": importance,
        "scale_pos_weight": scale_pos_weight,
        "fpr": fpr.tolist(), "tpr": tpr.tolist(),
        "oof_pred": oof.tolist(),
        "mmsi_order": df["mmsi"].tolist(),
    }


def pretty_print(result: dict) -> None:
    print(f"  XGBoost — 5-fold CV stratifié (scale_pos_weight = {result['scale_pos_weight']:.1f}) :")
    print(f"    AUC train (moy) : {result['auc_train_mean']:.3f}")
    print(f"    AUC out-of-fold : {result['auc_oof']:.3f}")
    print(f"    precision@k OOF : {result['precision_at_k_oof']:.2%} "
          f"(k = {result['k']} vrais positifs)")
    print("  Top-10 features par importance moyenne :")
    items = sorted(zip(result["feature_cols"], result["feature_importance"]),
                   key=lambda kv: -kv[1])[:10]
    for col, imp in items:
        bar = "█" * int(imp * 50)
        print(f"    {col:>26s} {bar} {imp:.3f}")
