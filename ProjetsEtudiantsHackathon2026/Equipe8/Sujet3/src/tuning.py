"""Tuning des poids du score multi-facteurs par régression logistique L2.

Au lieu de tâtonner manuellement, on apprend les poids optimaux à partir de
la **vérité terrain** `anomalies_large.csv` (96 MMSI explicitement anormaux,
plus solide que `is_suspicious` qui est trop bruité — 491/1000 = ~aléatoire).

Méthodologie (cours d'A. Bogroff, ML II + ML III §4 — évaluation business) :
  1. Features = les 11 contributions calculées par `anomaly_score.build_global_score`.
  2. Label   = `mmsi ∈ anomalies_large.mmsi`.
  3. Split   = stratifié 70/30 (déséquilibre 96/1000 → stratify obligatoire).
  4. Modèle  = `LogisticRegression(class_weight='balanced', C=...)` — la
               class_weight balanced compense le déséquilibre, C optimisé en CV.
  5. Métriques = AUC + precision@k + coût asymétrique (cours 4 §8).

Sortie : les **coefficients positifs** de la régression deviennent les nouveaux
poids du score (les coefficients négatifs sont remis à 0 — un détecteur qui
*anti-corrèle* avec la vérité serait probablement à supprimer).
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .anomaly_score import ScoreWeights

CONTRIB_COLS = [
    "fake_flag", "name_change", "orphan", "ais_off", "position_mismatch",
    "speed", "spoofing_rules", "isolation_forest", "lof", "zone", "llm_tabular",
]


def tune_weights(global_score: pd.DataFrame,
                 ships: pd.DataFrame,
                 anomalies: pd.DataFrame,
                 n_folds: int = 5,
                 C: float = 0.3,
                 random_state: int = 42,
                 plot_path=None) -> dict:
    """Apprend les poids optimaux par **régression logistique L2 + 5-fold CV**.

    Méthodo (ML II/III, cours d'A. Bogroff) :
      1. Standardisation (StandardScaler) — pour rendre les coefs comparables.
      2. 5-fold stratifié — déséquilibre 96/1000 (≈ 10 % de positifs) → stratify
         obligatoire (sinon des folds sans positif).
      3. LogisticRegression L2 (`C = 0.3` régularisation forte choisie a priori
         car données rares + 11 features → grand risque d'overfit).
      4. AUC out-of-fold (concaténation des prédictions test des 5 folds)
         = estimation honnête de la généralisation.
      5. **Poids moyens** sur les 5 folds, repassés en échelle des features
         d'origine puis clippés ≥ 0 et renormalisés à la somme des poids
         uniformes initiaux.

    Returns:
        {weights_optimal, auc_oof, auc_train_mean, precision_at_k_oof,
         coefs_signed_mean, fpr, tpr, ...}
    """
    truth = set(anomalies["mmsi"])
    df = ships[["mmsi"]].merge(global_score, on="mmsi", how="left").fillna(0)
    y = df["mmsi"].isin(truth).astype(int).to_numpy()

    cols = [c for c in CONTRIB_COLS if c in df.columns]
    X = df[cols].to_numpy()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                          random_state=random_state)

    oof_pred = np.zeros_like(y, dtype=float)
    auc_trains: list[float] = []
    coefs_per_fold: list[np.ndarray] = []
    inv_stds_per_fold: list[np.ndarray] = []

    for tr_idx, te_idx in skf.split(X, y):
        sc = StandardScaler().fit(X[tr_idx])
        X_tr = sc.transform(X[tr_idx])
        X_te = sc.transform(X[te_idx])
        model = LogisticRegression(
            penalty="l2", C=C, class_weight="balanced",
            max_iter=2000, random_state=random_state, n_jobs=-1,
        ).fit(X_tr, y[tr_idx])
        oof_pred[te_idx] = model.predict_proba(X_te)[:, 1]
        auc_trains.append(roc_auc_score(y[tr_idx],
                                        model.predict_proba(X_tr)[:, 1]))
        coefs_per_fold.append(model.coef_.ravel())
        inv_stds_per_fold.append(1.0 / sc.scale_)

    auc_oof = float(roc_auc_score(y, oof_pred))
    k = int(y.sum())
    prec_at_k_oof = float(y[np.argsort(-oof_pred)[:k]].sum() / max(k, 1))
    auc_train_mean = float(np.mean(auc_trains))

    # Coefs moyennés (sur l'échelle des features originales = coef × inv_std)
    coefs_native_per_fold = np.array(
        [c * inv for c, inv in zip(coefs_per_fold, inv_stds_per_fold)]
    )
    coefs_mean = coefs_native_per_fold.mean(axis=0)
    coefs_std = coefs_native_per_fold.std(axis=0)
    signed = dict(zip(cols, coefs_mean.tolist()))
    signed_std = dict(zip(cols, coefs_std.tolist()))

    # Sélection : on retient les contributions dont le coef moyen ≥ 0
    # ET dont l'intervalle de confiance ne croise pas 0 trop largement
    # (heuristique simple : coef_moyen / coef_std > 0.5).
    weights_kwargs = {}
    for col, m, s in zip(cols, coefs_mean, coefs_std):
        keep = (m > 0) and (s == 0 or m / s > 0.5)
        weights_kwargs[col] = float(max(m, 0.0)) if keep else 0.0

    s_total = sum(weights_kwargs.values()) or 1.0
    scale = 12.4 / s_total
    weights_kwargs = {k: round(v * scale, 3) for k, v in weights_kwargs.items()}
    weights_optimal = ScoreWeights(**weights_kwargs)

    fpr, tpr, _ = roc_curve(y, oof_pred)

    if plot_path is not None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"Tuné L2 5-fold OOF (AUC={auc_oof:.3f})",
                linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="hasard (AUC=0.5)")
        ax.set_xlabel("Taux de faux positifs")
        ax.set_ylabel("Taux de vrais positifs (rappel)")
        ax.set_title("ROC — score de suspicion vs anomalies_large.csv")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)

    return {
        "weights_optimal": weights_optimal,
        "auc_oof": auc_oof,
        "auc_train_mean": auc_train_mean,
        "precision_at_k_oof": prec_at_k_oof,
        "coefs_signed": signed,
        "coefs_std": signed_std,
        "n_pos": int(y.sum()),
        "n_features": len(cols),
        "C": C, "n_folds": n_folds,
        "fpr": fpr.tolist(), "tpr": tpr.tolist(),
    }


def pretty_print_tuning(result: dict) -> None:
    print(f"  Régression L2 — {result['n_folds']}-fold CV (C = {result['C']}) :")
    print(f"    AUC train (moy folds) : {result['auc_train_mean']:.3f}")
    print(f"    AUC out-of-fold       : {result['auc_oof']:.3f}")
    print(f"    precision@k OOF       : {result['precision_at_k_oof']:.2%} "
          f"(k = {result['n_pos']} vrais positifs)")
    print("  Coefficients moyens ± écart-type sur les folds :")
    sorted_items = sorted(result["coefs_signed"].items(), key=lambda kv: -kv[1])
    for col, v in sorted_items:
        s = result["coefs_std"].get(col, 0.0)
        arrow = "⬆" if v > 0 else "⬇"
        print(f"    {arrow} {col:>20s} : {v:+.3f} ± {s:.3f}")
    print("  Poids retenus (coef ≥ 0 et coef/std > 0.5, somme renormalisée = 12.4) :")
    for k, v in asdict(result["weights_optimal"]).items():
        print(f"    • {k:>20s} : {v}")
