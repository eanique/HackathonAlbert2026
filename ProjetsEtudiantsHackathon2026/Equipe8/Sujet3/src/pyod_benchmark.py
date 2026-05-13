"""Benchmark de détecteurs d'anomalies (PyOD) au-delà du cours.

Le cours présente 4 détecteurs (EllipticEnvelope, OneClassSVM, IsolationForest,
LOF). On en benchmarke ici **5 de plus**, parmi les plus performants de la
littérature :

  - **ECOD** (Empirical CDF Outlier Detection, Li et al. 2022) — sans hyperparam.
  - **COPOD** (Copula-Based Outlier Detection, Li et al. 2020).
  - **HBOS** (Histogram-Based Outlier Score, Goldstein 2012) — très rapide.
  - **KNN** (k-NN distance to k-th neighbour) — baseline densité.
  - **PCA-OD** (reconstruction error sur PC) — anomalie = mal reconstruite.

Toutes évaluées vs `anomalies_large.csv` (AUC + precision@k).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.pca import PCA as PCAOD
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)


def _features(radio: pd.DataFrame) -> pd.DataFrame:
    """Mêmes features que pour iForest/LOF dans anomaly_score (one-hot inclus)."""
    num_cols = [c for c in ("frequency", "bandwidth", "power",
                            "signal_strength", "noise_level",
                            "signal_to_noise_ratio") if c in radio.columns]
    df = radio[num_cols].copy()
    if "modulation" in radio.columns:
        df = pd.concat([df, pd.get_dummies(radio["modulation"], prefix="mod",
                                          dtype=float)], axis=1)
    if "pulse_pattern" in radio.columns:
        df = pd.concat([df, pd.get_dummies(radio["pulse_pattern"], prefix="pp",
                                          dtype=float)], axis=1)
    return df.fillna(df.median(numeric_only=True))


def _score_per_mmsi(scores: np.ndarray, mmsi: pd.Series) -> pd.DataFrame:
    """Agrège un score par signature en max par MMSI."""
    df = pd.DataFrame({"mmsi": mmsi.values, "score": scores})
    return df.groupby("mmsi")["score"].max().reset_index()


def benchmark(radio: pd.DataFrame, ships: pd.DataFrame,
              anomalies: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Returns: DataFrame [model, contamination, auc, precision_at_k, time_s]."""
    import time

    truth = set(anomalies["mmsi"])
    X = _features(radio)
    X_std = StandardScaler().fit_transform(X.to_numpy())

    contamination = 0.02

    models = {
        "ECOD": ECOD(contamination=contamination),
        "COPOD": COPOD(contamination=contamination),
        "HBOS": HBOS(contamination=contamination),
        "KNN": KNN(contamination=contamination, n_neighbors=20),
        "PCA-OD": PCAOD(contamination=contamination, random_state=random_state),
    }

    rows: list[dict] = []
    for name, model in models.items():
        t0 = time.perf_counter()
        try:
            model.fit(X_std)
            sigscores = model.decision_function(X_std)
            per_mmsi = _score_per_mmsi(sigscores, radio["mmsi"])
            # On scorre tous les navires (les MMSI sans signature → 0)
            full = ships[["mmsi"]].merge(per_mmsi, on="mmsi", how="left").fillna(0)
            y = ships["mmsi"].isin(truth).astype(int).to_numpy()
            score = full["score"].to_numpy()
            auc = roc_auc_score(y, score)
            k = int(y.sum())
            top_k_idx = np.argsort(-score)[:k]
            prec_at_k = float(y[top_k_idx].sum() / max(k, 1))
            rows.append({
                "model": name,
                "contamination": contamination,
                "auc": float(auc),
                "precision_at_k": prec_at_k,
                "time_s": round(time.perf_counter() - t0, 2),
            })
        except Exception as exc:  # pragma: no cover
            rows.append({
                "model": name, "auc": float("nan"),
                "precision_at_k": float("nan"),
                "time_s": round(time.perf_counter() - t0, 2),
                "error": str(exc),
            })
    return pd.DataFrame(rows).sort_values("auc", ascending=False)


def pretty_print(df: pd.DataFrame) -> None:
    print("  PyOD benchmark (signatures radio enrichies — same features que iForest/LOF) :")
    print(df.to_string(index=False))
