"""Clustering K-Means K=5 + Silhouette + PCA 2D (Q3 — Sujet 3).

Pipeline rigoureux selon le cours d'A. Bogroff (cours 1 & 3) :

1. `StandardScaler` **obligatoire** sur (frequency, bandwidth, power) (cours 0 §4).
2. `KMeans(n_clusters=5, n_init=10, init='k-means++')` (cours 1 §2).
   K=5 est imposé par l'énoncé, mais on rapporte aussi :
     - l'**Elbow plot** (WCSS sur K ∈ [2, 10]),
     - le **Silhouette score** (cours 1 §3(c)) pour discuter la pertinence.
3. Visualisation : scatter `frequency × power` coloré par cluster + **PCA 2D**
   (biplot avec les loadings — cours 3 §2).

⚠️ Pièges (Memento cours, pièges #1, #2, #4) :
  - sans scaling, `frequency ∈ [156, 162]` MHz dominerait `bandwidth` ∈ [10, 50].
  - Voronoï : régions convexes → ne capture pas des formes en croissant. Sur
    nos 3 features (frequency, bandwidth, power), c'est OK (forme convexe).
  - K-Means converge vers un **min local**, d'où n_init=10 et K-Means++.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .config import KMEANS_FEATURES, KMEANS_K, KMEANS_N_INIT, OUTPUTS


def fit_kmeans(profiles: pd.DataFrame,
               k: int = KMEANS_K,
               features: Sequence[str] = KMEANS_FEATURES) -> dict:
    """Pipeline standard scaler + KMeans + Silhouette + PCA 2D.

    Args:
        profiles : DataFrame issu de profiles.build_profiles (1 ligne/navire).
                   Attendu : colonnes `freq_mean`, `bandwidth_mean`, `power_mean`
                   correspondant aux features choisies.
        k        : nombre de clusters (5 par défaut, imposé par l'énoncé).
        features : noms des colonnes à utiliser (alias via .replace).

    Returns:
        {"labels": np.ndarray, "centroids": np.ndarray, "wcss": float,
         "silhouette": float, "scaler": StandardScaler, "model": KMeans,
         "pca": PCA, "X_pca": np.ndarray, "X_std": np.ndarray}
    """
    # mapping feature name → colonne dans `profiles`
    col_map = {
        "frequency": "freq_mean",
        "bandwidth": "bandwidth_mean",
        "power": "power_mean",
    }
    cols = [col_map.get(f, f) for f in features]
    X = profiles[cols].dropna().to_numpy()

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    model = KMeans(n_clusters=k, n_init=KMEANS_N_INIT,
                   init="k-means++", random_state=42)
    labels = model.fit_predict(X_std)

    sil = silhouette_score(X_std, labels) if k > 1 else float("nan")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)

    return {
        "labels": labels,
        "centroids": model.cluster_centers_,
        "wcss": float(model.inertia_),
        "silhouette": float(sil),
        "scaler": scaler,
        "model": model,
        "pca": pca,
        "X_pca": X_pca,
        "X_std": X_std,
        "feature_cols": cols,
    }


def elbow_curve(profiles: pd.DataFrame, k_range: Sequence[int] = range(2, 11),
                features: Sequence[str] = KMEANS_FEATURES) -> pd.DataFrame:
    """Trace l'Elbow plot (cours 1 §3(b)) : WCSS et Silhouette pour K ∈ k_range.

    Returns: DataFrame [k, wcss, silhouette]
    """
    col_map = {"frequency": "freq_mean", "bandwidth": "bandwidth_mean", "power": "power_mean"}
    cols = [col_map.get(f, f) for f in features]
    X_std = StandardScaler().fit_transform(profiles[cols].dropna().to_numpy())
    rows = []
    for k in k_range:
        m = KMeans(n_clusters=k, n_init=KMEANS_N_INIT,
                   init="k-means++", random_state=42).fit(X_std)
        sil = silhouette_score(X_std, m.labels_) if k > 1 else float("nan")
        rows.append({"k": k, "wcss": float(m.inertia_), "silhouette": float(sil)})
    return pd.DataFrame(rows)


def plot_clusters_2d(result: dict, profiles: pd.DataFrame,
                     out: Path | None = None) -> Path:
    """Scatter freq × power + scatter PCA 2D, sauvegardé en PNG."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    labels = result["labels"]
    cols = result["feature_cols"]
    n = len(labels)
    df = profiles.iloc[:n]

    axes[0].scatter(df[cols[0]], df[cols[2]] if len(cols) > 2 else df[cols[1]],
                    c=labels, cmap="tab10", alpha=0.7, s=12)
    axes[0].set_xlabel(cols[0])
    axes[0].set_ylabel(cols[2] if len(cols) > 2 else cols[1])
    axes[0].set_title(f"K-Means K=5 — silhouette={result['silhouette']:.2f}")

    axes[1].scatter(result["X_pca"][:, 0], result["X_pca"][:, 1],
                    c=labels, cmap="tab10", alpha=0.7, s=12)
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    var = result["pca"].explained_variance_ratio_
    axes[1].set_title(f"PCA 2D — variance expliquée : PC1 {var[0]:.0%} · PC2 {var[1]:.0%}")

    fig.tight_layout()
    out = out or OUTPUTS / "clusters_kmeans.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
