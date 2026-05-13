"""P3 — Classification militaire/civil (Q7) + multi-classes (Q8) + erreurs (Q9).

Q7 : classifieur binaire `is_military` à partir des features bbox + contexte zone.
     ATTENTION : on N'utilise PAS `category_id` ni `category_name` en feature —
     ce serait du data leakage (la cible `is_military` est dérivée du category_id).

Q8 (Phase 2 suite) : multi-classes via embeddings ResNet50 sur HRSC2016 → autre module.
Q9 (Phase 2 suite) : t-SNE des erreurs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend non-interactif (headless)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from . import config as C
from . import load as L


# ----------------------------------------------------------------------
# Feature engineering (Q7)

_RISK_LEVELS = ["Low", "Medium", "High", "Critical"]


def build_features(df_det: pd.DataFrame, df_meta: pd.DataFrame) -> pd.DataFrame:
    """Construit la matrice de features X + target y pour Q7 binaire.

    Sortie : DataFrame avec colonnes numériques uniquement (prêt pour sklearn)
    + une colonne `is_military` (target bool).
    """
    df = df_det.merge(
        df_meta[["image_id", "resolution_m", "cloud_cover", "lat", "lon"]],
        on="image_id",
        how="left",
    )

    # Géométrie bbox (en normalisé — agnostique de la taille de l'image)
    df["bbox_area_norm"] = df["bbox_w_norm"] * df["bbox_h_norm"]
    df["bbox_ratio_wh"] = df["bbox_w_norm"] / df["bbox_h_norm"].replace(0, np.nan)
    df["x_center_norm"] = df["bbox_x_norm"] + df["bbox_w_norm"] / 2
    df["y_center_norm"] = df["bbox_y_norm"] + df["bbox_h_norm"] / 2
    df["dist_to_img_center"] = np.sqrt(
        (df["x_center_norm"] - 0.5) ** 2 + (df["y_center_norm"] - 0.5) ** 2
    )

    # Risk level one-hot (les valeurs hors liste sont mises à 0)
    enc = OneHotEncoder(
        categories=[_RISK_LEVELS], handle_unknown="ignore", sparse_output=False
    )
    risk_onehot = enc.fit_transform(df[["risk_level"]].fillna(""))
    risk_cols = [f"risk_{lvl.lower()}" for lvl in _RISK_LEVELS]
    df_risk = pd.DataFrame(risk_onehot, columns=risk_cols, index=df.index)

    feat = pd.concat(
        [
            df[
                [
                    "bbox_w_norm",
                    "bbox_h_norm",
                    "bbox_area_norm",
                    "bbox_ratio_wh",
                    "x_center_norm",
                    "y_center_norm",
                    "dist_to_img_center",
                    "resolution_m",
                    "cloud_cover",
                ]
            ],
            df_risk,
        ],
        axis=1,
    )
    feat["is_military"] = df["is_military"].astype(int)
    return feat.dropna()


# ----------------------------------------------------------------------
# Q7 — entraînement & évaluation 5-fold CV

def _make_classifiers() -> dict:
    return {
        # RF : pas de scaling nécessaire — baseline interprétable
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=C.SPLIT_SEED, n_jobs=-1
        ),
        # SVM RBF : scaling obligatoire → pipeline
        "SVM_RBF": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svc", SVC(kernel="rbf", C=1.0, probability=True, random_state=C.SPLIT_SEED)),
            ]
        ),
        # MLP : scaling obligatoire → pipeline
        "MLP": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        max_iter=2000,
                        early_stopping=True,
                        random_state=C.SPLIT_SEED,
                    ),
                ),
            ]
        ),
    }


def q7_binary(feat_df: pd.DataFrame, out_dir: Path = C.OUTPUTS) -> dict:
    """Entraîne 3 classifieurs binaires en 5-fold CV.

    Métriques par classe « militaire » (positive=1). On optimise sur le RAPPEL
    (coût asymétrique : rater un militaire est plus grave qu'alerter à tort).
    """
    y = feat_df["is_military"].to_numpy()
    X = feat_df.drop(columns=["is_military"])
    feature_names = list(X.columns)
    X = X.to_numpy()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.SPLIT_SEED)
    rows = []
    rocs = {}
    importances = None

    for name, clf in _make_classifiers().items():
        # CV cross_val_predict pour avoir des prédictions out-of-fold
        y_pred = cross_val_predict(clf, X, y, cv=skf, n_jobs=-1)
        y_proba = cross_val_predict(clf, X, y, cv=skf, method="predict_proba", n_jobs=-1)[:, 1]

        p = precision_score(y, y_pred, pos_label=1, zero_division=0)
        r = recall_score(y, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y, y_pred, pos_label=1, zero_division=0)
        auc = roc_auc_score(y, y_proba)
        cm = confusion_matrix(y, y_pred)

        rows.append(
            {
                "classifier": name,
                "precision_mil": round(p, 4),
                "recall_mil": round(r, 4),
                "f1_mil": round(f1, 4),
                "roc_auc": round(auc, 4),
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            }
        )
        fpr, tpr, _ = roc_curve(y, y_proba)
        rocs[name] = (fpr, tpr, auc)

        # On capture l'importance des features sur RF (interprétable)
        if name == "RandomForest":
            clf_fit = RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=C.SPLIT_SEED, n_jobs=-1
            ).fit(X, y)
            importances = pd.Series(
                clf_fit.feature_importances_, index=feature_names
            ).sort_values(ascending=False)

    df_metrics = pd.DataFrame(rows).sort_values("recall_mil", ascending=False)

    # Sauvegardes
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / "q7_metrics_binaire.csv"
    df_metrics.to_csv(metrics_csv, index=False)

    # ROC plot
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (fpr, tpr, auc) in rocs.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("Q7 — Binaire militaire/civil · 5-fold CV")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    roc_png = out_dir / "q7_roc_binaire.png"
    fig.savefig(roc_png, dpi=120)
    plt.close(fig)

    # Feature importance plot (RF)
    importance_png = None
    if importances is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        importances.plot.barh(ax=ax)
        ax.invert_yaxis()
        ax.set_xlabel("Importance (RandomForest)")
        ax.set_title("Q7 — Importance des features (RandomForest)")
        fig.tight_layout()
        importance_png = out_dir / "q7_feature_importance.png"
        fig.savefig(importance_png, dpi=120)
        plt.close(fig)
        importances.to_csv(out_dir / "q7_feature_importance.csv", header=["importance"])

    return {
        "df_metrics": df_metrics,
        "best_recall_clf": df_metrics.iloc[0]["classifier"],
        "best_recall": float(df_metrics.iloc[0]["recall_mil"]),
        "best_f1_clf": df_metrics.sort_values("f1_mil", ascending=False).iloc[0][
            "classifier"
        ],
        "best_f1": float(df_metrics["f1_mil"].max()),
        "feature_importance": importances.to_dict() if importances is not None else None,
        "files": {
            "metrics_csv": str(metrics_csv.relative_to(C.ROOT)),
            "roc_png": str(roc_png.relative_to(C.ROOT)),
            "importance_png": str(importance_png.relative_to(C.ROOT))
            if importance_png
            else None,
        },
        "_note": (
            "Coût asymétrique [Cours 4 §8] : on classe par rappel décroissant. "
            "Features = bbox (w, h, area, ratio, position, distance au centre) + "
            "résolution image + cloud_cover + zone risk_level (one-hot). "
            "`category_id` exclu (data leakage : c'est la source de la cible)."
        ),
    }


# ----------------------------------------------------------------------
# Run P3 (Q7 uniquement en Phase 2 préliminaire ; Q8/Q9 = lots suivants)

def run_p3_q7() -> dict:
    df_meta = L.load_images_metadata()
    df_det = L.load_detections()
    feat = build_features(df_det, df_meta)
    return {"Q7": q7_binary(feat), "features_df": feat}
