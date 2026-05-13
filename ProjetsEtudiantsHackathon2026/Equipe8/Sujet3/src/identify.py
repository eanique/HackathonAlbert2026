"""Pipeline d'identification passive (Q12 — Sujet 3).

Entrée : une signature radio (dict).
Sortie : dict structuré {mmsi, confidence, suspect, raisons, novelty}.

Combine :
  - **k-NN d'identification** dans l'espace standardisé des profils navire.
  - **One-Class SVM en novelty detection** (cours 4 §3 & §5) :
    entraîné sur le « normal propre » (navires `is_suspicious=False`),
    flag si la nouvelle signature tombe hors de la frontière apprise.

L'objet `Identifier` est sérialisable via `joblib` pour la démo.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from .config import OCSVM_KERNEL, OCSVM_NU


@dataclass
class IdentificationResult:
    """Résultat structuré d'une identification."""
    mmsi: str | None
    confidence: float            # 1 / (1 + distance), ∈ (0, 1]
    suspect: bool                # vrai si la signature est « anormale »
    novelty_score: float         # score OCSVM (négatif = hors frontière)
    reasons: list[str]
    ship_info: dict


_NUM = ("frequency", "bandwidth", "power", "signal_strength", "signal_to_noise_ratio")
_CAT = ("modulation", "pulse_pattern")


def _make_X(radio: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Features riches pour le k-NN : numériques + one-hot catégoriels."""
    parts = [radio[[c for c in _NUM if c in radio.columns]].fillna(0)]
    for c in _CAT:
        if c in radio.columns:
            parts.append(pd.get_dummies(radio[c], prefix=c, dtype=float))
    X = pd.concat(parts, axis=1)
    return X.to_numpy(), list(X.columns)


class Identifier:
    """Pipeline d'identification passive : k-NN + One-Class SVM novelty.

    Entraîné sur **toutes les signatures individuelles** (radio_signatures_large),
    avec features enrichies (numériques + one-hot `modulation` × `pulse_pattern`).
    """

    K = 10  # k-NN voisinage (vote majoritaire pondéré)

    def __init__(self, radio: pd.DataFrame, ships: pd.DataFrame,
                 nu: float = OCSVM_NU, kernel: str = OCSVM_KERNEL,
                 exclude_signature_ids: list[str] | None = None):
        self.ships = ships.set_index("mmsi")[
            [c for c in ("name", "flag", "type", "is_suspicious") if c in ships.columns]
        ] if "mmsi" in ships.columns else pd.DataFrame()

        train_radio = radio
        if exclude_signature_ids:
            train_radio = radio[~radio["signature_id"].isin(exclude_signature_ids)]

        X, self.feature_cols = _make_X(train_radio)
        self.scaler = StandardScaler().fit(X)
        X_std = self.scaler.transform(X)

        self.knn = NearestNeighbors(n_neighbors=self.K).fit(X_std)
        self.profile_mmsi = train_radio["mmsi"].to_numpy()

        # Novelty Detection sur navires « normaux propres »
        susp_mmsi = set(ships.loc[ships.get("is_suspicious", False).astype(bool), "mmsi"])
        clean_mask = ~train_radio["mmsi"].isin(susp_mmsi)
        X_clean = X_std[clean_mask.to_numpy()] if clean_mask.any() else X_std
        if len(X_clean) > 2000:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_clean), 2000, replace=False)
            X_clean = X_clean[idx]
        self.ocsvm = OneClassSVM(kernel=kernel, nu=nu, gamma="scale").fit(X_clean)

    # ------------------------------------------------------------------
    # Inférence
    # ------------------------------------------------------------------

    def identify(self, signature: dict | pd.Series) -> IdentificationResult:
        """Une signature → MMSI candidat + diagnostic suspect.

        Vote majoritaire pondéré 1/distance parmi les k=10 voisins, sur
        features enrichies (one-hot modulation + pulse_pattern).
        """
        sig = pd.DataFrame([signature])
        # Reconstruit le vecteur dans le même espace que le training
        parts = [sig.reindex(columns=list(_NUM), fill_value=0).fillna(0)]
        for c in _CAT:
            if c in sig.columns:
                dum = pd.get_dummies(sig[c], prefix=c, dtype=float)
                parts.append(dum)
        x_df = pd.concat(parts, axis=1).reindex(columns=self.feature_cols, fill_value=0.0)
        x_std = self.scaler.transform(x_df.to_numpy())

        dist, idx = self.knn.kneighbors(x_std, n_neighbors=self.K)
        candidate_mmsis = self.profile_mmsi[idx[0]]
        weights = 1.0 / (1.0 + dist[0])
        votes: dict = {}
        for m, w in zip(candidate_mmsis, weights):
            votes[m] = votes.get(m, 0.0) + float(w)
        mmsi, vote_score = max(votes.items(), key=lambda kv: kv[1])
        conf = vote_score / weights.sum()

        # Novelty Detection
        ocsvm_pred = int(self.ocsvm.predict(x_std)[0])  # +1 = normal, -1 = nouveau
        novelty_score = float(self.ocsvm.decision_function(x_std)[0])
        is_novel = ocsvm_pred == -1

        reasons: list[str] = []
        ship_info: dict = {}
        if mmsi in self.ships.index:
            row = self.ships.loc[mmsi]
            ship_info = {
                "name": row.get("name"), "flag": row.get("flag"),
                "type": row.get("type"),
                "is_suspicious": bool(row.get("is_suspicious", False)),
            }
            if ship_info["is_suspicious"]:
                reasons.append(f"Navire {mmsi} listé `is_suspicious=True`.")
        if is_novel:
            reasons.append(f"Signature hors frontière OCSVM (score={novelty_score:.3f}).")
        if conf < 0.3:
            reasons.append(f"Confiance k-NN faible ({conf:.2f}) → MMSI candidat peu fiable.")

        return IdentificationResult(
            mmsi=mmsi,
            confidence=conf,
            suspect=is_novel or ship_info.get("is_suspicious", False),
            novelty_score=novelty_score,
            reasons=reasons,
            ship_info=ship_info,
        )

    # ------------------------------------------------------------------
    # Validation Q13
    # ------------------------------------------------------------------

    def validate(self, radio: pd.DataFrame, ships: pd.DataFrame, n: int = 10,
                 seeds=(1, 2, 3, 4, 5)) -> dict:
        """Validation **leave-one-out** : pour chaque signature testée, on refit
        l'`Identifier` SANS cette signature, puis on identifie. C'est la
        méthode honnête (pas de data leakage).

        Returns: {per_seed, mean, std, matrix:[[TP,FN],[FP,TN]]}
        """
        rates = []
        TP = FN = FP = TN = 0
        for s in seeds:
            sample = radio.sample(n=min(n, len(radio)), random_state=s)
            correct = 0
            for _, row in sample.iterrows():
                # leave-one-out
                ident = Identifier(
                    radio, ships,
                    exclude_signature_ids=[row["signature_id"]],
                )
                res = ident.identify(row.to_dict())
                truth_suspect = bool(ships.loc[ships["mmsi"] == row.get("mmsi"),
                                              "is_suspicious"].iloc[0]) \
                    if (ships["mmsi"] == row.get("mmsi")).any() else False
                if res.mmsi == row.get("mmsi"):
                    correct += 1
                # Matrice de confusion sur la suspicion prédite vs réelle
                if res.suspect and truth_suspect:
                    TP += 1
                elif not res.suspect and not truth_suspect:
                    TN += 1
                elif res.suspect and not truth_suspect:
                    FP += 1
                else:
                    FN += 1
            rates.append(correct / len(sample))
        return {
            "per_seed": rates,
            "mean": float(np.mean(rates)),
            "std": float(np.std(rates)),
            "matrix": [[TP, FN], [FP, TN]],
        }

    # ------------------------------------------------------------------
    # Persistance
    # ------------------------------------------------------------------

    def save(self, path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path) -> "Identifier":
        return joblib.load(path)
