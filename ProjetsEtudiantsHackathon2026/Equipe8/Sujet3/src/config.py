"""Configuration centrale du Sujet 3.

- Chemins (data, outputs, cache).
- Seuils métier (cf. docs/hypotheses.md).
- Mapping MID UIT → pays (3 premiers chiffres du MMSI).
- Chargement des variables d'environnement (.env).

Tous les seuils référencent une justification dans `docs/hypotheses.md` ou
le cours de M. Bogroff (cours ml/FICHE_REVISION_DATA.md).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# -- Chemins -----------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS = PROJECT_ROOT / "outputs"
CACHE_DB = PROJECT_ROOT / "data" / "cache.sqlite"

for d in (DATA_RAW, DATA_PROCESSED, OUTPUTS):
    d.mkdir(parents=True, exist_ok=True)

# -- Variables d'environnement ----------------------------------------------

load_dotenv(PROJECT_ROOT / ".env")

LLM_BACKEND = os.getenv("LLM_BACKEND", "template")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

EQUASIS_USERNAME = os.getenv("EQUASIS_USERNAME")
EQUASIS_PASSWORD = os.getenv("EQUASIS_PASSWORD")
OPENSANCTIONS_API_KEY = os.getenv("OPENSANCTIONS_API_KEY")
GFW_API_TOKEN = os.getenv("GFW_API_TOKEN")
AISSTREAM_API_KEY = os.getenv("AISSTREAM_API_KEY")

# -- Seuils métier (cf. docs/hypotheses.md) ----------------------------------

# Q3 — K-Means
KMEANS_K = 5                         # imposé par l'énoncé
KMEANS_N_INIT = 10                   # cours 1 §2
KMEANS_FEATURES = ("frequency", "bandwidth", "power")

# Q4 — EllipticEnvelope par pavillon (cours 4 §4)
FLAG_MIN_SHIPS = 30                  # nb min de navires/pavillon pour fitter MCD
FLAG_CONTAMINATION = 0.05            # par défaut sklearn

# Q7 — AIS off
AIS_OFF_HOURS = 24                   # imposé par l'énoncé
AIS_OFF_EXCLUDE_STATUS = {"Moored", "At Anchor"}

# Q8 — écart AIS↔radio
POS_MISMATCH_KM = 1.0                # imposé par l'énoncé
ASOF_TOLERANCE_MIN = 60 * 24         # tolérance du merge_asof = 1 jour (dataset synthétique
                                     # avec ~5 signatures/navire/an → tolerance large nécessaire)

# Q9 — changement brutal
FREQ_JUMP_MHZ = 1.0                  # |Δfrequency| > 1 MHz
SIGNAL_JUMP_DBM = 15.0               # |Δsignal_strength| > 15 dBm

# Vitesse implicite (règle de spoofing)
SPEED_IMPLICIT_KMH_MAX = 50.0        # > 50 km/h sauf militaires

# L3 — Isolation Forest / LOF (cours 4 §6 & §7)
IFOREST_CONTAMINATION = 0.02         # ≈ 100/5000 dans anomalies_large.csv
IFOREST_N_ESTIMATORS = 200
LOF_N_NEIGHBORS = 20
LOF_CONTAMINATION = 0.02

# L3 — One-Class SVM (Q12, novelty detection, cours 4 §5)
OCSVM_NU = 0.05                      # tolérance 5% d'outliers dans le training
OCSVM_KERNEL = "rbf"

# Coût asymétrique (cours 4 §8)
COST_FN = 10.0                       # rater un navire suspect (très coûteux pour Minarm)
COST_FP = 1.0                        # alerte à tort

# -- MID UIT (3 premiers chiffres du MMSI → pays) ----------------------------
# Liste partielle — à compléter au besoin. Source : UIT-R M.1371.
# https://www.itu.int/en/ITU-R/terrestrial/fmd/Pages/mid.aspx

UIT_MID = {
    # France métropolitaine + outre-mer
    "227": "France", "228": "France",
    # Panama
    "351": "Panama", "352": "Panama", "353": "Panama", "354": "Panama",
    "355": "Panama", "356": "Panama", "357": "Panama",
    "370": "Panama", "371": "Panama", "372": "Panama", "373": "Panama",
    # Liberia
    "636": "Liberia", "637": "Liberia",
    # Marshall Islands
    "538": "Marshall Islands",
    # Singapore
    "563": "Singapore", "564": "Singapore", "565": "Singapore", "566": "Singapore",
    # Malta
    "215": "Malta", "229": "Malta", "248": "Malta", "249": "Malta", "256": "Malta",
    # Bahamas
    "308": "Bahamas", "309": "Bahamas", "311": "Bahamas",
    # China
    "412": "China", "413": "China", "414": "China",
    # USA
    "338": "USA", "366": "USA", "367": "USA", "368": "USA", "369": "USA",
    # Denmark
    "219": "Denmark", "220": "Denmark",
}


def mmsi_country(mmsi: str) -> str | None:
    """Retourne le pays attribué (par MID UIT) ou None si MID inconnu."""
    if not mmsi or len(mmsi) < 3 or not mmsi.isdigit():
        return None
    return UIT_MID.get(mmsi[:3])
