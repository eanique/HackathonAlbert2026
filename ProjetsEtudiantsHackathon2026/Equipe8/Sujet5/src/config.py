"""Constantes & paramètres — chemins, taxonomie, seuils.

Toute valeur magique du projet vit ici. Importée par tous les autres modules.
Référencer `hypotheses.md` pour la justification de chaque seuil.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# === Chemins =========================================================
ROOT = Path(__file__).resolve().parent.parent          # sujet5/
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_IMAGES_REAL = ROOT / "data" / "images_real"
DATA_IMAGES_TRAIN = ROOT / "data" / "images_train"
OUTPUTS = ROOT / "outputs"
CACHE_DB = ROOT / "data" / "cache.sqlite"

for d in (DATA_PROCESSED, DATA_IMAGES_REAL, DATA_IMAGES_TRAIN, OUTPUTS):
    d.mkdir(parents=True, exist_ok=True)

# === Fichiers fournis ================================================
F_IMAGES_META = DATA_RAW / "images_metadata_large.csv"
F_ANNOTATIONS = DATA_RAW / "annotations_large.json"
F_DETECTIONS = DATA_RAW / "detection_results.csv"
F_MIL_ZONES = DATA_RAW / "military_zones.csv"
F_MEJ_IMAGES = DATA_RAW / "mise_en_jambe" / "images_metadata_small.csv"
F_MEJ_ANNOT = DATA_RAW / "mise_en_jambe" / "annotations_small.json"

# === Taxonomie =======================================================
# 13 catégories de la Généralisation. Hypothèse explicite : militaire ⟺ cat_id ∈ MILITARY_IDS.
# Voir hypotheses.md §A.
CATEGORIES_GEN = {
    1: "Navire civil",
    2: "Navire de guerre",
    3: "Frégate",
    4: "Destroyer",
    5: "Porte-avions",
    6: "Pétrolier",
    7: "Cargo",
    8: "Chalutier",
    9: "Sous-marin",
    10: "Croiseur",
    11: "Corvette",
    12: "Navire de soutien",
    13: "Bâtiment de débarquement",
}
MILITARY_IDS = {2, 3, 4, 5, 9, 10, 11, 12, 13}
CIVIL_IDS = {1, 6, 7, 8}

# === Seuils géospatiaux (P4) =========================================
# Rayon de proximité détection ↔ zone militaire (cf. hypotheses §B Q10).
MIL_ZONE_RADIUS_KM = 25.0

# === Détection / classification (P2/P3) ==============================
YOLO_CONF_OPTIQUE = 0.45
YOLO_CONF_SAR = 0.30
YOLO_IOU_NMS = 0.45
YOLO_IMGSZ = 640
SPLIT_SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train/val/test

# === Piste B : la chasse =============================================
# (base_name, (south, west, north, east) en degrés)
BASES_NAVALES = {
    "Toulon (FR)":         (43.05,  5.85,  43.15,   6.00),
    "Brest (FR)":          (48.33, -4.55,  48.42,  -4.40),
    "Norfolk (US)":        (36.92, -76.40, 37.00, -76.25),
    "Severomorsk (RU)":    (69.05, 33.35,  69.10,  33.45),
    "Sébastopol (RU/UA)":  (44.60, 33.45,  44.65,  33.55),
}
HUNT_WINDOW_DAYS = 30
HUNT_TILE_SIZE = 1024
HUNT_TILE_OVERLAP = 128
HUNT_MIN_SHIP_PX = 15  # navire de moins de 15 px sur scène 10 m = ignoré

# === Croisement AIS (Levier L2) ======================================
AIS_TIME_WINDOW_MIN = 30
AIS_DIST_WINDOW_KM = 5.0

# === Infrastructure critique sous-marine (Levier L4) =================
# Câbles télécom/électriques + gazoducs/oléoducs sous-marins (OSM Overpass).
# Précédents : Nord Stream (sep. 2022), Eagle S × EstLink (déc. 2024) —
# un cargo qui croise à < 2 km d'un câble est un signal de guerre hybride.
INFRA_DIST_KM = 2.0           # seuil d'alerte « anomalie type C »
INFRA_SEARCH_RADIUS_KM = 10.0  # rayon Overpass autour de chaque détection

# === OSINT — flux de presse navale & corroboration ===================
# Couche de CORROBORATION / CIBLAGE, jamais de détection primaire (cf. rapport §OSINT).
# Pas d'API X/Twitter (payante & bridée → mal alignée avec « automatisation & API »).
# Flux RSS libres (parsés avec feedparser si dispo, sinon parseur regex de secours).
OSINT_RSS_FEEDS = {
    "naval-news":         ("Naval News",           "https://www.navalnews.com/feed/"),
    "usni-news":          ("USNI News",            "https://news.usni.org/feed"),
    "gcaptain":           ("gCaptain",             "https://gcaptain.com/feed/"),
    "maritime-executive": ("The Maritime Executive", "https://maritime-executive.com/articles.rss"),
    "mer-et-marine":      ("Mer et Marine",        "https://www.meretmarine.com/fr/rss.xml"),
    "defense-news-naval": ("Defense News (Naval)", "https://www.defensenews.com/arc/outboundfeeds/rss/category/naval/?outputType=xml"),
}
# Chaînes Telegram OSINT navales (noms publics indicatifs — à confirmer le jour J).
# Lecture seule via Telethon, *compte dédié*. NON branché en flux automatique (cf. rapport).
OSINT_TELEGRAM_CHANNELS = {
    "navalnews":      ("Naval News",        "https://t.me/navalnews"),
    "Osinttechnical": ("OSINTtechnical",    "https://t.me/Osinttechnical"),
    "CovertShores":   ("Covert Shores",     "https://t.me/CovertShores"),
}
# Mots-clés de pertinence navale (filtrage FR/EN des items de flux).
OSINT_NAVAL_KEYWORDS = (
    "frigate", "destroyer", "carrier", "aircraft carrier", "corvette", "submarine",
    "cruiser", "warship", "navy", "naval", "fleet", "amphibious", "patrol vessel",
    "frégate", "porte-avions", "porte-aéronefs", "sous-marin", "croiseur",
    "marine nationale", "flotte", "bâtiment de guerre", "patrouilleur",
)
# Pages Wikipedia « composition de flotte » par base navale (vérité terrain L1).
OSINT_WIKI_FLEET_PAGES = {
    "Toulon (FR)":        "French Mediterranean Fleet",
    "Brest (FR)":         "French Atlantic Fleet",
    "Norfolk (US)":       "United States Fleet Forces Command",
    "Severomorsk (RU)":   "Northern Fleet",
    "Sébastopol (RU/UA)": "Black Sea Fleet",
}
OSINT_FEED_TTL_S = 600        # cache des flux d'actualité (10 min)
OSINT_WIKI_TTL_S = 86400      # cache des pages Wikipedia (24 h)
