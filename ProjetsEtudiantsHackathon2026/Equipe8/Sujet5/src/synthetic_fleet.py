"""Générateur de flotte synthétique — démo offline + tests reproductibles.

Repris et adapté depuis `Sujet5_Navires/generate_fleet.py` (BDD-MinArm). Sert :
    1. À tester le dashboard 3D sans dépendre du dataset hackathon (démo
       portable sur n'importe quelle machine, sans `make data`).
    2. À avoir un jeu de **100 navires sur 30 zones stratégiques mondiales**
       — utile pour la slide "vue globale" du pitch.

⚠️ **C'est volontairement DIFFÉRENT du dataset hackathon** : on ne triche pas
sur Q1-Q12 (toutes les réponses sont produites sur le CSV fourni `_large`).
Ce générateur fournit uniquement un *fallback* offline et une démo globale.

Usage:
    from src.synthetic_fleet import generate_demo_fleet
    df = generate_demo_fleet(n=100, seed=42)
    df.to_csv("data/processed/demo_fleet.csv", index=False)
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

import pandas as pd

from . import config as C

# ----------------------------------------------------------------------
# Zones stratégiques (catalogue OSINT — mêmes que Sujet5_Navires + nos bases)

_ZONES_CATALOG = [
    # CRITICAL
    {"lat": 22.3, "lon": 114.2, "zone": "Mer de Chine Méridionale", "risk": "critical", "country": "Chine"},
    {"lat": 44.6, "lon": 33.5, "zone": "Mer Noire — Sébastopol", "risk": "critical", "country": "Ukraine/Russie"},
    {"lat": 12.5, "lon": 43.5, "zone": "Détroit de Bab-el-Mandeb", "risk": "critical", "country": "Yémen"},
    {"lat": 27.0, "lon": 56.0, "zone": "Détroit d'Ormuz", "risk": "critical", "country": "Iran"},
    {"lat": 35.5, "lon": 36.0, "zone": "Côte syrienne — Tartous", "risk": "critical", "country": "Syrie"},
    # HIGH
    {"lat": 26.5, "lon": 50.5, "zone": "Golfe Persique", "risk": "high", "country": "Bahreïn"},
    {"lat": 37.9, "lon": 23.7, "zone": "Méditerranée orientale", "risk": "high", "country": "Grèce"},
    {"lat": 1.35, "lon": 103.8, "zone": "Détroit de Malacca", "risk": "high", "country": "Singapour"},
    {"lat": 15.0, "lon": 42.0, "zone": "Mer Rouge — Nord", "risk": "high", "country": "Arabie Saoudite"},
    {"lat": 69.0, "lon": 33.0, "zone": "Mer de Barents", "risk": "high", "country": "Russie"},
    {"lat": 34.5, "lon": 129.0, "zone": "Détroit de Corée", "risk": "high", "country": "Corée du Sud"},
    {"lat": 13.0, "lon": 109.0, "zone": "Mer de Chine méridionale Est", "risk": "high", "country": "Vietnam"},
    # MEDIUM
    {"lat": 36.9, "lon": -76.3, "zone": "Norfolk Naval Station", "risk": "medium", "country": "USA"},
    {"lat": 35.28, "lon": 136.87, "zone": "Yokosuka Naval Base", "risk": "medium", "country": "Japon"},
    {"lat": 32.7, "lon": -117.2, "zone": "San Diego Naval Base", "risk": "medium", "country": "USA"},
    {"lat": 4.0, "lon": 7.0, "zone": "Golfe de Guinée", "risk": "medium", "country": "Nigeria"},
    {"lat": 40.6, "lon": 29.0, "zone": "Mer de Marmara — Istanbul", "risk": "medium", "country": "Turquie"},
    {"lat": 55.0, "lon": 20.0, "zone": "Mer Baltique", "risk": "medium", "country": "Pologne"},
    # LOW
    {"lat": 43.1, "lon": 5.92, "zone": "Base navale de Toulon", "risk": "low", "country": "France"},
    {"lat": 48.4, "lon": -4.5, "zone": "Base navale de Brest", "risk": "low", "country": "France"},
    {"lat": 51.5, "lon": -0.1, "zone": "Manche — trafic commercial", "risk": "low", "country": "Royaume-Uni"},
    {"lat": -33.9, "lon": 151.2, "zone": "Sydney Harbour", "risk": "low", "country": "Australie"},
    {"lat": 25.8, "lon": -80.1, "zone": "Miami — Port commercial", "risk": "low", "country": "USA"},
    {"lat": 53.5, "lon": 9.9, "zone": "Hamburg — Port commercial", "risk": "low", "country": "Allemagne"},
    {"lat": 51.9, "lon": 4.5, "zone": "Rotterdam — Port commercial", "risk": "low", "country": "Pays-Bas"},
    {"lat": 1.28, "lon": 103.85, "zone": "Singapour — Port commercial", "risk": "low", "country": "Singapour"},
    {"lat": 31.2, "lon": 121.5, "zone": "Shanghai — Port commercial", "risk": "low", "country": "Chine"},
]

_CATEGORIES_WEIGHTED = {
    "Navire civil": 0.20, "Cargo": 0.22, "Pétrolier": 0.14, "Chalutier": 0.06,
    "Navire de guerre": 0.10, "Frégate": 0.08, "Destroyer": 0.05,
    "Porte-avions": 0.02, "Sous-marin": 0.03, "Croiseur": 0.04,
    "Corvette": 0.03, "Navire de soutien": 0.02, "Bâtiment de débarquement": 0.01,
}

_DANGER_REASONS = {
    "critical": [
        "Militaire sans AIS en zone contestée",
        "Mouvement suspect près d'infrastructure stratégique",
        "Escorte de sous-marin non identifié",
        "Activité en zone d'embargo",
    ],
    "high": [
        "AIS désactivé depuis > 48 h",
        "Pétrolier hors corridor en zone de tension",
        "Exercice non notifié",
    ],
    "medium": [
        "Changement de cap inhabituel",
        "Vitesse anormale pour la catégorie",
    ],
    "low": [],
}


def _weighted_choice(catalog: dict, rng: random.Random) -> str:
    keys = list(catalog.keys())
    weights = list(catalog.values())
    return rng.choices(keys, weights=weights, k=1)[0]


def _jitter(lat: float, lon: float, radius_deg: float, rng: random.Random) -> tuple[float, float]:
    return (
        round(lat + rng.uniform(-radius_deg, radius_deg), 4),
        round(lon + rng.uniform(-radius_deg, radius_deg), 4),
    )


def generate_demo_fleet(
    n: int = 100,
    seed: int = 42,
    start_date: datetime | None = None,
    span_days: int = 41,
) -> pd.DataFrame:
    """Génère un DataFrame de `n` détections synthétiques.

    Colonnes alignées sur `q13_pipeline_detections.csv` pour pouvoir alimenter
    le dashboard 3D sans aucun fichier hackathon.
    """
    rng = random.Random(seed)
    start_date = start_date or datetime(2026, 4, 1)
    rows = []
    for i in range(n):
        z = rng.choice(_ZONES_CATALOG)
        lat, lon = _jitter(z["lat"], z["lon"], 0.8, rng)
        cat = _weighted_choice(_CATEGORIES_WEIGHTED, rng)
        is_mil = cat in C.MILITARY_IDS_NAMES if hasattr(C, "MILITARY_IDS_NAMES") else (
            cat not in ("Navire civil", "Cargo", "Pétrolier", "Chalutier")
        )
        conf = round(rng.uniform(0.55, 0.99), 3)
        risk = z["risk"]
        # Alerte = militaire ET risk high/critical (avec un peu de bruit)
        alert = is_mil and risk in ("high", "critical") and rng.random() < (0.7 if risk == "critical" else 0.35)
        reasons = _DANGER_REASONS.get(risk, [])
        alert_reason = rng.choice(reasons) if alert and reasons else ""
        dt = (start_date + timedelta(days=rng.randint(0, span_days),
                                     hours=rng.randint(0, 23))).isoformat() + "Z"
        rows.append({
            "detection_id": f"DEMO-{i+1:03d}",
            "image_id": f"DEMO-IMG-{(i % 25) + 1:03d}",
            "category_id": None,
            "category": cat,
            "is_military": is_mil,
            "confidence": conf,
            "lat": lat,
            "lon": lon,
            "timestamp": dt,
            "in_military_zone": risk in ("high", "critical") and rng.random() < 0.6,
            "nearest_mil_zone_name": z["zone"],
            "nearest_mil_zone_risk": risk.capitalize(),
            "nearest_mil_zone_dist_km": round(rng.uniform(0.5, 30.0), 1),
            "alert": alert,
            "alert_reason": alert_reason,
            "source": rng.choice(["Sentinel-2", "Maxar", "Planet Labs", "COSMO-SkyMed", "Sentinel-1 RTC"]),
            "country": z["country"],
            "is_dark": (alert and rng.random() < 0.3),  # 30% des alertes sont sombres
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out_path = C.DATA_PROCESSED / "demo_fleet.csv"
    df = generate_demo_fleet(n=100, seed=42)
    df.to_csv(out_path, index=False)
    print(f"[synthetic_fleet] -> {out_path.relative_to(C.ROOT)} ({len(df)} navires)")
    print("\nRépartition risque :")
    print(df["nearest_mil_zone_risk"].value_counts().to_string())
    print(f"\nMilitaires : {df['is_military'].sum()} / {len(df)}")
    print(f"Alertes    : {df['alert'].sum()}")
    print(f"Sombres    : {df['is_dark'].sum()}")
