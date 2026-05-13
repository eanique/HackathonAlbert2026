# Rapport de qualité des données — Sujet 3

## 1. Volumétrie
- `ships` : 1000 lignes × 18 colonnes
- `radio` : 5000 lignes × 13 colonnes
- `ais` : 10000 lignes × 11 colonnes
- `anom` : 100 lignes × 7 colonnes

## 2. Valeurs manquantes
- `ships` : {'historical_names': 249}

## 3. Doublons (lignes entières)
- `ships` : 0
- `radio` : 0
- `ais` : 0
- `anom` : 0

## 4. Contrôle des bornes (vs dictionnaires de données)
- Aucune valeur hors-bornes.

## 5. Valeurs catégorielles hors-liste
- Aucune.

## 6. Intégrité référentielle
- MMSI orphelins : radio=0, ais=0, anom=0
- Navires sans signature radio : 6 (['363951510', '435131361', '522451298', '534553120', '535117872', '675464001'])
- Navires sans donnée AIS : 0
- MMSI préfixés `FAKE` dans radio : 0

## 7. Formats d'identifiants
```json
{
  "mmsi_9digits_ships": true,
  "mmsi_9digits_radio": true,
  "mmsi_9digits_ais": true,
  "imo_7digits_ships": true,
  "name_is_placeholder": 1.0,
  "mmsi_known_mid_pct": 0.061
}
```

## 8. Temporel
```json
{
  "ships.last_ais_update": {
    "min": "2026-01-01 04:00:00+00:00",
    "max": "2027-01-01 02:00:00+00:00",
    "n_nat": 0
  },
  "radio.timestamp": {
    "min": "2026-01-01 02:00:00+00:00",
    "max": "2027-01-01 22:00:00+00:00",
    "n_nat": 0
  },
  "ais.timestamp": {
    "min": "2026-01-01 00:21:00+00:00",
    "max": "2027-01-01 21:00:00+00:00",
    "n_nat": 0
  },
  "anom.timestamp": {
    "min": "2026-01-01 00:00:00+00:00",
    "max": "2026-12-31 00:00:00+00:00",
    "n_nat": 0
  },
  "ais_temporal_regressions": 0
}
```

## 9. Tests de structure (le dataset est-il réaliste ?)
- V de Cramér `navigational_status` ↔ `status` : **0.0313** (≈ 0 ⇒ aucune structure ; le vrai AIS aurait V ≈ 1)
- V de Cramér `ais_active` ↔ `status` : **0.0101** ; `ais_active=True` : 50.1% (≈ 50 % ⇒ aléatoire)
- Vitesse implicite (distance géodésique / Δt entre points AIS consécutifs) : médiane **9.1 kn**, p95 **133.6 kn**, max **12338.0 kn** ⇒ les positions AIS sont **des points aléatoires, pas des trajectoires**.
- Distribution du nombre de noms historiques : {0: 249, 1: 253, 2: 240, 3: 258} (≈ uniforme 0/1/2/3 — templates `OLD_NAME_k`).

## 10. Plafond de rappel atteignable
- 16 anomalies « Speed » + 21 « Course » = 37 anomalies dont la valeur n'est **que dans la description** (`speed` plafonné à 30 kn, descriptions citant jusqu'à 49.2 kn).
- ⇒ **Rappel maximal atteignable depuis les tables fournies ≈ 63%** (sur 100 anomalies de référence).
- *Les valeurs de vitesse/cap des anomalies Speed/Course n'apparaissent que dans la description ; la colonne `speed` est plafonnée et les positions sont aléatoires (vitesse implicite non discriminante) ⇒ ces anomalies ne sont pas détectables depuis les tables fournies.*

## 11. Actions de nettoyage appliquées
- 6 navire(s) sans aucune signature radio → marqués `has_radio_signature=False` ; absents de ship_radio_profiles.csv ; utilisés comme signal de suspicion (« navire jamais entendu »).
- Le champ `name` (dataset large) est un placeholder dérivé du MMSI (NAVIRE-<4 derniers chiffres>) → on identifie par MMSI, le `name` n'apporte aucune information (≠ dataset small qui a de vrais noms).
- `historical_names` vide pour 249 navires = « jamais renommé » (n_names_historical=0), PAS une donnée manquante. Parsé proprement par split(',').
