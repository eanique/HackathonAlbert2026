# BDD-MinArm — Sujet 3 (livrable)

Hackathon Albert School mai 2026 · Ministère des Armées.
**Sujet 3 — Aide à l'identification des navires par analyse intelligente de leur radio signature.**

## Composition du groupe
Voir [`EQUIPE.md`](EQUIPE.md).

## Ce que contient ce dossier
- [`reponses_mise_en_jambe.py`](reponses_mise_en_jambe.py) — code Python répondant aux questions Q1→Q12 de la mise en jambe (`SujetsHackathon2026/Sujet3/MiseEnJambe/`).
- [`reponses_generalisation.py`](reponses_generalisation.py) — code Python répondant aux questions Q1→Q14 de la généralisation (`SujetsHackathon2026/Sujet3/Généralisation/`). C'est le livrable noté.
- [`rapport_generalisation.md`](rapport_generalisation.md) — **descriptions des opérations réalisées** + **résultats obtenus** (chiffres, métriques, visualisations).
- [`src/`](src/) — modules métier (`load`, `profiles`, `cluster`, `anomalies`, `anomaly_score`, `spoofing_rules`, `identify`, `osint_enrich`, `intel_report`, `llm`, `maps`, …) appelés par les `reponses_*.py`.
- [`outputs/`](outputs/) — **résultats matériels** : carte folium (`carte_anomalies.html`), graphe de connaissances (`knowledge_graph.html`), clusters K-Means PNG, courbes ROC, alertes PDF, tableaux de métriques CSV, `data_quality_report.md`, `results.json`.
- [`docs/`](docs/) — `cadrage.md` (problème, périmètre, hypothèses), `hypotheses.md` (seuils), `plan-3-jours.md`.
- [`notebooks/`](notebooks/) — versions Jupyter des `reponses_*.py` (exploration).
- [`SOURCES.md`](SOURCES.md) — sources OSINT mobilisées (avec URLs et licences).
- [`requirements.txt`](requirements.txt) — dépendances figées.
- [`.env.example`](.env.example) — variables d'environnement (clés API OSINT/LLM, toutes optionnelles).

## Comment lancer

Prérequis : Python 3.12 (le système macOS 3.9 est trop vieux). Recommandé : `uv` (`brew install uv`).

```bash
# 1. environnement isolé + dépendances
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt

# 2. (optionnel) configurer les clés OSINT/LLM
cp .env.example .env   # puis remplir ; sans clés, le pipeline tourne avec un fallback template

# 3. exécution des livrables
python reponses_mise_en_jambe.py    # Q1→Q12 sur le petit dataset
python reponses_generalisation.py   # Q1→Q14 sur le grand dataset (le livrable noté)
```

Les chemins vers les données fournies (`ships_large.csv`, `radio_signatures_large.csv`, `ais_data_large.csv`, `anomalies_large.csv`) sont résolus automatiquement par `src/config.py` (recherche du dossier `SujetsHackathon2026/Sujet3/` à la racine du repo).

## Démarche & critères de notation
Le rapport [`rapport_generalisation.md`](rapport_generalisation.md) couvre les 6 critères d'appréciation :
1. **Bibliothèques data science / IA** — pandas, NumPy, scikit-learn (K-Means, IsolationForest, EllipticEnvelope, LocalOutlierFactor, OneClassSVM, k-NN), PyOD, XGBoost, NetworkX (graphe de connaissances), folium, Mistral LLM, Jinja2 (fiche de renseignement).
2. **Emploi des données** — normalisation StandardScaler, nettoyage (types, doublons MMSI), liens entre fichiers (jointure radio/AIS/ships par `mmsi`), enrichissement OSINT (Equasis, UIT MMSI, OpenSanctions, RSS), labellisation via `is_suspicious` et `anomalies_large.csv` (vérité terrain).
3. **Précision & complétude** — precision/recall/F1 par type d'anomalie, AUC ROC, taux d'identification correcte du pipeline passif, comparaison de 5 détecteurs (cf. `outputs/pyod_benchmark.csv`).
4. **Automatisation & API** — pipeline reproductible (un seul `python reponses_generalisation.py`), 4 API OSINT externes intégrées, alerte PDF/email automatique, websocket AIS temps réel (`src/realtime_ais.py`).
5. **Collaboration** — répartition formalisée dans `EQUIPE.md` (2 streams : données/détection vs pipeline/renseignement).
6. **Présentation** — rapport structuré, visualisations dans `outputs/`, dashboards interactifs (cf. `v-final-minarm/` à la racine du repo pour la version pitch).

## Périmètre — ce qu'on ne fait PAS (et pourquoi)
Le RF fingerprinting physique (échantillons IQ bruts via SDR matériel) est **hors périmètre** : `radio_signatures_large.csv` contient des **métadonnées tabulaires** (4 colonnes numériques + 2 catégorielles), pas du signal radio. Notre approche est donc du **ML tabulaire rigoureux**, avec le SDR matériel cité comme extension future. Voir `docs/cadrage.md` §3.3.
