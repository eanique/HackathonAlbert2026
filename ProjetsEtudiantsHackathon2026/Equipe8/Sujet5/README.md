# BDD-MinArm — Sujet 5 (livrable)

Hackathon Albert School mai 2026 · Ministère des Armées.
**Sujet 5 — Chasse aux navires de guerre par imagerie satellite.**

## Composition du groupe
Voir [`EQUIPE.md`](EQUIPE.md).

## Ce que contient ce dossier
- [`reponses_mise_en_jambe.py`](reponses_mise_en_jambe.py) — code Python répondant à la mise en jambe (`SujetsHackathon2026/Sujet5/MiseEnJambe/`).
- [`reponses_generalisation_detection_navires.py`](reponses_generalisation_detection_navires.py) — code Python répondant aux questions Q1→Q18 de la généralisation. C'est le livrable noté.
- [`rapport_generalisation_detection_navires.md`](rapport_generalisation_detection_navires.md) — **descriptions des opérations réalisées** + **résultats obtenus** (mAP, precision/recall, F1, temps d'exécution, benchmarks).
- [`src/`](src/) — modules métier (`config`, `load`, `p1_explore`, `p2_detection`, `p3_classify`, `p4_geospatial`, `hunt`, `ais_cross`, `osint_enrich`, `maps`) appelés par les `reponses_*.py`.
- [`outputs/`](outputs/) — **résultats matériels** :
  - `detections_carte.html`, `anomalies_carte.html` (cartes folium)
  - `intel_DET-007-00023.md` / `.pdf` (fiche de renseignement)
  - `q4_yolo_comparison.json`, `q4_yolo_finetune.json`, `q4_yolo_sar_finetune.json` (mAP YOLO)
  - `q7_metrics_binaire.csv`, `q7_roc_binaire.png`, `q7_feature_importance.{csv,png}` (classifieur militaire/civil)
  - `q13_pipeline_*.csv` (pipeline + alertes)
  - `q14_onnx_benchmark.csv` (latence ONNX vs PyTorch)
  - `q16_detectors_benchmark.csv`, `q17_classifiers_benchmark.csv`, `q18_*.{png,md}` (benchmarks)
  - `yolo_finetune_*.preview.jpg`, `yolo_sar_*.preview.jpg` (détections sur scènes Sentinel-1 réelles)
  - `piste_b_diagnostic.md`, `piste_b_toulon_yolo_coco_diag.jpg` (chasse réelle Toulon)
  - `claude_vision/` (LLM vision sur scènes satellites)
- [`docs/`](docs/) — `cadrage.md` (problème, périmètre, hypothèses), `hypotheses.md`, `plan-3-jours.md`, `mcp-setup.md`.
- [`requirements.txt`](requirements.txt) — dépendances figées.
- [`.env.example`](.env.example) — variables d'environnement (clés API : Copernicus / Sentinel Hub, Planet, AISStream, GFW, Mistral / Anthropic, OpenWeather).

## Comment lancer

Prérequis : Python 3.12 (le système macOS 3.9 est trop vieux). Recommandé : `uv` (`brew install uv`).

```bash
# 1. environnement isolé + dépendances
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt

# 2. (optionnel) configurer les clés API (imagerie réelle + LLM)
cp .env.example .env   # puis remplir ; sans clés, P1/P3/P4 tournent sur les fichiers fournis

# 3. exécution des livrables
python reponses_mise_en_jambe.py                       # mise en jambe sur le petit dataset
python reponses_generalisation_detection_navires.py    # Q1→Q18 (livrable noté)
```

## Démarche & critères de notation
Le rapport [`rapport_generalisation_detection_navires.md`](rapport_generalisation_detection_navires.md) couvre les 6 critères :
1. **Bibliothèques data science / IA** — pandas, NumPy, scikit-learn, PyTorch, ultralytics (YOLOv8 / YOLOv11), torchvision (Faster R-CNN), RT-DETR, ResNet50 (embeddings), ONNX Runtime (quantisation / accélération), pycocotools, OpenCV, rasterio, folium, Mistral LLM (génération de fiches de renseignement).
2. **Emploi des données** — fusion `images_metadata_large.csv` × `detection_results.csv`, conversion bboxes COCO normalisées → pixels, taxonomies harmonisées (8 vs 13 classes), augmentation (rotation/flip/luminosité), enrichissement OSINT (zones militaires OpenStreetMap, AIS Danish Maritime Authority).
3. **Précision & complétude** — mAP du détecteur (Piste A sur fichiers fournis + Piste B sur Sentinel-1 réel), precision/recall du classifieur binaire militaire/civil, accuracy multi-classes, analyse des erreurs, comparaison à la vérité terrain.
4. **Automatisation & API** — pipeline complet (charge → détecte → classifie → alerte), accélération ONNX, intégration Copernicus Data Space / Sentinel Hub / Planet / AISStream / OpenStreetMap, alerte automatique sur militaire en zone sensible.
5. **Collaboration** — répartition formalisée dans `EQUIPE.md`.
6. **Présentation** — rapport structuré, visualisations dans `outputs/`, dashboards interactifs (cf. `app5/dashboard/` ou repo source).

## Stratégie deux pistes assumée
- **Piste A — « répondre aux questions »** sur les fichiers fournis (P1/P4 directement ; P2/P3 substitués par un dataset public car les `satellite_*.jpg` référencés dans le repo n'existent pas).
- **Piste B — « la vraie opération de recherche »** : Sentinel-1 SAR + Sentinel-2 optique sur 3-5 bases navales réelles (Toulon, Brest, Severomorsk, Sébastopol, Zhanjiang, Norfolk…), zones militaires OpenStreetMap, vérité terrain Wikipedia/Jane's, croisement AIS pour révéler les **navires sombres**.

Détails et chiffres dans le rapport.
