# Équipe 8 — BDD-MinArm

Hackathon Albert School mai 2026 · **Ministère des Armées**.

## Composition
Voir [`EQUIPE.md`](EQUIPE.md).

## Livrables

Deux sujets traités en parallèle (récit commun : « rendre visible un navire qui veut rester invisible » — par ses émissions radio et depuis l'orbite).

### [`Sujet3/`](Sujet3/) — Identification navale par signature radio
- [`reponses_mise_en_jambe.py`](Sujet3/reponses_mise_en_jambe.py) — Q1→Q12 (mise en jambe).
- [`reponses_generalisation.py`](Sujet3/reponses_generalisation.py) — **livrable noté** Q1→Q14.
- [`rapport_generalisation.md`](Sujet3/rapport_generalisation.md) — descriptions des opérations + résultats chiffrés.
- [`outputs/`](Sujet3/outputs/) — carte folium, clusters K-Means, ROC, alertes PDF, métriques CSV, graphe de connaissances.
- Détails et instructions de lancement : [`Sujet3/README.md`](Sujet3/README.md).

### [`Sujet5/`](Sujet5/) — Chasse aux navires de guerre par imagerie satellite
- [`reponses_mise_en_jambe.py`](Sujet5/reponses_mise_en_jambe.py).
- [`reponses_generalisation_detection_navires.py`](Sujet5/reponses_generalisation_detection_navires.py) — **livrable noté** Q1→Q18.
- [`rapport_generalisation_detection_navires.md`](Sujet5/rapport_generalisation_detection_navires.md) — descriptions + résultats (mAP, P/R/F1, latence ONNX, benchmarks).
- [`outputs/`](Sujet5/outputs/) — cartes folium, previews Sentinel-1 réels (Toulon), benchmarks YOLO/RT-DETR/Faster R-CNN, ROC militaire/civil, fiche intel PDF, Claude Vision.
- Détails et instructions de lancement : [`Sujet5/README.md`](Sujet5/README.md).

## Démarche & critères de notation
Chaque sujet déroule les 6 critères dans son `rapport_*.md` :
1. **Bibliothèques data science / IA** (LLM, clustering, graphes, YOLO/RT-DETR, ONNX, …).
2. **Emploi des données** (normalisation, nettoyage, liens entre fichiers, enrichissement OSINT, labellisation).
3. **Précision & complétude** (precision/recall/F1, mAP, AUC, comparaison à la vérité terrain).
4. **Automatisation & API** (pipeline reproductible, API OSINT, alerte automatique).
5. **Collaboration** (répartition formalisée dans EQUIPE.md).
6. **Présentation** (rapport structuré, visualisations, dashboards).
