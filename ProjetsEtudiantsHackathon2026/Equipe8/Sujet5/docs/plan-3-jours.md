# Plan d'exécution — Sujet 5 (mode 2 agents IA, 3 phases)

> Travail exécuté par **deux streams Claude parallèles** (Agent A = Détection/Classification Piste A, Agent B = Chasse réelle + Géospatial + Pipeline Piste B), orchestrés via le tool `Agent`. Les « 3 jours » deviennent **3 phases logiques** ; chaque phase = un point de synchronisation + revue croisée.
>
> **Règle de coupe en cas de retard** : on coupe d'abord (1) le levier L5 graphe, puis (2) Q15 météo (garder Nominatim seul), puis (3) le 3ᵉ détecteur du benchmark Q16, puis (4) la quantisation INT8 (garder l'export ONNX). On **ne coupe pas** : P1, P3 binaire, P4 (incl. carte), Q4 mAP, le pipeline, le tableau de chasse Piste B, le navire sombre, la fiche LLM.

---

## Phase 1 — Données, exploration, baseline pré-entraînée, accès Piste B

### Préalable (commun aux 2 agents, ~20 min)
- Relire `CLAUDE.md` (énoncé, données, pièges) + `cadrage.md` + `hypotheses.md`.
- Initialiser l'environnement : `uv venv && uv pip install -r requirements.txt`.
- Copier `SujetsHackathon2026/Sujet5/Généralisation/*` dans `sujet5/data/raw/`.
- Copier `SujetsHackathon2026/Sujet5/MiseEnJambe/*` dans `sujet5/data/raw/mise_en_jambe/`.
- Créer `.env` (à minima vide, sinon avec les clés déjà disponibles).

### Agent A — Données fournies, P1, Mise en jambe, baseline détecteur
1. **`src/config.py`** : chargement `.env`, constantes (chemins, taxonomie militaire, rayons, seuils, mapping `category_id → is_military`).
2. **`src/load.py`** : (i) lecture `images_metadata_large.csv` (parse `coordinates "lat,lon"` → tuple float, parse `cloud_cover` → float, parse `resolution "Xm"` → float m, validation `image_id` unique) ; (ii) lecture `annotations_large.json` (parser COCO, reconvertir `bbox` normalisé → px avec `image.width/height`, recalculer `area_px`) ; (iii) lecture `detection_results.csv` (parser `bbox` string → list[float]) ; (iv) lecture `military_zones.csv` (parse `coordinates`, parse `active` → bool). Loggage qualité (manquants, doublons, taxonomies par fichier).
3. **`reponses_mise_en_jambe.py`** : Q1→Q? du `MiseEnJambe/README.md` sur `*_small.csv` + `annotations_small.json` (~1 h, premier livrable).
4. **`src/p1_explore.py`** : Q1-Q3 :
   - Q1 : `value_counts(source).head(3)`, `mean(resolution_m)`, `(cloud_cover > 30).mean() × 100`.
   - Q2 : nb annotations militaires (taxonomie hypotheses § A), répartition % par `category`, taille moyenne bbox par catégorie **en pixels**.
   - Q3 : `merge(images_metadata, detection_results, on='image_id')`, `groupby(image_id).size().mean()`, `idxmax()`.
5. **Sanity check YOLO** : `model = YOLO('yolov8n.pt')` ; `model.predict('test_image.jpg')` sur une image quelconque (téléchargement Wikipedia d'une photo aérienne de port) → s'assurer que la pipeline YOLO tourne.

### Agent B — Accès Piste B, OSINT, LLM, infra
1. Créer les comptes/clés gratuits :
   - **Copernicus Data Space Ecosystem** (`COPERNICUS_USER` / `COPERNICUS_PASSWORD`) — pour Sentinel-1 SAR + Sentinel-2 optique.
   - **OpenWeatherMap** (clé gratuite, `OPENWEATHER_API_KEY`).
   - **Mistral API** (`MISTRAL_API_KEY`).
   - **Global Fishing Watch** (`GFW_API_TOKEN`, gratuit pour usage non commercial — utile pour gap events).
2. **`src/llm.py`** : interface unique `generate_intel_note(...)` avec backend Mistral, fallbacks Anthropic / Ollama / template Jinja2 via `LLM_BACKEND`. Cache des réponses dans `data/cache.sqlite` (clé = hash du prompt). **Mêmes signatures que `sujet3/src/llm.py`** → mutualisable.
3. **`src/maps.py`** : helpers folium (fond de carte, couches `detections/zones/sombres/alertes`, popup standardisé, MarkerCluster).
4. **`src/hunt.py`** (squelette + 1 base testée) :
   - `fetch_sentinel1_scene(bbox, date_range) → str (chemin GeoTIFF)` via `pystac-client` + Microsoft Planetary Computer (sans inscription) OU `sentinelhub` (avec compte Copernicus).
   - Tester sur **Toulon** (bbox approx `(43.05, 5.85, 43.15, 5.95)`, derniers 7 jours).
   - Inférence `YOLO('yolov8n.pt')` out-of-the-box pour sanity (ne classera pas les types correctement, mais doit trouver des points lumineux sur la rade) → preuve de vie.
5. **`src/osint_enrich.py`** (stubs avec interfaces fixées, implémentation finalisée en Phase 2) :
   - `nearest_port_name(lat, lon) → str` (Nominatim avec rate-limit 1 req/s + cache SQLite).
   - `military_zones_osm(lat, lon, radius_km=50) → list[Polygon]` (Overpass).
   - `weather_at(lat, lon, dt) → dict` (OpenWeatherMap historique).
   - `wrecks_nearby(lat, lon, radius_m=500) → list` (MCP `shom-wrecks`).
6. **`SOURCES.md`** + **`README.md`** initiaux (lancement, variables d'env, sources créditées avec URL + licence).

### Sync fin de Phase 1
- ✅ Réponses P1 (Q1-Q3) produites.
- ✅ Mise en jambe livrée.
- ✅ YOLO out-of-the-box tourne sur 1 image (sanity).
- ✅ Au moins **1 scène Sentinel-1 téléchargée** (Toulon) et passée dans YOLO → preuve de vie Piste B.
- ✅ Backend LLM choisi et `generate_intel_note("hello")` fonctionne (ou fallback template).
- ✅ Squelette du repo en place, `make all` ne plante pas (sections vides ok).
- ✅ Revue croisée : A vérifie hunt/llm/maps de B, B vérifie load/p1 de A.

---

## Phase 2 — Détection, Classification, Géospatial, Chasse réelle (cœur noté)

### Agent A — Détecteur + Classifieurs (P2 + P3)
1. **`src/p2_detection.py`** :
   - **Choix du dataset substitué** (selon ce que J1 a téléchargé) : **xView3-SAR** (recommandé, SAR baseline) OU **Airbus Ship Detection** Kaggle. Documenter le choix dans le code (`# Substitution : énoncé S5 ne fournit pas d'images, on entraîne sur <X> et applique la méthodologie demandée.`).
   - **Q4** : split 70/15/15 (`random_state=42`), entraînement `YOLO('yolov8n.pt')` `epochs=20-50`, `imgsz=640`. Reporter **mAP@0.5** et **mAP@[.5:.95]** (`model.val()`).
   - **Q5** : inférence sur 5 images du test split → precision/recall par image (matching IoU≥0.5).
   - **Q6** : ajout albumentations (`HorizontalFlip`, `RandomRotate90`, `RandomBrightnessContrast`, `HueSaturationValue`) + tuning `lr0 ∈ {1e-3, 5e-4, 1e-4}` ; reporter Δ mAP.
2. **`src/p3_classify.py`** :
   - **Q7 binaire** : features bbox + zone risk_level ; comparer `RandomForestClassifier(n_estimators=200)`, `SVC(kernel='rbf')`, `MLPClassifier(hidden=(64,32))` en **5-fold CV** ; rapport precision/recall/F1/ROC-AUC ; **on optimise le rappel** [Cours 4 §8].
   - **Q8 multi-classes** : sur **HRSC2016** (téléchargement J1 soir), embeddings ResNet50 pre-pool, tête `LogisticRegression` + `RandomForestClassifier`. Accuracy top-1 + top-3 ; matrice de confusion.
   - **Q9** : matrice de confusion + **t-SNE 2D** des embeddings colorés par classe vraie vs prédite ; lister les 3 paires les plus confondues + hypothèses (Frégate↔Destroyer, etc.).

### Agent B — Géospatial + Chasse réelle (Piste B) + Croisement AIS
1. **`src/p4_geospatial.py`** :
   - **Q10** : pour chaque détection (`detection_results.csv`), calculer la distance géodésique aux 20 zones militaires ; flag `in_military_zone` si distance < 25 km ET `mil.active == True`. **Pas de merge sur `zone_id`** (cf. piège hypotheses § B Q10). Carte folium avec **MarkerCluster** + couches par catégorie.
   - **Q11** : série temporelle par `zone_name` (rolling 7 j), détection de pics (`> µ + 2σ`), commentaire qualitatif sur 2-3 pics.
   - **Q12** : anomalies type A (militaire hors zone, haut risque seulement) + type B (civil en zone Critical) ; carte folium dédiée.
2. **`src/hunt.py`** (mode complet) :
   - Pour chaque base **Toulon, Brest, Severomorsk, Sébastopol, Norfolk** : récupérer 2-3 scènes Sentinel-1 et 1-2 Sentinel-2 sur les 30 derniers jours.
   - Découpage en tuiles 1024×1024 avec overlap 128 px.
   - Inférence YOLOv8 fine-tuné xView3 (si poids dispo) sinon YOLO DOTA.
   - Sortie : `data/processed/chasse.csv` (colonnes : `base, scene_id, source, date, n_detections, n_militaires_estimés`).
3. **`src/ais_cross.py`** (Levier L2 « navire sombre ») :
   - Pour chaque détection Piste B : requête AIS (Danish Maritime Authority CSV gratuit, fenêtre `timestamp ± 30 min`, bbox `± 5 km`) → liste des MMSI actifs au moment de la scène.
   - Détection « sombre » = aucun MMSI dans la fenêtre.
   - Sortie : `data/processed/navires_sombres.csv`.
4. **`src/osint_enrich.py`** (implémentation complète) + **`src/intel_report.py`** : pour chaque détection militaire en zone sensible OU chaque navire sombre, assembler `{port, météo, classe, AIS}` → fiche Mistral + PDF (`reportlab`).
5. **`src/p5_pipeline.py`** :
   - **Q13** : `pipeline.predict(image_path) → list[Detection]` + rapport markdown + alerte. Testé sur 10 images du test split.
   - **Q15** : intégration Nominatim + OpenWeatherMap + Overpass.

### Sync fin de Phase 2
- ✅ Détecteur YOLOv8 fine-tuné : mAP rapporté, Q4-Q6 codés et évalués.
- ✅ Classifieurs Q7-Q8 entraînés, matrice de confusion, t-SNE Q9.
- ✅ Carte folium Q10-Q12 ouvre dans un navigateur.
- ✅ **Tableau de chasse Piste B sur ≥ 3 bases** rempli.
- ✅ **≥ 1 navire sombre confirmé** (croisement AIS).
- ✅ Pipeline `pipeline.predict(image_path)` fonctionnel.
- ✅ Fiche PDF générée pour ≥ 1 détection (test live).
- ✅ Revue croisée + commit propre.

---

## Phase 3 — Optimisation, Benchmark, Rapport, Pitch

### Matin — Agent A
1. **Q14** (avec B) : export ONNX du YOLO entraîné, quantisation INT8, benchmark CPU PyTorch vs ONNX vs INT8 (10 inférences, médiane).
2. **`src/p6_benchmark.py`** :
   - **Q16** : YOLOv8 (déjà entraîné) vs Faster R-CNN ResNet50-FPN (torchvision pré-entraîné COCO, inférence directe + métriques sur le test substitué) vs RT-DETR-l ou EfficientDet (light, inférence seule). Tableau mAP × FPS.
   - **Q17** : RF vs SVM-RBF vs ResNet50+LogReg sur le binaire militaire/civil ; accuracy/F1/temps.
3. **Q18** : finalisation du tableau global (mAP, P, R, F1, temps, limites, pistes).
4. **Finaliser les visualisations** dans `outputs/` : `detections_carte.html`, `anomalies_carte.html`, `tableau_chasse.png`, `roc_binaire.png`, `confusion_multiclasses.png`, `tsne_erreurs.png`, `metrics_table.csv`, `alerte_<image_id>.pdf`.

### Matin — Agent B
1. **Finaliser la chasse Piste B sur les 5 bases** (compléter si certaines n'ont pas tourné J2).
2. **Levier L5 graphe** (si temps) : `networkx.DiGraph` `Navire ↔ Scène ↔ Zone ↔ Événement`, viz `pyvis` HTML.
3. **Assemblage `reponses_generalisation_detection_navires.py`** : une section par question Q1→Q18 (commentaire `### Q1`, `### Q2`, …) ; chaque section affiche/sauvegarde son résultat + note d'analyse critique en commentaire. Vérifier que `python reponses_generalisation_detection_navires.py` tourne **de bout en bout** depuis un env neuf. `Makefile` (`make all`, `make hunt`, `make report`).
4. Rédiger **`rapport_generalisation_detection_navires.md`** (plan du skill `livrable-hackathon`, aligné sur les 6 critères) :
   1. **Contexte & périmètre** (décision « 2 pistes » + OUT justifiés).
   2. **Données & nettoyage** *(critère 2)* — fournies + substitution (xView3 ou Airbus Ship) + Piste B (Copernicus) + OSINT (OSM, Wiki, AIS, SHOM, weather). URLs, licences, schémas, étapes.
   3. **Démarche & méthodes** *(critère 1)* — Q1→Q18, chaque algo cité (cours Bogroff pour classif tabulaire, baselines pré-entraînées pour CV).
   4. **Résultats** *(critère 3)* — tous les chiffres : mAP, P/R, accuracy multi-classes, **tableau de chasse**, **liste navires sombres**, ROC, matrice de confusion.
   5. **Automatisation & API** *(critère 4)* — pipeline, ONNX, Copernicus/OSM/Nominatim/Weather/GFW/SHOM/Mistral, alertes auto.
   6. **Limites & pistes** — résolution sub-métrique, fine-tune SAR plus profond, croisement avec le Sujet 3 « signature radio ».
   7. **Annexes** — dictionnaire de données, hypothèses (`hypotheses.md`), liens vers `outputs/`.

### Après-midi — ensemble (synchro)
- **Geler** les livrables : copie dans `ProjetsEtudiantsHackathon2026/BDD-MinArm-Sujet5/` (EQUIPE.md, reponses_*.py, rapport_*.md, README.md, requirements.txt, outputs/). Dérouler la checklist du skill **`livrable-hackathon`**.
- **Vérifications finales** : (i) `uv pip install -r requirements.txt` depuis zéro + `python reponses_generalisation_detection_navires.py` produit tout ; (ii) aucun fichier > 50 Mo committé (images Piste B en `data/images_real/` gitignored) ; (iii) aucune clé / token dans le code ; (iv) toutes les questions traitées ou listées « hors périmètre » dans le rapport ; (v) sources OSINT créditées (nom + URL + licence) ; (vi) `ruff check` propre ; (vii) `pytest` passe.
- **Slides + répétition pitch** (4-5 slides + démo, 5 min max) : Problème → Démarche & périmètre 2 pistes → **Démo live** (mappemonde → zoom Toulon → scène Sentinel-1 + bboxes → tableau de chasse → **moment navire sombre** : clic, AIS muet, fiche Mistral + PDF) → Résultats chiffrés (mAP X, accuracy multi-classes Y, **N navires militaires sur Z bases**, **≥3 sombres**, temps ONNX) → Limites & extensions. **Fallback** : screenshots + vidéo prêts.
- **Sync avec l'équipe Sujet 3** sur le récit commun et l'ordre de passage. Insertion de la **slide commune « navire sombre »** dans les deux decks.

---

## Récapitulatif (vue d'avion)

| Phase | Agent A — Détection/Classification (Piste A) | Agent B — Chasse/Géo/Pipeline (Piste B) |
|---|---|---|
| **1** | `config.py`, `load.py`, Mise en jambe, `p1_explore.py` (Q1-Q3), sanity YOLO | Comptes Copernicus/Mistral/GFW, `llm.py`, `maps.py`, **hunt.py squelette (Toulon)**, `osint_enrich.py` stubs, SOURCES.md/README |
| **2** | `p2_detection.py` (YOLO fine-tune Q4-Q6 sur xView3/Airbus), `p3_classify.py` (Q7 binaire, Q8 multi-classes ResNet50, Q9 t-SNE) | `p4_geospatial.py` (jointure distance Q10-Q12), `hunt.py` complet 5 bases, `ais_cross.py` navires sombres, `osint_enrich.py` complet, `intel_report.py` PDF, `p5_pipeline.py` |
| **3** | `p6_benchmark.py` (Q16-Q17), visualisations finales, ONNX + INT8 (Q14) | Finalisation chasse, **graphe L5** (option), assemblage `reponses_generalisation_detection_navires.py`, `Makefile`, **rapport** |
| **3 PM** | **ENSEMBLE** : gel des livrables, checklist `livrable-hackathon`, slides, répétition pitch, **sync avec équipe S3** (slide « navire sombre » commune) | |

---

## Conventions de versionnage (interne aux agents)

- Un commit par module finalisé, message au format `feat(QX): …` ou `chore: …`.
- Pas de push tant que `ruff check` n'est pas propre et qu'au moins 1 test pytest sur le module passe.
- Le rapport est versionné en continu (chaque chiffre traçable à une cellule du code).
- Fichiers lourds (`data/images_real/`, `data/images_train/`, poids YOLO `*.pt`, modèles ONNX) → **gitignored**, listés dans `README.md` avec script de téléchargement.
