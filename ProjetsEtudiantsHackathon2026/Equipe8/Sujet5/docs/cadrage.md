# Cahier des charges — Sujet 5 : Traque de navires militaires par imagerie satellite

> Équipe **BDD-MinArm** · Hackathon Albert School mai 2026 · Commanditaire Ministère des Armées. Doc de cadrage **formel** (≈ PRD), issu du skill `cadrage-hackathon`. Daté du 2026-05-11. **À mettre à jour au fil de l'avancée.** Référentiel ML pour la partie tabulaire/classification : cours « Unsupervised Learning Problems » d'Alexis Bogroff, Albert School (`cours ml/FICHE_REVISION_DATA.md`). Référentiel CV : modèles pré-entraînés (YOLOv8/11, Faster R-CNN, ResNet50) + dataset substitué (xView3-SAR / Airbus Ship / HRSC2016).

---

## 0. Mode d'exécution — « 2 agents IA » (Claude)

Le travail est exécuté par **deux streams parallèles tenus par Claude (même modèle, joué sur 2 rôles)**, orchestrés via le tool `Agent` pour les briques indépendantes :

| Agent | Périmètre | Modules |
|---|---|---|
| **Agent A — « Détection & Classification » (Piste A)** | Charger les données fournies, P1 (exploration), P2 (détection sur dataset substitué), P3 (classification militaire/civil + multi-classes), P6 light (benchmark), évaluation rigoureuse | `load.py`, `p1_explore.py`, `p2_detection.py`, `p3_classify.py`, `p6_benchmark.py`, viz |
| **Agent B — « Chasse réelle + Géospatial + Pipeline » (Piste B)** | P4 (géospatial + temporel), P5 (pipeline + API externes), **Piste B = la vraie chasse aux navires sur imagerie Sentinel-1/Maxar**, croisement AIS, fiche LLM, alerte, rapport | `hunt.py`, `p4_geospatial.py`, `p5_pipeline.py`, `maps.py`, `osint_enrich.py`, `intel_report.py`, `llm.py`, `Makefile`, `rapport_generalisation_detection_navires.md` |

**Règles d'orchestration** : (i) chaque agent travaille dans son périmètre, fonctions pures aux signatures clairement spécifiées (cf. § 9) ; (ii) point de synchronisation en fin de chaque phase = revue croisée + tests d'intégration ; (iii) un seul propriétaire par module ; (iv) chaque livraison passe par `ruff` + un mini test pytest avant intégration ; (v) le rapport est assemblé en fin par Agent B à partir des notes laissées par les deux agents.

> **Garde-fou de rigueur** : tout choix de modèle / seuil / métrique cite (a) le cours [Cours N §M] pour la partie tabulaire (classifieur militaire/civil) OU (b) la baseline officielle / le papier de référence (xview3-reference, YOLO, ResNet, COCO mAP) pour la partie CV. Recherches papier listées dans `BDD-MinArm/hackathon-2026/docs/recherche-arxiv-sujets-3-5.md`.

---

## 1. Contexte & problème

À partir d'images satellites publiques, détecter et classifier des navires sur la mer ou dans les ports, **distinguer civils vs militaires**, et **identifier les types** (frégate, destroyer, porte-avions, sous-marin, …). L'énoncé prend la forme d'une **véritable opération de recherche** : trouver le **maximum de navires militaires** dans des zones stratégiques (bases navales, mers contestées) — *« l'équipe gagnante est celle qui en identifie le plus en démontrant la pertinence de sa méthode »*. Livrables imposés : `reponses_mise_en_jambe.py`, `reponses_generalisation_detection_navires.py`, `rapport_generalisation_detection_navires.md`.

**Particularité piégeuse** : aucune image n'est fournie dans le repo. `images_metadata_large.csv` référence des `satellite_000.jpg`…`satellite_099.jpg` qui n'existent pas. Les `url` sont `https://example.com/...` (décor). → **stratégie « 2 pistes » obligatoire** (cf. § 3).

**Critères de notation à viser** (consigne officielle) : (1) usage de libs DS/IA (NLP, **LLM**, automatisation, **graphes**, viz) · (2) qualité des données (normalisation, nettoyage, enrichissement, labellisation) · (3) précision & complétude des résultats · (4) **automatisation & API** · (5) collaboration · (6) clarté & professionnalisme. **Le jury survalorise la découverte de sources OSINT pertinentes** et la **restriction intelligente du périmètre**.

---

## 2. Le « woah » visé (en une phrase)

> Un **tableau de chasse réel** sur 5 bases navales (Toulon, Brest, Severomorsk, Sébastopol, Norfolk) à partir d'imagerie Copernicus Sentinel-1 SAR + Maxar Open Data, **enrichi d'un « navire sombre démasqué »** : un bâtiment détecté par satellite, **classé frégate** par notre pipeline, mais **invisible à l'AIS** au même moment (croisé avec Danish Maritime Authority / Global Fishing Watch) — accompagné d'une **fiche de renseignement générée par un LLM (Mistral)** + alerte PDF. **Le satellite voit ce que l'AIS cache.** ← pont narratif avec l'équipe Sujet 3 (« rendre visible un navire qui veut rester invisible »).

---

## 3. Périmètre IN / OUT (la restriction est explicitement valorisée)

### 3.0 Stratégie « 2 pistes » (assumée et présentée comme un choix)

- **Piste A — « répondre aux questions »** : P1, P3 (binaire militaire/civil), P4 (géospatial), P5 (pipeline), P6 light se font **sur les CSV/JSON synthétiques fournis**. Pour P2 (entraînement détecteur) et P3 multi-classes, comme **aucune image n'est fournie**, on **substitue un dataset public réel** (xView3-SAR pour la détection SAR / Airbus Ship Detection Kaggle pour la détection optique / HRSC2016 ou ShipRSImageNet pour la classification de **types**) et on applique la même méthodologie. **Choix documenté dans le rapport.**
- **Piste B — « la vraie opération de recherche »** : récupérer de **l'imagerie réelle** (Copernicus Data Space → **Sentinel-1 SAR**, Sentinel-2 optique 10 m ; Maxar Open Data 0,5 m ; IGN BD ORTHO 20 cm sur bases FR), passer un détecteur pré-entraîné, classer, **chasser sur 3-5 bases navales réelles**. Zones militaires réelles via **OSM `military=naval_base`** (Overpass). Vérité terrain par Wikipedia/Jane's. **Croisement AIS** (Danish Maritime Authority / GFW) → navires « sombres ».

### 3.1 IN — exigences fonctionnelles (les 18 questions de la Généralisation)

| ID | Question (synthèse) | Méthode mobilisée | Piste |
|---|---|---|---|
| **Q1** | Top 3 sources d'images, résolution moyenne, % cloud>30 dans `images_metadata_large.csv` | `pandas.value_counts`, `mean`. Réponses chiffrées + commentaire sur l'**incohérence physique source × résolution** (Sentinel-2 réel = 10 m **toujours**) | A |
| **Q2** | Nb navires militaires annotés, répartition par type %, taille moyenne bbox par type | Parse `annotations_large.json` (COCO), join `categories`, reconvertir bbox normalisée → px avec `width/height` de l'image | A |
| **Q3** | Fusion `images_metadata` × `detection_results`, nb détections/image, image max | `merge(on='image_id')`. Commentaire : `detection_results.csv` est ≈ une recopie de `annotations_large.json` → fusion partiellement triviale | A |
| **Q4** | Entraîner YOLOv8/Faster R-CNN sur `annotations_large.json`, split 70/15/15, mAP | **Piste A** : entraîner **YOLOv8n** (Ultralytics) sur **xView3-SAR ou Airbus Ship** (substitution documentée), 50 epochs, métrique `mAP@0.5` et `mAP@[.5:.95]`. **Piste B** : YOLO pré-entraîné DOTA + fine-tune léger | A + B |
| **Q5** | Appliquer à 5 images non vues + precision/recall | Inférence sur 5 images du test set ; comparaison avec annotations ; precision/recall par image | A |
| **Q6** | Optimisation (augmentation, hyperparams) → gain de mAP | Albumentations (rotation, flip, brightness, **HSV jitter**, mosaic), tuning `lr0`, `imgsz`, `batch` ; rapporter le **delta de mAP** | A |
| **Q7** | Classifieur binaire militaire/civil sur features (taille bbox, ratio, position) | **Random Forest** + **SVM** + **MLP** ; features tirées de `detection_results.csv` (bbox normalisée, ratio, distance au centre, `image.width × image.height`, contexte zone). Évaluation : precision/recall/F1, ROC. `is_military` comme target [Cours classif §RF/§SVM] | A |
| **Q8** | Classifieur multi-classes (frégate/destroyer/porte-avions) + embeddings ResNet50 | Sur **HRSC2016** (ou ShipRSImageNet) : extraire embeddings ResNet50 (pre-pool 2048-d), concaténer aux features bbox, RF / linear head. Accuracy + matrice de confusion + top-k accuracy | A |
| **Q9** | Analyse des erreurs (Frégate → Cargo, etc.) | Matrice de confusion sur le test, **t-SNE/UMAP** des embeddings ResNet50 colorés par classe **vraie** vs **prédite**, identification des paires confondues, hypothèses (taille similaire → frégate↔destroyer) | A |
| **Q10** | Détections en zones militaires + carte folium | Jointure géospatiale **par distance** (PAS par `zone_id` — cf. piège § 6.3) : `geopy.distance.geodesic` entre `(detection.coordinates)` et chaque `military_zones.coordinates` ; flag si distance < rayon (par défaut **25 km**). Carte folium MarkerCluster, popup avec catégorie + confidence | B |
| **Q11** | Évolution temporelle par zone, pics, corrélation événements géopolitiques | Time series `groupby(['zone_name', date])`, `rolling(7).sum()`, détection de pics > µ+2σ ; corrélation manuelle avec un calendrier d'événements (exercices OTAN, dates clés) | B |
| **Q12** | Anomalies : civil en zone militaire, militaire hors zone | Croisement `is_military` × `in_military_zone` ; tableau des incohérences + carte folium dédiée (couleur rouge) | B |
| **Q13** | Pipeline complet (charge image → détecte → classifie → rapport → alerte) | `pipeline.py` : `predict(image_path) → list[Detection]` ; rapport markdown + PDF (`reportlab`) ; alerte si militaire détecté dans zone `risk_level ∈ {High, Critical}` ET `active==True`. **Testé sur 10 nouvelles images** | B |
| **Q14** | Optim perfs : ONNX, quantisation, temps | Export `model.export(format='onnx')` ; mesurer temps inférence PyTorch vs ONNX vs ONNX-quantisé (`onnxruntime` INT8) ; gain attendu ×3-5 sur CPU | B |
| **Q15** | API externes : géoloc + météo | Nominatim (`geopy.geocoders.Nominatim`) → nom de port le plus proche ; **OpenWeatherMap** historique → conditions à la date de la scène ; **Overpass OSM** → polygone de la base navale | B |
| **Q16** | Benchmark 3 détecteurs (YOLOv8 / Faster R-CNN / RT-DETR ou EfficientDet) | Tableau mAP × FPS sur le test substitué. Justifier le « meilleur compromis » | A |
| **Q17** | Benchmark 3 classifieurs (RF / SVM / ResNet50 fine-tuné) | Accuracy + temps + interprétabilité | A |
| **Q18** | Rapport perf global (mAP, P, R, F1, temps), limites, pistes | Section dédiée du rapport ; checklist du skill `livrable-hackathon` | A + B |

### 3.2 IN — Leviers (valeur ajoutée, 0 €)

- **Levier L1 (La vraie chasse)** — cœur de différenciation jury : **Piste B**, tableau de chasse réel sur **3-5 bases navales** (Toulon, Brest, Severomorsk, Sébastopol, Norfolk), scènes Sentinel-1/2 réelles, comptes ouverts J1.
- **Levier L2 (Navire sombre)** — croisement satellite × AIS réel (Danish Maritime Authority CSV gratuit / Global Fishing Watch API) sur les détections Piste B → liste des détections **sans correspondance AIS** au même `(timestamp ± 30 min, bbox ± 5 km)`. **Pont narratif explicite avec l'équipe Sujet 3.**
- **Levier L3 (Fiche LLM Mistral)** — pour chaque détection militaire en zone sensible, génération d'une **fiche de renseignement** (port, classe estimée, conditions météo, événements géopolitiques contextuels, présence/absence AIS) par **Mistral** (`mistral-large-latest`) + **PDF** (`reportlab`). Aligné avec la convention du Sujet 3 (souveraineté FR).
- **Levier L4 (Anti-faux-positifs)** — MCP `shom-wrecks` pour écarter les épaves connues ; OSM `landuse=industrial/harbour` pour filtrer les structures portuaires fixes.
- **Levier L5 (Graphe)** — petit `networkx` **navire ↔ scène ↔ zone ↔ événement** + viz `pyvis`. Coche le critère « graphes de connaissance ».

### 3.3 OUT — explicitement exclu et justifié dans le rapport

- ❌ **Entraînement from scratch** d'un détecteur YOLO ou d'un Faster R-CNN : pas le temps, pas de GPU local. → **fine-tuning** d'un pré-entraîné DOTA / COCO uniquement, sur dataset substitué documenté.
- ❌ **Entraînement sur les `satellite_*.jpg` fournis** : les images n'existent **pas** dans le repo. La substitution est annoncée comme un choix méthodologique.
- ❌ **Détection sub-métrique sur Sentinel-2** : la résolution 10 m de Sentinel-2 permet de **détecter** un navire (porte-avions ~300 m = 30 px, frégate ~150 m = 15 px) mais **ne permet pas** de classifier porte-avions vs cargo sans confusion. → classification de type uniquement sur Maxar (0,5 m) / IGN BD ORTHO (20 cm). **Limite assumée.**
- ❌ **API payantes** (Maxar Pro, Airbus OneAtlas) : on reste sur du gratuit (Copernicus Data Space gratuit, Maxar Open Data gratuit, IGN gratuit).
- ❌ **Annotation manuelle** d'images Piste B : on évalue uniquement contre **vérité terrain Wikipedia/Jane's** (composition publique de la flotte), pas d'annotation pixel.
- ❌ **Taxonomie « MiseEnJambe » 8 classes** : ne pas mélanger avec la Généralisation à 13 classes — on travaille **uniquement sur les 13 classes Généralisation** pour tout résultat noté.

---

## 4. Référentiel modèles — choix d'algorithmes

### 4.1 Détection (Piste A : P2)

| Tâche | Algo retenu | Justification |
|---|---|---|
| **Détecteur principal** | **YOLOv8n** (Ultralytics) | Plus rapide à fine-tuner ; mAP convenable sur petits objets satellite ; ONNX export natif ; doc + tutos abondants. |
| **Benchmark Q16 #2** | **Faster R-CNN** (torchvision, ResNet50-FPN pré-entraîné COCO) | Plus précis sur petits objets, plus lent → contraste pédagogique. |
| **Benchmark Q16 #3** | **RT-DETR-l** (Ultralytics) ou **YOLO11n** | Détecteur transformer-based, état de l'art 2024. |
| **Dataset d'entraînement** | **xView3-SAR** (Sentinel-1 SAR, navires + DARK/AIS labels) ⭐ **OU** **Airbus Ship Detection** Kaggle (optique) | xView3 = baseline officielle dans `BDD-MinArm/xview3-reference`. Airbus = optique, plus simple, 40 Go. **Choix selon temps de téléchargement J1.** |

### 4.2 Classification militaire/civil (Piste A : P3 Q7)

| Tâche | Algo retenu | Justification cours |
|---|---|---|
| **Binaire militaire/civil** | **Random Forest** + **SVM (kernel RBF)** + **MLP (sklearn)** | Features tabulaires bbox (taille, ratio, position, zone_risk_level). RF baseline robuste, SVM en frontière non-linéaire, MLP capture interactions. **Validation croisée 5-fold**. |
| **Multi-classes (Q8)** | Embeddings **ResNet50** pre-pool (2048-d) + tête linéaire / Random Forest | Sur **HRSC2016** ou ShipRSImageNet (classification de types). Pas de fine-tune ResNet50 entier (temps + GPU). |
| **Métrique business** | **Coût asymétrique** [Cours 4 §8] : `coût(FN militaire)` ≫ `coût(FP militaire)` (rater un militaire est plus grave qu'alerter à tort sur un cargo) → optimiser le **rappel** sur la classe militaire. |
| **Réduction de dim. / viz erreurs (Q9)** | **t-SNE / UMAP** [Cours 3 §3] | Visualiser les paires confondues dans l'espace des embeddings ResNet50. |

### 4.3 Détection « out of the box » (Piste B : la chasse)

| Source | Détecteur |
|---|---|
| **Sentinel-1 SAR** | YOLOv8 fine-tuné xView3 (poids dispo dans `BDD-MinArm/xview3-reference`) **OU** détecteur naïf de seuillage (navire = pixel brillant > µ+kσ sur fond noir) en fallback |
| **Sentinel-2 / Maxar optique** | YOLOv8 fine-tuné DOTA (`yolov8n-obb` pour bbox orientées) |
| **IGN BD ORTHO 20 cm** | YOLOv8 fine-tuné DOTA, taille d'image découpée en tuiles 1024×1024 |

---

## 5. Architecture LLM (Mistral, aligné Sujet 3)

- **Backend** : Mistral (`mistralai` SDK), modèles `mistral-small-latest` (par défaut, économique) ou `mistral-large-latest` (rédaction soignée de la fiche).
- **Pourquoi Mistral** : (i) cohérence narrative avec le Sujet 3 → un seul backend LLM pour les deux livrables, un seul `.env`, un seul `llm.py` ; (ii) souveraineté numérique française → cohérent avec un livrable Minarm.
- **Variable d'env** : `MISTRAL_API_KEY` (dans `.env` non committé). Backend configurable via `LLM_BACKEND=mistral|anthropic|ollama|template`.
- **Usage** : génération de la **fiche de renseignement** par détection militaire en zone sensible (port, classe estimée, conditions météo, événements géopolitiques contextuels, présence/absence AIS).
- **Garde-fous** : pas d'appel LLM en boucle non bornée ; cache des réponses dans `data/cache.sqlite` (clé = hash du prompt) ; `temperature ≤ 0.3` ; fallback hiérarchique Mistral → Anthropic → Ollama → template Jinja2.

---

## 6. Données

### 6.1 Fournies (base notée)

| Fichier | Lignes | Réalité |
|---|---|---|
| `images_metadata_large.csv` | 100 | Métadonnées synthétiques. ⚠️ Couple `(source, resolution)` incohérent (Sentinel-2 jamais 1 m dans la réalité). `url = https://example.com/...` → **décor**. |
| `annotations_large.json` | 100 images × 256 annotations × **13 catégories** | Format COCO. **bbox en normalisé [0,1]**, `area` en px (incohérent → reconvertir). Champ `confidence` ⇒ ce ne sont **pas** de vraies annotations manuelles. |
| `detection_results.csv` | 256 | ≈ recopie de `annotations_large.json` + colonnes zone. **Fusion partiellement triviale.** `is_military` ∈ {True (134), False (122)}. |
| `military_zones.csv` | 20 | `coordinates` = point unique (« lat,lon »), pas de polygone. **`zone_id = MIL-000…` ≠ `zone_id = ZONE-002…` de `images_metadata`** → ne **pas** joindre par id, joindre par distance. |
| `MiseEnJambe/*` | ~20 + 43 annot | **Taxonomie différente (8 classes)** — ne pas mélanger. |

### 6.2 Substitution pour entraînement (Piste A : P2/P3 multi-classes)

| Dataset | Type | Taille | Usage |
|---|---|---|---|
| **xView3-SAR** ⭐ | SAR Sentinel-1, navires + labels DARK/AIS | ~100 Go (réduit possible) | Entraînement YOLOv8 SAR, Q4-Q6 — *dérivé maritime de xView (cité dans le README du Sujet 5)*, baseline officielle dans `BDD-MinArm/xview3-reference` |
| **xView** (original) | Optique aérienne, 60 classes dont navires | ~30 Go | Cité par le README du Sujet 5. Backup si xView3-SAR indisponible. |
| **Airbus Ship Detection** | Optique, segmentation navires | 40 Go | Alternative plus légère pour Q4-Q6 |
| **HRSC2016** | Optique, classification de **types** de navires (frégate, destroyer, …) | ~3 Go | **Critique pour Q8** (multi-classes) |
| **ShipRSImageNet** | Optique, 17 types, 17k images | ~10 Go | Alternative HRSC pour Q8/Q9 |
| **HuggingFace `DefendIntelligence/vessel-detection-labeled-patches`** | Patches labellisés navires | léger | **Cité par le README du Sujet 5**, intégration via `datasets` HF — alternative quick-start à HRSC2016 |
| **Ships in Satellite Imagery** (Kaggle, *rhammell*) | Optique, binaire ship/no-ship | 250 Mo | Cité par le README du Sujet 5. Quick sanity check, démo |

### 6.3 Imagerie réelle (Piste B : la chasse)

| Source | Accès | Clé | Usage |
|---|---|---|---|
| **Copernicus Data Space Ecosystem** (Sentinel-1 SAR + Sentinel-2 optique) | API + STAC | `COPERNICUS_USER` / `COPERNICUS_PASSWORD` (gratuit) | Scènes des 5 bases ; SAR = tout temps/nuit |
| **Sentinel Hub** | API config | `SENTINELHUB_INSTANCE_ID` | Cité par le README du Sujet 5. Alternative quick-look (essai gratuit limité). |
| **Microsoft Planetary Computer** (`pystac-client` + `odc-stac`) | STAC public | — | Sentinel-1/2 **sans inscription** → choix par défaut pour la démo (J1) |
| **Google Earth Engine** | Python API | inscription gratuite (compte Google) | Cité par le README du Sujet 5. Alternative en cas de quota PC ; accès Sentinel + Landsat + datasets dérivés. *Choix par défaut = Planetary Computer car pas d'inscription, mais GEE reste l'option historique citée par le sujet.* |
| **Maxar Open Data** | HTTP | — | Crises ouvertes, 0,5 m. Cité par le README du Sujet 5. |
| **IGN BD ORTHO** (FR) | Géoplateforme WMS | — | Bases FR (Toulon, Brest, Cherbourg) à 20 cm |

### 6.4 Vérité terrain & enrichissement (gratuit)

| Source | Usage |
|---|---|
| **OSM Overpass** (`military=naval_base`, `landuse=harbour`) | Zones militaires **réelles**, polygones |
| **NGA World Port Index** | Ports + caractéristiques |
| **Wikipedia (composition de flotte par base)** | Vérité terrain « combien de frégates à Sébastopol au mois X » |
| **Danish Maritime Authority** (`web.ais.dk/aisdata`) | CSV AIS gratuits — croisement satellite × AIS, navires sombres |
| **NOAA MarineCadastre** | AIS US historique |
| **Global Fishing Watch API** | Gap events / encounters pré-calculés mondial |
| **OpenWeatherMap History** | Météo à la date d'une scène (Q15) |
| **Nominatim** (`geopy`) | Port le plus proche d'un point (Q15) |
| **MCP `shom-wrecks`** | Anti-faux-positifs (épaves SHOM) |

**Cache local** : `data/cache.sqlite` (tables : `osint_osm`, `osint_wiki`, `ais_real`, `weather`, `llm_responses`, `alerts`).
**Sécurité** : toutes les clés via variables d'env, listées dans `README.md`, **jamais committées** ; `.env` dans `.gitignore`.

---

## 7. Métriques d'acceptation (les chiffres du pitch)

| Métrique | Cible | Comment on la mesure |
|---|---|---|
| **mAP@0.5** détecteur Piste A | **≥ 0.60** sur le test set substitué | Ultralytics `val()` |
| **mAP@[.5:.95]** | rapporté | idem |
| **Precision/Recall binaire militaire** (Q7) | **rappel ≥ 0.85** (coût asymétrique [Cours 4 §8]) | 5-fold CV |
| **Accuracy multi-classes** (Q8) sur HRSC2016 | **≥ 0.60** top-1, **≥ 0.85** top-3 | test split |
| **Nb total de navires militaires identifiés** Piste B | **≥ 80** sur 5 bases | tableau de chasse manuel |
| **Nb navires « sombres »** (sat = oui, AIS = non) | **≥ 3** confirmés croisés | analyse Levier L2 |
| **Temps pipeline** (Q14) | **≤ 5 s** par scène 1024×1024 sur CPU après ONNX | benchmark |
| **Couverture des 18 questions** | **18/18** traitées (avec « hors périmètre justifié » accepté) | checklist |

**Cadre business [Cours 4 §8]** : à coût asymétrique, on **calibre vers le rappel** sur la classe militaire — rater un militaire = bien plus grave qu'alerter à tort.

---

## 8. Automatisation & API

- **Pipeline reproductible** : `python reponses_generalisation_detection_navires.py` (ou `make all`) exécute Q1→Q18 de bout en bout, lit `data/raw/`, écrit `data/processed/` + `outputs/`, **sans intervention**.
- **API externes intégrées** : Mistral (LLM), Copernicus Data Space (imagerie), Overpass OSM (zones), Nominatim (geocoding), OpenWeatherMap (météo), Danish Maritime Authority (AIS gratuit), GFW (`gfw-api-python-client`), MCP `shom-wrecks` (anti-FP).
- **Alerte automatique** : **PDF** par détection militaire en zone sensible (`reportlab`) avec fiche LLM ; option **e-mail** (`smtplib`).
- **ONNX** : `model.export(format='onnx')` + `onnxruntime` ; quantisation INT8 ; benchmark CPU.
- **Tests** : `pytest` sur (i) parsing COCO bbox normalisé→px ; (ii) jointure géo par distance ; (iii) classification binaire ; (iv) pipeline end-to-end sur 1 image.

---

## 9. Architecture du projet (arborescence cible)

```
sujet5/
├── data/
│   ├── raw/                      ← CSV/JSON fournis (copie de SujetsHackathon2026/Sujet5/Généralisation/*)
│   ├── processed/                ← résultats intermédiaires (tableau de chasse, dataset_card.csv…)
│   ├── images_real/              ← scènes Sentinel-1/Maxar téléchargées (gitignored)
│   ├── images_train/             ← dataset substitué (xView3 ou Airbus Ship — gitignored)
│   └── cache.sqlite              ← cache OSINT + journal d'alertes
├── src/
│   ├── __init__.py
│   ├── config.py                 ← chargement .env, constantes (rayons, seuils, listes de bases)
│   ├── load.py                   ← lecture + normalisation (Agent A)
│   ├── p1_explore.py             ← Q1-Q3 exploration (Agent A)
│   ├── p2_detection.py           ← YOLO fine-tune, mAP, P/R sur 5 images, optimisation Q4-Q6 (Agent A)
│   ├── p3_classify.py            ← Q7 binaire + Q8 multi-classes + Q9 analyse erreurs (Agent A)
│   ├── p4_geospatial.py          ← Q10-Q12 jointure par distance + temporel + anomalies (Agent B)
│   ├── p5_pipeline.py            ← Q13-Q15 pipeline + ONNX + APIs (Agent B)
│   ├── p6_benchmark.py           ← Q16-Q18 benchmarks détecteurs / classifieurs (Agent A)
│   ├── hunt.py                   ← Piste B : récupération scènes Sentinel-1/2, détection out-of-box, tableau de chasse (Agent B)
│   ├── ais_cross.py              ← croisement détection × AIS (Danish Maritime / GFW) → navires sombres (Agent B)
│   ├── osint_enrich.py           ← Overpass / Wiki / Nominatim / Weather / SHOM wrecks (Agent B)
│   ├── intel_report.py           ← fiche LLM Mistral + PDF (Agent B)
│   ├── maps.py                   ← helpers folium (couches : detections / zones / sombres / alertes)
│   └── llm.py                    ← backend LLM configurable (Mistral / Anthropic / Ollama / template)
├── notebooks/                    ← exploration (un par agent)
├── outputs/                      ← carte HTML, PNG, PDF d'alerte, tableau métriques, tableau de chasse
├── docs/                         ← cadrage.md, hypotheses.md, plan-3-jours.md
├── reponses_mise_en_jambe.py     ← LIVRABLE : Mise en jambe (Agent A)
├── reponses_generalisation_detection_navires.py  ← LIVRABLE : Généralisation Q1→Q18
├── rapport_generalisation_detection_navires.md   ← LIVRABLE
├── README.md
├── Makefile                      ← `make install` / `make all` / `make hunt` / `make report`
├── requirements.txt
└── .env.example                  ← MISTRAL_API_KEY=... COPERNICUS_USER=... COPERNICUS_PASSWORD=... SENTINELHUB_INSTANCE_ID=... OPENWEATHER_API_KEY=... GFW_API_TOKEN=...
```

Env Python : **3.12 via `uv`** (`uv venv` + `uv pip install -r requirements.txt`) — le système 3.9 est trop vieux.

---

## 10. Plan d'exécution & livrables

→ voir [`plan-3-jours.md`](plan-3-jours.md) (3 phases, mode 2 agents IA).

**Livrables finaux** déposés dans `ProjetsEtudiantsHackathon2026/BDD-MinArm-Sujet5/` : `EQUIPE.md`, `reponses_mise_en_jambe.py`, `reponses_generalisation_detection_navires.py`, `rapport_generalisation_detection_navires.md`, `README.md`, `requirements.txt`, `outputs/` (carte HTML, tableau de chasse PNG, PDF d'alerte, métriques CSV).

---

## 11. Risques & parades

| Risque | Parade | Minimum viable garanti |
|---|---|---|
| Pas d'images dans le repo (bloquant P2/P3) | Piste A : dataset substitué documenté (xView3 ou Airbus Ship) ; télécharger J1 | Si rien ne s'entraîne : YOLO pré-entraîné DOTA out-of-the-box sur les scènes Piste B |
| Quotas/clés Copernicus tardives | Comptes ouverts J1 matin, scènes téléchargées J1 soir | Piste B sur 1-2 bases avec Sentinel-2 PC sans inscription via Planetary Computer |
| Pas de GPU local | Colab T4 / Kaggle P100 gratuits | Inférence pré-entraîné (CPU OK) |
| Mistral KO / quota | Fallback hiérarchique : Anthropic → Ollama → template Jinja2 | La fiche se génère via template |
| Démo live qui plante | HTML statique + screenshots dans `outputs/` ; vidéo de secours | screenshots |
| Faux positifs (épaves, plateformes) | MCP `shom-wrecks` + OSM `landuse=industrial` + filtrage taille minimale | les détections incluent un flag `likely_fp` |
| 2 agents IA → erreur silencieuse | Tests pytest sur les fonctions critiques ; revue croisée fin de phase | — |

---

## 12. Pitch (5 min)

Problème (30 s) → Démarche & périmètre **2 pistes** assumées (1 min) → **Démo live** (2 min : mappemonde folium → zoom Toulon → scène Sentinel-1 + bboxes + types → tableau de chasse mis à jour → **moment « navire sombre »** : on clique sur un point, AIS muet, fiche Mistral générée + PDF d'alerte) → Résultats chiffrés (1 min : mAP X, accuracy multi-classes Y, **N navires militaires sur Z bases**, ≥3 navires sombres confirmés, temps ONNX) → Limites & extensions (30 s : résolution sub-métrique uniquement classifiable, croisement avec Sujet 3 « signature radio », fine-tune SAR plus profond).

**Slide commune** « navire sombre » à insérer dans les pitchs S3 et S5 → pont narratif explicite : « rendre visible un navire qui veut rester invisible — par ses émissions radio (S3) et depuis l'orbite (S5) ».

---

## 13. Validation du cahier des charges

✅ Couvre les 18 questions de la Généralisation (Piste A + Piste B) · ✅ choix d'algorithmes justifiés (cours Bogroff pour la classif tabulaire, baselines pré-entraînées pour la CV) · ✅ pour chaque algo : pourquoi + paramétrage + métrique · ✅ Piste B = vraie chasse opérationnelle = différenciation jury · ✅ enrichissement OSINT réel gratuit (Copernicus, OSM, AIS, Wiki, SHOM) · ✅ pipeline reproductible + alertes auto · ✅ métriques d'acceptation chiffrées + cadre business asymétrique · ✅ risques + parades + minimum viable · ✅ 0 € · ✅ aligné sur les 6 critères officiels et la consigne générale · ✅ **pont narratif explicite avec le Sujet 3 (« navire sombre »)**.
