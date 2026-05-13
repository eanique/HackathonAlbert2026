# Hypothèses, seuils & choix algorithmiques — Sujet 5

> Document **vivant**. Toute hypothèse / tout seuil / tout choix d'algo est noté ici avec sa justification (cours Bogroff pour la partie tabulaire, baselines pré-entraînées / papier de référence pour la CV). Repris tel quel en annexe du `rapport_generalisation_detection_navires.md`. Créé le 2026-05-11.

> **Convention de citation** : `[Cours N §M]` = cours d'Alexis Bogroff, Albert School (`cours ml/FICHE_REVISION_DATA.md`) ; `[arXiv ID]` ou `[Repo X]` = baseline ou paper externe.

---

## A. Conventions de données

| Convention | Détail |
|---|---|
| IDs | `image_id`, `zone_id`, `mmsi` traités comme **chaînes** (jamais en int). |
| Timestamps | Tous convertis en `datetime` UTC, format ISO 8601 (`2026-08-31T17:17:58Z`). |
| Coordonnées | `(lat, lon)` en **degrés décimaux** ; CRS = **EPSG:4326** (WGS84). Parsing strict des chaînes « lat,lon » et « lat,lon;lat,lon ». |
| Bbox COCO | `annotations_large.json` les fournit en **normalisé [0,1]** `[x, y, w, h]`. **Toujours reconvertir en pixels** avec `image.width / image.height` avant tout calcul d'IoU, d'aire, ou d'entrée modèle. **Aire en px =** `w_norm × W × h_norm × H`. |
| Détection « militaire » | `is_military == True` ⟺ `category_id ∈ {2, 3, 4, 5, 9, 10, 11, 12, 13}` = `{Navire de guerre, Frégate, Destroyer, Porte-avions, Sous-marin, Croiseur, Corvette, Navire de soutien, Bâtiment de débarquement}`. Les `Pétrolier (6), Cargo (7), Chalutier (8)` et `Navire civil (1)` sont **civils**. → Hypothèse explicite, à valider sur 5 lignes du CSV. |
| Manquants | Loggés (combien, quelle colonne, traitement). Aucune imputation silencieuse. |
| Scaling | `StandardScaler` avant tout algo basé distance/variance (SVM, MLP, k-NN, PCA, t-SNE, UMAP) [Cours 0 §4]. ⚠️ **Random Forest ne demande PAS de scaling**. |
| Dédup | Lignes strictement identiques → drop. Doublons `(image_id, bbox)` → on garde la plus haute `confidence`. |

---

## B. Seuils & paramètres par question

### Partie 1 — Exploration

| ID | Hypothèse / seuil | Justification |
|---|---|---|
| **Q1** Sources & résolutions | `value_counts(source).head(3)`. Résolution moyenne : parser le string « 1m / 5m / 10m / 0.5m » en float et faire `mean`. % cloud>30 : `(cloud_cover > 30).mean() × 100`. | Énoncé. ⚠️ Commenter l'**incohérence physique** Sentinel-2 réel = 10 m toujours (jamais 1 m) → fichier synthétique, à signaler dans le rapport. |
| **Q2** Annotations | Filtrer `category_id ∈ militaires` (cf. § A). Répartition % via `value_counts(normalize=True) × 100`. Taille bbox **en pixels** : `w_norm × W` et `h_norm × H`, puis `area_px = w × h`, agrégation `groupby(category).mean()`. | Énoncé. ⚠️ Champ `area` du JSON déjà fourni mais **incohérent** avec bbox normalisée → on recalcule. |
| **Q3** Fusion | `merge(images_metadata, detection_results, on='image_id')`. Nb détections/image = `groupby(image_id).size().mean()`. Image max = `idxmax()`. | Énoncé. Commenter que la « fusion » est partiellement triviale (detection_results ≈ recopie de annotations_large + colonnes zone). |

### Partie 2 — Détection (Piste A : YOLOv8 sur dataset substitué)

| ID | Hypothèse / seuil | Justification |
|---|---|---|
| **Q4** Entraînement YOLO | **Substitution documentée** : entraînement sur **xView3-SAR** (Sentinel-1, baseline `BDD-MinArm/xview3-reference`) OU **Airbus Ship Detection** Kaggle (optique, 40 Go, plus rapide). Split **70/15/15** déterministe (`random_state=42`). YOLOv8n, `imgsz=640`, `epochs=50` (Piste A) ou `epochs=20` (mode rapide). Optimiseur AdamW, `lr0=1e-3`. Métriques : **mAP@0.5** et **mAP@[.5:.95]** (Ultralytics `model.val()`). | Énoncé impose mAP. Substitution justifiée : aucune image n'est fournie dans le repo. YOLOv8n choisi pour la rapidité d'itération sur CPU/Colab T4. |
| **Q5** P/R sur 5 images non vues | 5 images du **test split** (jamais vues). Pour chaque : `model.predict(img, conf=0.25, iou=0.45)` ; matcher avec GT par IoU≥0.5 ; calculer **precision = TP/(TP+FP)** et **recall = TP/(TP+FN)**. | Énoncé. Seuils conf=0.25 et IoU=0.45 = défauts Ultralytics. |
| **Q6** Optimisation | Augmentation **albumentations** : `HorizontalFlip(p=0.5)`, `RandomRotate90(p=0.3)`, `RandomBrightnessContrast(p=0.3)`, `HueSaturationValue(p=0.2)`. Hyperparams : `lr0 ∈ {1e-3, 5e-4, 1e-4}`, `batch ∈ {8, 16, 32}`, `imgsz ∈ {512, 640, 768}`. Rapporter **Δ mAP** vs baseline. | Énoncé. Limite : 1 grid search rapide (3-5 configs max, GPU Colab). |

### Partie 3 — Classification

| ID | Hypothèse / seuil | Justification |
|---|---|---|
| **Q7** Binaire militaire/civil | **Features** extraites de `detection_results.csv` + jointure `images_metadata` : `(bbox_w_px, bbox_h_px, bbox_ratio = w/h, area_px, x_center_norm, y_center_norm, distance_to_image_center, image_resolution_m, zone_risk_level_onehot)`. **Classifieurs comparés** : `RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)`, `SVC(kernel='rbf', C=1.0)`, `MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500)`. Validation **5-fold StratifiedKFold**. Target = `is_military`. **Métrique principale** : **rappel** (classe militaire), F1, ROC-AUC. | [Cours 4 §8] Coût asymétrique : `coût(FN militaire) ≫ coût(FP militaire)` → calibrer sur le rappel. RF baseline robuste sans scaling ; SVM RBF en frontière non-linéaire ; MLP capture interactions. |
| **Q8** Multi-classes | Sur **HRSC2016** (ou ShipRSImageNet) pour la **vraie classification de types**. Pipeline : `ResNet50(weights='IMAGENET1K_V2')` → pre-pool (output `(B, 2048)` avant la dernière FC) → tête `LogisticRegression` ou `RandomForestClassifier`. **Pas de fine-tune** du ResNet50 (temps + GPU). Métriques : accuracy top-1, top-3 ; matrice de confusion. | Embeddings ResNet50 = state of the art pré-entraîné ImageNet, transférable. Tête légère = entraînement CPU-friendly. |
| **Q9** Analyse d'erreurs | Matrice de confusion `sklearn.metrics.confusion_matrix`. **t-SNE 2D** des embeddings ResNet50 colorés par `(classe_vraie, classe_prédite)` [Cours 3 §3]. Identifier les **3 paires les plus confondues**. Hypothèses a priori : Frégate ↔ Destroyer (taille proche, ~140 m vs ~160 m), Cargo ↔ Pétrolier (silhouette proche). | t-SNE > UMAP ici car interprétation visuelle d'erreurs (UMAP préserve mieux la structure globale mais t-SNE met mieux en évidence les clusters locaux confondus). |

### Partie 4 — Géospatial & temporel

| ID | Hypothèse / seuil | Justification |
|---|---|---|
| **Q10** Détections en zones militaires | **PIÈGE MAJEUR** : `military_zones.zone_id = MIL-XXX` **≠** `images_metadata.zone_id = ZONE-XXX`. → **NE PAS** faire `merge(on='zone_id')`. Faire une **jointure par distance géodésique** : pour chaque détection, calculer `geopy.distance.geodesic((det.lat, det.lon), (mil.lat, mil.lon))` pour chaque zone militaire ; flag `in_military_zone = True` si **distance < 25 km** ET `mil.active == True`. Rayon 25 km = ordre de grandeur d'une base navale + zone d'approche. À documenter et discuter. | Vérifié dans les CSV : les deux `zone_id` ne se joignent pas. 25 km = compromis (Norfolk ≈ 20 km de rayonnement, Toulon plus serré ≈ 10 km). En réel : remplacer par polygones OSM `military=naval_base`. |
| **Q11** Temporel | Pour chaque `zone_name`, `groupby(date).size()` → série journalière ; **rolling 7 jours** ; pics = points où `count > µ + 2σ`. Corrélation manuelle avec un calendrier court d'exercices/événements (à ajouter au rapport, ~5 lignes max). | Sur 100 images, série courte → on rapporte les pics mais on n'overfit pas. |
| **Q12** Anomalies géospatiales | Anomalie type A : `is_military == True ET in_military_zone == False` (militaire en mer libre — légitime si transit, suspect si proximité d'une zone sensible étrangère). Anomalie type B : `is_military == False ET in_military_zone == True ET zone.risk_level ∈ {High, Critical}` (civil en zone militaire critique). Carte folium avec couches rouge pour ces deux types. | Énoncé. Distinction risk_level = fait baisser les FP triviaux (civil dans port militaire à faible risque = trafic normal). |

### Partie 5 — Pipeline & API

| ID | Hypothèse / seuil | Justification |
|---|---|---|
| **Q13** Pipeline | `pipeline.predict(image_path) → list[Detection]` où `Detection = {bbox_px, category, confidence, is_military, in_military_zone, alert: bool}`. **Alerte = True** ssi `is_military ET in_military_zone ET zone.risk_level ∈ {High, Critical} ET zone.active`. Rapport markdown + PDF par image. Testé sur **10 images** du test split + idéalement 1-2 scènes réelles Piste B. | Énoncé. Logique alerte conservatrice (3 conditions ET) = limite FP en démo. |
| **Q14** Optim | Export `model.export(format='onnx', dynamic=True, simplify=True)`. Benchmark : 10 inférences sur CPU, PyTorch vs ONNX vs ONNX-INT8 (`onnxruntime.quantization.quantize_dynamic`). Rapporter **ms/inférence** + ratio. | Gain attendu ×3 à ×5 sur CPU avec ONNX-INT8. |
| **Q15** API externes | (i) `geopy.geocoders.Nominatim` (`user_agent='bdd-minarm-s5'`, rate-limit 1 req/s) → port le plus proche. (ii) `OpenWeatherMap History API` → `cloud_cover, visibility, wind_speed` à la `(lat, lon, date)`. (iii) Overpass OSM → polygone `military=naval_base` autour de la détection. | Toutes gratuites. Nominatim impose 1 req/s → cache SQLite obligatoire. OpenWeatherMap historique = clé gratuite limitée. |

### Partie 6 — Benchmark

| ID | Hypothèse / seuil | Justification |
|---|---|---|
| **Q16** Détecteurs | YOLOv8n vs Faster R-CNN ResNet50-FPN (torchvision) vs RT-DETR-l (Ultralytics). Tableau : `mAP@0.5, mAP@[.5:.95], FPS CPU, FPS GPU si dispo, taille modèle MB`. « Meilleur compromis » = règle de Pareto sur (mAP, FPS). | 3 archis très différentes (single-stage / two-stage / DETR). |
| **Q17** Classifieurs | RF vs SVM-RBF vs ResNet50 features + LogReg. Tableau : accuracy, F1 macro, **temps train**, **temps inférence**, interprétabilité. | Énoncé. ResNet50 = baseline deep, RF = baseline interprétable. |
| **Q18** Rapport global | Sections : Contexte → Données → Démarche (Piste A + Piste B) → Résultats (tableaux mAP + tableau de chasse Piste B) → Automatisation → Limites → Pistes. Tous les chiffres traçables au code. | Checklist du skill `livrable-hackathon`. |

---

## C. Leviers de valeur ajoutée

| Levier | Hypothèse / seuil | Justification |
|---|---|---|
| **L1 La vraie chasse (Piste B)** | Bases ciblées : **Toulon (FR), Brest (FR), Severomorsk (RU), Sébastopol (RU), Norfolk (US)**. Pour chaque : 2-3 scènes Sentinel-1 (SAR) sur les **30 derniers jours**. Détecteur out-of-the-box : YOLOv8 fine-tuné xView3 (poids `BDD-MinArm/xview3-reference`). Vérité terrain : Wikipedia (composition publique de la flotte). Cible : **≥ 80 navires militaires détectés** sur l'ensemble. | Sentinel-1 SAR = tout temps/nuit, gratuit, idéal pour la chasse. xView3 pré-entraîné = baseline officielle. |
| **L2 Navire sombre** | Sur chaque détection de la Piste B : requête AIS au `(timestamp ± 30 min, bbox ± 5 km)` via Danish Maritime Authority (Baltique gratuit) ou Planetary Computer datasets `ais-global` ou GFW gap events. Détection « sombre » = aucun MMSI dans la fenêtre. Cible : **≥ 3 navires sombres confirmés** dans le rapport. | Pont narratif explicite avec le Sujet 3. Croisement = différenciation jury. |
| **L3 Fiche LLM** | Mistral `mistral-large-latest`, `temperature=0.2`, prompt structuré : `{port, classe estimée, confidence, conditions météo, événements géopolitiques contextuels (3 lignes manuelles), présence/absence AIS}`. Sortie : markdown + PDF via `reportlab`. Cache SQLite. | Souveraineté FR ; aligné Sujet 3 ; coche critère « LLM ». |
| **L4 Anti-faux-positifs** | (i) MCP `shom-wrecks` : si une détection est à < 200 m d'une épave SHOM connue → flag `likely_fp_wreck`. (ii) OSM `landuse ∈ {industrial, harbour, port}` ou `man_made=pier` → flag `likely_fp_port_structure`. (iii) Bbox `area_px < 100 px` ET résolution > 5 m → flag `too_small`. | Réduit FP triviaux. SHOM uniquement utile dans les eaux françaises. |
| **L5 Graphe** | `networkx.DiGraph` : nœuds = `{Navire, Scène, Zone, Événement}` ; arêtes = `{detected_in, located_at, occurred_during}`. Viz `pyvis` HTML interactif. | Coche critère « graphes de connaissance ». |

---

## D. Hypothèses spécifiques à la Piste B (chasse réelle)

| Hypothèse | Valeur | Justification |
|---|---|---|
| Fenêtre temporelle | **30 derniers jours** | Compromis fraîcheur × disponibilité Sentinel-1 (revisite ~6 jours). |
| Tuile d'inférence | **1024×1024 px** avec **overlap 128 px** | Limite mémoire CPU/Colab T4 ; overlap pour ne pas couper des navires en bord de tuile. |
| Seuil de confiance YOLO | **0.30** sur SAR (recall-orienté), **0.45** sur optique | SAR : haut bruit, on tolère plus de FP qu'on filtrera ensuite. Optique : signal net. |
| NMS IoU | **0.45** (défaut) | Standard. |
| Distance max détection ↔ zone navale | **5 km** | Pour comptabiliser une détection au profit d'une base spécifique. |
| Taille minimale d'un navire militaire sur scène 10 m | **15 px = 150 m** | Filtre les bouées, mais conserve corvettes (~90 m) à la limite — à signaler dans le rapport. |
| Vérité terrain croisée | Wikipedia (composition flotte du port) + Jane's open snippets | Pas d'annotation manuelle. |

---

## E. Hypothèses sur le récit & pitch

| Hypothèse | Justification |
|---|---|
| Le jury Minarm valorise **la chasse réelle ≫ le benchmark Kaggle** | Énoncé explicite : « l'équipe gagnante sera celle qui parviendra à identifier le plus grand nombre de navires militaires ». |
| Le récit « navire sombre » est le moment **émotionnel** du pitch | Pont S3↔S5 ; demande peu de travail si Piste B + croisement AIS fonctionnent. |
| **Restreindre intelligemment** le périmètre est explicitement noté | Consigne hackathon ; on assume P6 light + pas de fine-tune from scratch + classification multi-classes sur dataset substitué. |
| **Mistral** plutôt que Claude/GPT | Souveraineté FR cohérent avec un livrable Minarm + un seul backend LLM partagé avec S3. |

---

## F. Journal des décisions

| Date | Décision | Raison |
|---|---|---|
| 2026-05-11 | Sujet 5 retenu (équipe verrouillée S3 + S5) | Décision équipe BDD-MinArm. |
| 2026-05-11 | Stratégie 2 pistes (A : répondre aux questions sur dataset substitué ; B : vraie chasse) | Pas d'images dans le repo + énoncé impose une « opération de recherche ». |
| 2026-05-11 | Mistral comme backend LLM | Cohérence avec Sujet 3 + souveraineté FR. |
| 2026-05-11 | Jointure zones militaires par **distance géodésique 25 km**, pas par `zone_id` | Les `zone_id` des deux fichiers ne se correspondent pas. |
| 2026-05-11 | Reconvertir bbox COCO normalisée → pixels avant tout calcul | `area` du JSON est incohérent ; standard CV = pixels. |
| 2026-05-11 | Taxonomie militaire = `{2, 3, 4, 5, 9, 10, 11, 12, 13}` (9 catégories sur 13) | Hypothèse explicite ; vérifiable sur 5 lignes du CSV. |
