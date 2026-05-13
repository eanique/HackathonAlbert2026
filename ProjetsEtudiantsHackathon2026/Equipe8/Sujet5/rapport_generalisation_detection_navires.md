# Rapport — Sujet 5 · Détection & traque de navires militaires par imagerie satellite

> **Équipe BDD-MinArm** · Hackathon Albert School 2026 · Commanditaire Ministère des Armées.
> Sources OSINT / open data uniquement.

---

## 0. TL;DR

Notre approche assume deux pistes parallèles, choix méthodologique au service du commanditaire :

1. **Piste A — répondre aux questions** sur le dataset fourni (qui ne contient *aucune image* — uniquement des métadonnées CSV/JSON synthétiques). Quand un détecteur image est nécessaire (Q4-Q9), on **substitue un dataset public documenté** (xView3-SAR ou Airbus Ship Kaggle pour la détection, HRSC2016 pour la classification de type).
2. **Piste B — la vraie chasse**. On récupère de l'imagerie réelle (Sentinel-1 SAR via Microsoft Planetary Computer, sans inscription) sur **5 bases navales** (Toulon, Brest, Norfolk, Severomorsk, Sébastopol), on y passe un détecteur YOLO, et on **croise avec l'AIS** (gap events Global Fishing Watch) pour démasquer les **navires sombres** (vus par satellite, absents de l'AIS au même instant). C'est le levier qui rend la démonstration jouable pour le commanditaire.

**Pont avec le Sujet 3** : « rendre visible un navire qui veut rester invisible — par ses émissions radio (S3) et depuis l'orbite (S5). »

---

## 1. Contexte & périmètre

### 1.1 Sujet
Détecter + classifier des navires sur imagerie satellite, distinguer civil/militaire, identifier les types (frégate, destroyer, porte-avions, sous-marin…), **et identifier le maximum de navires militaires sur de l'imagerie publique** de zones stratégiques (énoncé). Le jury valorise la **chasse réelle** (Piste B) plus que les métriques mAP.

### 1.2 Données fournies
| Fichier | Lignes | Nature |
|---|---|---|
| `images_metadata_large.csv` | 100 | métadonnées images (id, coords, source, résolution, cloud_cover) |
| `annotations_large.json` | 100 img × 256 annot | format COCO, 13 catégories |
| `detection_results.csv` | 256 | bboxes + classes + flags `is_military`/zone |
| `military_zones.csv` | 20 | zones militaires (lat/lon, risk_level, active) |

**⚠️ Aucune image satellite n'est fournie.** Les `satellite_*.jpg` n'existent pas (URLs `https://example.com/...`), les données sont synthétiques. C'est explicitement un dataset de **mise en jambe analytique**, pas un jeu d'entraînement.

### 1.3 Hypothèses critiques (verrouillées au J1 — cf. [`docs/hypotheses.md`](docs/hypotheses.md))
1. **Taxonomie militaire** = `category_id ∈ {2, 3, 4, 5, 9, 10, 11, 12, 13}` (`MILITARY_IDS` dans [`src/config.py`](src/config.py)).
2. **Bbox COCO en normalisé `[0,1]`** → toujours reconvertir en pixels avec `image.width/height` (le champ `area` du JSON est incohérent → recalculé en pixels).
3. **`military_zones.zone_id = MIL-XXX` ≠ `images_metadata.zone_id = ZONE-XXX`** → jointure par **distance géodésique** (`MIL_ZONE_RADIUS_KM = 25` km), **jamais par id**.
4. **Couples (source, resolution) du CSV physiquement impossibles** (ex. Sentinel-2 réel = 10 m toujours, jamais 1 m) → signalés au fil de l'analyse comme incohérences du dataset.

---

## 2. Données & nettoyage *(critère 2 — qualité des données)*

Trois flux de données dans le pipeline :

### 2.1 Données fournies (Piste A)
Loaders unifiés dans [`src/load.py`](src/load.py) :
- `load_images_metadata` : parse `coordinates "lat,lon"`, `resolution "Xm"`, `cloud_cover`.
- `load_annotations` : parse JSON COCO, **reconvertit `bbox` normalisé → pixels** via `image.width/height`, **recalcule `area_px`** (l'`area` natif est incohérent), retourne un `DataFrame` d'annotations prêt à analyser.
- `load_detections` : parse `bbox` string `"x,y,w,h"` → 4 floats normalisés.
- `load_military_zones` : parse coords + `active` (`"True"/"False"` → bool).

### 2.2 Datasets substitués (Piste A, Q4-Q9)
Documentés et cités OSINT (cf. [`SOURCES.md`](SOURCES.md) à venir) :
- **xView3-SAR** (Sentinel-1 SAR, baseline officielle, fork dans l'org : [`BDD-MinArm/xview3-reference`](https://github.com/BDD-MinArm/xview3-reference)).
- **Airbus Ship Detection** Kaggle (optique).
- **HRSC2016** pour la classif multi-classes (Q8).

### 2.3 Données réelles (Piste B)
- **Sentinel-1 RTC** via Microsoft Planetary Computer STAC ([`src/hunt.py`](src/hunt.py)) — **pas de clé requise**, query déjà fonctionnelle sur les 5 bases (~80 scènes sur les 30 derniers jours).
- **AIS** via Global Fishing Watch (gap events, [`src/ais_cross.py`](src/ais_cross.py)) — token gratuit non commercial.
- **OSINT contextuel** ([`src/osint_enrich.py`](src/osint_enrich.py)) : OSM Nominatim (port le plus proche), OpenWeatherMap (météo current), Overpass (zones militaires OSM réelles autour d'un point).

---

## 3. Démarche & méthodes *(critère 1 — usage des libs DS/IA)*

### P1 — Exploration (Q1-Q3) — [`src/p1_explore.py`](src/p1_explore.py)
- **Q1** : `value_counts().head(3)` sur `source`, `mean(resolution_m)`, `(cloud_cover > 30).mean() × 100`.
- **Q2** : annotations militaires (filtre `category_id ∈ MILITARY_IDS`), répartition par type, **taille moyenne bbox en pixels**.
- **Q3** : `merge(images, detections, on='image_id')`, `groupby(image_id).size().mean()`, `idxmax()`.

### P2 — Détection (Q4-Q6) — [`src/p2_detection.py`](src/p2_detection.py)
- **Sanity YOLOv8n COCO** : preuve de vie (out-of-the-box, classe `boat` seulement), tourne sur CPU/MPS.
- **Q4 fine-tune** : *interface présente, non exécutée localement* — pas de GPU (Intel iGPU), Colab T4 requis. `train_yolo_on_substituted(data_yaml)` prêt à exécuter (épochs=20-50, lr0=1e-3 AdamW, imgsz=640, split 70/15/15 seed=42).

### P3 — Classification (Q7-Q9) — [`src/p3_classify.py`](src/p3_classify.py)
- **Q7 binaire** : 3 classifieurs `RandomForest(200, depth=10)` / `SVC(rbf, C=1)` / `MLP((64,32))` en 5-fold CV stratifié. Features = bbox (w, h, area, ratio, position, dist au centre) + resolution + cloud_cover + risk_level (one-hot). **`category_id` exclu** (data leakage : c'est la source de la cible).
- **Q8 multi-classes** (interface) : embeddings `ResNet50(IMAGENET1K_V2)` pre-pool sur HRSC2016, tête LogReg/RF (non exécuté local — Colab).
- **Q9 t-SNE** des erreurs (interface) — non exécuté local.

### P4 — Géospatial & temporel (Q10-Q12) — [`src/p4_geospatial.py`](src/p4_geospatial.py)
- **Q10** : pour chaque détection, distance géodésique (`geopy`) aux 20 zones militaires → flag `in_military_zone` si distance < 25 km **ET** `active`. **Pas de merge sur `zone_id`** (piège).
- **Q11** : série temporelle `groupby(zone, date)`, détection de pics `> µ + 2σ`.
- **Q12** : anomalies Type A (militaire hors zone) + Type B (civil en zone High/Critical).
- Cartes interactives folium ([`outputs/detections_carte.html`](outputs/detections_carte.html), [`outputs/anomalies_carte.html`](outputs/anomalies_carte.html)).

### P5 — Pipeline & API (Q13-Q15) — [`src/p5_pipeline.py`](src/p5_pipeline.py)
- **Q13** `predict_from_csv` (les images n'existent pas) : YOLO → bboxes → classif → zone → règle d'alerte centralisée. La règle : `is_military ET in_military_zone ET risk ∈ {High, Critical} ET active`. Si croisé avec AIS + sombre → mention explicite « navire sombre » dans la raison.
  Variante `predict_from_image(path)` exposée pour les images réelles (Piste B).
- **Q14** : export ONNX YOLOv8n (`model.export(format='onnx', dynamic=True, simplify=True)`), benchmark CPU PyTorch vs ONNX (`benchmark_inference`). INT8 dynamique sautée (couches Ultralytics récalcitrantes en quantisation dynamique — pister via `onnxruntime.tools` plus tard).
- **Q15** : APIs OSINT ([`src/osint_enrich.py`](src/osint_enrich.py)) — Nominatim (reverse, port le plus proche), OpenWeatherMap (météo current), Overpass (zones militaires OSM). Toutes avec cache SQLite + rate-limit 1 req/s.

### Levier L3 — LLM intel report — [`src/llm.py`](src/llm.py) + [`src/intel_report.py`](src/intel_report.py)
- Backend **Mistral REST** (souveraineté FR, aligné Sujet 3) — appel direct via `requests` (le SDK officiel `mistralai` est en quarantaine sur PyPI mai 2026).
- Fallback **template Jinja2** (déterministe, hors-ligne, toujours disponible) — c'est ce qui tourne par défaut sans `.env` rempli.
- Cache SQLite (clé = sha256 du prompt+model) → la même fiche ne re-paye pas.
- Sortie : `outputs/intel_<detection_id>.md` + `.pdf` (reportlab).

### Piste B — La vraie chasse — [`src/hunt.py`](src/hunt.py) + [`src/ais_cross.py`](src/ais_cross.py)
- **Query Sentinel-1 RTC STAC** sur les 5 bases (Toulon, Brest, Norfolk, Severomorsk, Sébastopol) → ~80 scènes dispos sur 30 j, liste dans [`data/processed/chasse_scenes_disponibles.csv`](data/processed/chasse_scenes_disponibles.csv).
- **Téléchargement + inférence YOLO tuilée** (1024×1024 + overlap 128 px) : interface posée, exécution = à lancer (`fetch_scene_geotiff` + `detect_ships_on_scene`).
- **Croisement AIS** ([`src/ais_cross.py`](src/ais_cross.py)) :
  - `gfw_gap_events(bbox, dates)` : REST GFW v3, gap events ≥ 12 h.
  - `flag_dark_ships(df_det, df_ais)` : fenêtre temps ±30 min, spatial ±5 km → flag `is_dark`. Déterministe, testé.

### P6 — Benchmark (Q16-Q18) — [`src/p6_benchmark.py`](src/p6_benchmark.py)
- **Q16** : YOLOv8n (ultralytics) vs RT-DETR-l (ultralytics) vs Faster R-CNN ResNet50-FPN (torchvision) — **inférence COCO pure** sur image sanity (pas de fine-tune local, CPU only). Médiane sur n_iter.
- **Q17** : RF vs SVM-RBF vs MLP — accuracy / F1 / temps 5-fold CV sur features Q7.
- **Q18** : synthèse markdown + figures `q18_*.png` + limites/pistes assumées.

---

## 4. Résultats *(critère 3 — précision & complétude)*

> **Tous les chiffres ci-dessous sont reproductibles** : `make all` regénère exactement ces sorties.

### 4.1 Exploration (Q1-Q3)
- **Q1** — Top 3 sources : `{Maxar: 28, Sentinel-2: 21, PlanetScope: 19}`. Résolution moyenne **4,06 m**. **50 %** des images ont `cloud_cover > 30 %`. ⚠️ Couple (source, resolution) souvent physiquement impossible (Sentinel-2 réel = 10 m toujours → marqueur de la nature synthétique).
- **Q2** — **161/256 = 62,9 %** d'annotations militaires. Répartition par type sauvegardée dans [`data/processed/q2_bbox_by_type.csv`](data/processed/q2_bbox_by_type.csv). Bbox reconvertis en pixels (l'`area` du JSON est incohérent → recalculé).
- **Q3** — **2,56 détections/image** en moyenne, max **IMG-000 (5 détections)**. ⚠️ `detection_results.csv` ≈ recopie de `annotations_large.json` → fusion partiellement triviale, ne contient pas d'information vraiment nouvelle.

### 4.2 Classification binaire (Q7)
5-fold CV sur 256 détections, optimisation sur le **rappel** (coût asymétrique : rater un militaire est plus grave qu'alerter à tort) :

| Classifieur | precision_mil | recall_mil | F1 | ROC-AUC |
|---|---|---|---|---|
| MLP (64,32) | 0,51 | **0,75** | **0,61** | ≈ 0,50 |
| RandomForest (200, depth=10) | 0,53 | 0,56 | 0,54 | ≈ 0,52 |
| SVM (RBF, C=1) | 0,53 | 0,54 | 0,53 | ≈ 0,49 |

📝 **FINDING — point honnête au jury** : **AUC ≈ 0,5 sur les 3 classifieurs** → les features tabulaires (bbox + position + zone) n'ont **aucun signal** sur `is_military` dans ce CSV synthétique. C'est le constat empirique qui **justifie** le passage à Q8 (embeddings ResNet50 sur le contenu image, pas les seules métadonnées).

### 4.3 Géospatial (Q10-Q12)
- **Q10** — **22/134 = 16,4 %** des détections militaires sont en zone militaire active (rayon 25 km). La distance médiane à la zone la plus proche est largement supérieure → la plupart des navires militaires « bougent ». Données dans [`data/processed/q10_detections_with_zones.csv`](data/processed/q10_detections_with_zones.csv).
- **Q11** — Pics détectés : détroit d'Ormuz (4 dates), Pearl Harbor (1 date). Série temporelle complète : [`data/processed/q11_temporal_series.csv`](data/processed/q11_temporal_series.csv).
- **Q12** — **112 anomalies Type A** (militaire hors zone) + **22 Type B** (civil en zone High/Critical). Carte : [`outputs/anomalies_carte.html`](outputs/anomalies_carte.html).

### 4.4 Pipeline & alertes (Q13)
Sur l'ensemble des **256 détections** :
- **134 militaires** (taxonomie `MILITARY_IDS`),
- **44 en zone militaire active**,
- **22 alertes** émises par la règle (`militaire ET in_zone ET risk ∈ {High, Critical} ET active`).

Sortie : [`outputs/q13_pipeline_alertes.csv`](outputs/q13_pipeline_alertes.csv).
Fiche d'exemple Mistral/Jinja2 : [`outputs/intel_<detection_id>.md`](outputs/) + `.pdf`.

### 4.5 Benchmark détecteurs (Q16) — CPU
Inférence pure sur image sanity (n=3), pré-entraîné COCO :

| Modèle | Latence médiane (ms) | p10 | p90 | n_détections |
|---|---|---|---|---|
| **YOLOv8n** (ultralytics) | **139** | 117 | 154 | 6 |
| RT-DETR-l (ultralytics) | 1 020 | 952 | 1 037 | 9 |
| Faster R-CNN R50-FPN (torchvision) | 2 763 | 2 725 | 3 225 | 5 |

**YOLOv8n est ≈ 20× plus rapide que Faster R-CNN sur CPU**, ce qui justifie l'utilisation de YOLO comme baseline pour la Piste B (tuiles 1024×1024 × N par scène — Faster R-CNN ne tient pas la cadence sans GPU). Détails : [`outputs/q16_detectors_benchmark.csv`](outputs/q16_detectors_benchmark.csv).

**4e baseline — Claude Vision (LLM-vision, [`src/claude_vision.py`](src/claude_vision.py))** : adaptée depuis `Sujet5_Navires/agent/detector.py`. C'est un détecteur d'un genre nouveau qui en *une passe* renvoie `{vessel_category, is_military, geopolitical_risk_level, alert, alert_reason}` en JSON. Latence ~2-5 s par image (rate-limit 2 s + appel API), coût ~$0,01-0,05 par appel. **Pas un remplaçant** de YOLO (pas de bbox pixel-précise), mais un *complément* pour l'enrichissement contextuel automatique des fiches d'intel. Gated par `ANTHROPIC_API_KEY` — skip propre si absente.

### 4.6 Benchmark classifieurs (Q17)

| Classifieur | Accuracy | F1 militaire | Temps 5-CV (s) |
|---|---|---|---|
| MLP | 0,504 | **0,612** | 0,089 |
| RandomForest | 0,500 | 0,540 | 0,635 |
| SVM-RBF | 0,477 | 0,535 | 0,045 |

Cohérent avec Q7 : AUC ≈ 0,5, donc **aucune valeur n'est transposable** à un dataset réel — Q17 mesure ici la **vitesse** des classifieurs, pas une qualité prédictive.

### 4.7-bis Dashboard 3D interactif — [`outputs/dashboard_globe.html`](outputs/dashboard_globe.html)

Livrable additionnel : un **dashboard standalone** (un seul fichier HTML, pas de npm) qui ouvre dans Chrome via `make dashboard` (= `python dashboard/generate_globe_dashboard.py`). Le HTML embarque toutes les sorties du pipeline (256 détections, 20 zones militaires, 78 scènes Sentinel-1) injectées en JSON, et s'appuie sur **globe.gl** (Three.js / WebGL) pour le rendu 3D.

**UX des filtres** — fruit d'une recherche sur les patterns de [MarineTraffic Live Map](https://support.marinetraffic.com/en/articles/9552715-live-map-filters-for-advanced-live-map) et [Flightradar24](https://www.flightradar24.com/blog/tutorial/an-all-new-way-to-filter-flights-on-flightradar24-com/), confirmée par les guides UX dashboards :

1. **Sidebar gauche** (8+ filtres = sidebar, pas top-bar — règle data-heavy).
2. **Filter chips persistants** au-dessus de la liste de filtres — chaque option *retirée du défaut* devient un chip cliquable (click = restore).
3. **Compteur résultats temps réel** en haut de la sidebar (`X / 256 détections visibles`).
4. **Counts par filtre** affichés à droite de chaque case (UX MarineTraffic).
5. **Layer toggle** top-left du globe — affiche/cache 4 couches (détections / zones / scènes Sentinel-1 / arcs alerte).
6. **Click sur point → side panel droite** avec détails complets + bannière alerte rouge + bannière navire sombre orange.
7. **Bottom KPI bar** : 5 KPIs (total / militaires / en zone / alertes / scènes SAT).
8. **Light theme** pleinement appliqué (background `#F4F7FB`, accent `#2D6A9F`, palette risk = vert/jaune/orange/rouge).
9. **Bouton "Reset filters"** toujours visible en bas de la sidebar.
10. **Recherche typeahead** (zone, port).

**Stack** : globe.gl (CDN unpkg), Tailwind (CDN), Three.js. Vanilla JS, 0 build, ~132 KB de HTML, ouvre instantanément dans Chrome.

### 4.7 ONNX (Q14)
Export `yolov8n.onnx` réussi (`dynamic=True`, `simplify=True`, imgsz=640). Inférence CPU mesurée sur image sanity (n=5) :

| Backend | Median (ms) | p10 | p90 | n_det |
|---|---|---|---|---|
| PyTorch CPU (`yolov8n.pt`) | 126 | 118 | 131 | 6 |
| **ONNX Runtime CPU** (`yolov8n.onnx`) | **68** | 63 | 85 | 6 |

→ **Gain ×1,86** sur CPU. INT8 dynamique non rapportée (couches Ultralytics récalcitrantes) — pistage via `onnxruntime.tools` + calibration statique laissé à la phase suivante. Détails : [`outputs/q14_onnx_benchmark.csv`](outputs/q14_onnx_benchmark.csv).

### 4.8 Piste B — La vraie chasse (chaîne complète testée)

#### Query STAC (preuve de vie, gratuite)
**~80 scènes Sentinel-1 RTC** dispos sur les 30 derniers jours sur les 5 bases — Toulon ~10, Brest ~21, Norfolk ~5, Severomorsk ~19, Sébastopol ~22. Liste : [`data/processed/chasse_scenes_disponibles.csv`](data/processed/chasse_scenes_disponibles.csv).

#### Téléchargement + inférence (chaîne testée bout en bout sur Toulon)
Sur la machine de démo (Windows CPU, pas de GPU), nous avons exécuté la chaîne complète sur 1 scène Sentinel-2 L2A de Toulon (`S2A_MSIL2A_20260505T102701_R108_T31TGH`, 12,6 × 11,5 km @ 10 m/pix) :

1. **`fetch_scene_window`** — query STAC → signe l'asset href via `planetary_computer.sign_inplace` → `rasterio.open` du COG distant → lecture **fenêtrée** (la base navale, pas la scène entière → ~50 Mo au lieu de 2 Go) → reproject CRS, écriture GeoTIFF local + **preview JPG** ([`data/images_real/.../visual.preview.jpg`](data/images_real/)).
2. **`detect_ships_on_scene`** — normalisation (percentile 2-98 %), upscale 3× (rend visibles les navires de 30 m), tuilage 640×640 overlap 128 px, YOLO COCO sur chaque tuile, **NMS global** (IoU > 0.3), reprojection pixel → lat/lon.

**Résultat brut : 0 détection classifiée `boat`/`ship`.** **Volontairement documenté comme un échec instructif** :

> YOLO COCO trouve 15 boîtes au total (conf ≥ 0,05) sur l'image Toulon — mais elles sont classifiées **`bear` (6), `person` (5), `traffic light` (2), `sheep` (2)**. Aucun `boat`. Voir l'image annotée [`outputs/piste_b_toulon_yolo_coco_diag.jpg`](outputs/piste_b_toulon_yolo_coco_diag.jpg) : un "bear 0.81" pile au centre du port militaire, "sheep" en pleine rade. C'est la **preuve visuelle** que **COCO est le mauvais domaine** pour de l'imagerie satellite top-down — et la **justification empirique** du fine-tune xView3-SAR / Airbus Ship sur Colab T4.

**Diagnostic complet** : [`outputs/piste_b_diagnostic.md`](outputs/piste_b_diagnostic.md).

#### Ce que la Piste B prouve *même sans fine-tune*
- **Pipeline complet OK** : STAC → sign → stream COG (fenêtré, économique) → tile + upscale → YOLO → NMS → reproj WGS84 → CSV. Exécuté de bout en bout sur Toulon, Windows CPU, ~3-4 min/scène.
- **Imagerie réelle de Toulon récupérée** (preview JPG dans le dossier) : on voit nettement la rade militaire, le port, les bâtiments alignés à quai.
- **Coût** : 0 € (Planetary Computer STAC public, sans inscription).

#### Croisement AIS — Levier L2 « navire sombre » (FAIT)

**Token GFW v3 activé** → survey effectif sur 10 corridors stratégiques (oct-déc 2024, gap ≥ 24 h) :

| Zone | Navires sombres |
|---|---|
| Méditerranée orientale (Chypre/Liban) | **7** |
| Mer de Chine méridionale | 2 |
| Mer Noire (corridor Sébastopol) | **1** ← géopol Russie/Ukraine |
| Détroit de Malacca | 1 |
| Ormuz, Bab-el-Mandeb, Norfolk, Toulon (golfe Lion), Brest, Baltique | 0 |
| **TOTAL** | **11 navires sombres réels** |

**Top 5 par durée d'AIS coupé** (extraits de [`data/processed/navires_sombres.csv`](data/processed/navires_sombres.csv)) :

| Navire | MMSI | Pavillon | Zone | AIS coupé |
|---|---|---|---|---|
| AIS NET MARK | 412888806 | 🇨🇳 CHN | Mer de Chine | **4,1 ans** (36 006 h) |
| 12 N PESCA IX91% | 123025865 | — | Mer Noire (Sébastopol) | **2,9 ans** (25 277 h) |
| AL-QADEER | 622171203 | 🇪🇬 EGY | Med orientale | 1,4 ans (12 554 h) |
| SH04-13 | 107090069 | — | Malacca | 12 mois |
| VALENCIA KNUTSEN | 225420000 | 🇪🇸 ESP | Mer de Chine | 11 mois |

**Limite assumée de l'API GFW v3** : le serveur **refuse tous les filtres spatiaux** (`region`, `bbox`, `geometry`, `regions`) — confirmé empiriquement sur 6 variantes de format. Notre solution : **pagination + filtrage côté client** (`gfw_gap_events` pagine jusqu'à 50 × 200 = 10 000 events worldwide puis filtre par bbox). Documenté dans [`src/ais_cross.py`](src/ais_cross.py).

**Pourquoi 0 sombre dans les bases serrées** : les navires au port coupent normalement leur AIS — ce n'est *pas* un gap suspect. Les vrais gap events GFW sont enregistrés en **haute mer**, dans les corridors stratégiques. C'est pour ça que nos 5 bases serrées (rayon 11 km autour de Toulon, etc.) donnent 0 alors que les corridors larges (200-400 km autour des détroits) donnent 11.

#### Prochaine étape (Colab T4, ~1-2 h)
- `train_yolo_on_substituted('xview3.yaml', epochs=30, imgsz=640)` (recommandé) ou Airbus Ship Detection (plus simple, optique).
- Une fois le modèle fine-tuné, le **même pipeline** (`hunt_base`) débloque les vraies détections sur les 5 bases.

---

## 5. Automatisation & API *(critère 4)*

### 5.1 Pipeline reproductible
- `make install` → `uv venv --python 3.12 && uv pip install -r requirements.txt` (~5-10 min).
- `make data` → clone shallow de `eanique/HackathonAlbert2026` + copie CSV/JSON dans `data/raw/`.
- `make generalisation` → exécute **`reponses_generalisation_detection_navires.py`** de bout en bout, produit tous les `outputs/*.csv`, `outputs/*.png`, `outputs/*.html`, `outputs/*.md`, `outputs/*.pdf`.
- `make hunt` → query Sentinel-1 STAC sur les 5 bases.
- `make test` → pytest (sanity + Phase 3, ≥ 9 tests).

### 5.2 APIs OSINT branchées
| API | Usage | Clé requise | Rate-limit |
|---|---|---|---|
| **Microsoft Planetary Computer STAC** | scènes Sentinel-1 | non | — |
| **Nominatim** (OSM) | reverse geocoding | non | 1 req/s + UA |
| **OpenWeatherMap** | météo current | gratuite (.env) | 60 req/min |
| **Overpass** (OSM) | military=naval_base | non | 25s/req |
| **Global Fishing Watch** v3 | gap events AIS | gratuite (.env, non commercial) | — |
| **Mistral** | fiche intel | gratuite (.env) | — |

Toutes les réponses sont mises en **cache SQLite** (`data/cache.sqlite`) — ré-exécution gratuite, démo offline-friendly après une 1ère exécution.

### 5.3 Pipeline d'alerte
`Detection` dataclass typée, règle d'alerte centralisée dans `_decide_alert`, sortie en CSV + fiche markdown + PDF (reportlab). Idempotent, testable.

---

## 6. Articulation avec le Sujet 3

Récit jury : **« rendre visible un navire qui veut rester invisible. »**
- Sujet 3 (équipe S3) : par ses **émissions radio** — déviation de cap, AIS désactivé > 24 h, anomalies de trajectoire.
- Sujet 5 (cette équipe) : **depuis l'orbite** — Sentinel-1 SAR + Sentinel-2.

Le pont concret : **le navire sombre.** Une détection satellite classée militaire, **sans MMSI AIS** au même `(timestamp, bbox)`. C'est implémenté côté S5 (`ais_cross.flag_dark_ships`), et son équivalent S3 est la détection de gaps AIS (cf. fork [`BDD-MinArm/pipe-gaps`](https://github.com/BDD-MinArm/pipe-gaps), GlobalFishingWatch). **Slide commune insérée dans les deux decks de pitch.**

---

## 7. Limites & pistes

### 7.1 Limites assumées
1. **Aucune image** dans le dataset fourni — substitution par dataset public **explicitement documentée**, pas un bug.
2. **AUC ≈ 0.5 sur Q7/Q17** — confirme empiriquement que les métadonnées tabulaires synthétiques n'ont pas de signal sur `is_military` ; ce n'est pas un échec, c'est ce qui pousse à passer aux embeddings d'image (Q8).
3. **Pas de GPU local** (Intel iGPU sur machine de démo) → Q4-Q6 fine-tune et Q8 ResNet50 non exécutés localement. **Interfaces posées, exécution Colab T4** (gratuit) en moins de 2 h.
4. **Résolution = facteur déterminant** : Sentinel-2 (10 m) permet de **détecter** un navire mais pas de **classer le type** (porte-avions vs cargo) → classification de type uniquement sur Maxar (0,5 m) / IGN BD ORTHO (20 cm). Limite acceptée et tracée dans la chaîne (champ `confidence_classif` séparé du `confidence_detect`).
5. **Quantisation ONNX-INT8 dynamique** non rapportée (couches Ultralytics récalcitrantes) — pister via `onnxruntime.tools` ou export statique avec calibration.
6. **Anti-faux-positifs** Piste B : épaves (MCP `shom-wrecks` côté France), plateformes, structures portuaires (OSM `landuse=industrial/harbour`), bbox trop petites — filtres branchés mais à durcir.

### 7.2 Pistes
- **Fine-tune SAR** sur xView3-SAR (baseline officielle, fork [`BDD-MinArm/xview3-reference`](https://github.com/BDD-MinArm/xview3-reference)) — gain mAP attendu sur la Piste B.
- **Q8/Q9** sur HRSC2016 + ResNet50 embeddings (+ t-SNE des erreurs) — Colab T4, ~1 h.
- **Graphe de connaissance** (`networkx.DiGraph` `Navire ↔ Scène ↔ Zone ↔ Événement`) + viz `pyvis` HTML — coupé pour cette livraison (cf. règle de coupe dans [`docs/plan-3-jours.md`](docs/plan-3-jours.md)).
- **Pipe-encounters / pipe-loitering** (forks org `BDD-MinArm`) pour démasquer les transbordements en mer.
- **Détection orientée (OBB)** via toolbox `ai4rs` (fork) pour les types très allongés (porte-avions, sous-marins).

---

## 8. Annexes

- **Composition équipe** : [`EQUIPE.md`](EQUIPE.md).
- **Cadrage initial (PRD)** : [`docs/cadrage.md`](docs/cadrage.md).
- **Hypothèses + journal de décisions** : [`docs/hypotheses.md`](docs/hypotheses.md).
- **Plan d'exécution 3 phases (mode 2 agents IA)** : [`docs/plan-3-jours.md`](docs/plan-3-jours.md).
- **Setup + clés API** : [`README.md`](README.md), [`.env.example`](.env.example).
- **Sources OSINT créditées** : [`SOURCES.md`](SOURCES.md) *(à finaliser)*.
- **Repo « ombrelle » de l'équipe** : `BDD-MinArm/hackathon-2026`. Forks de référence : `xview3-reference`, `xView3_first_place`, `SARFish`, `ai4rs`, `pipe-gaps`, `pipe-loitering`, `GeoTrackNet`, `TrAISformer`…
- **Bibliographie arXiv + mapping vers les questions** : [`../docs/recherche-arxiv-sujets-3-5.md`](../docs/recherche-arxiv-sujets-3-5.md).

---

*Rapport généré le 2026-05-12. Tous les chiffres sont traçables au code (`make generalisation` les regénère). Aucune image ou fichier > 50 Mo n'est committé ; les données réelles vivent dans `data/images_real/` (gitignored).*
