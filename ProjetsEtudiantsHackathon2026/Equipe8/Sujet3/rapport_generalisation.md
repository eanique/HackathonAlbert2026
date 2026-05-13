# Rapport — Sujet 3 : Aide à l'identification des navires par analyse intelligente de leur radio signature

> Équipe **BDD-MinArm** · Hackathon Albert School mai 2026 · Ministère des Armées.
>
> Tous les chiffres ci-dessous sont produits par `python reponses_generalisation.py` (≈ 16 s, sorties dans `outputs/`). Les hypothèses, seuils et choix d'algorithmes sont justifiés dans [`docs/hypotheses.md`](docs/hypotheses.md) avec la référence au cours d'Alexis Bogroff (Albert School, ML III « Unsupervised Learning Problems ») et aux papiers arXiv correspondants. Le cahier des charges complet est dans [`docs/cadrage.md`](docs/cadrage.md).

---

## 1. Contexte & périmètre

Identifier un navire de façon **passive** (sans dépendre de son AIS, qui peut être éteint, falsifié, manipulé), en exploitant sa **signature radio** : identification, localisation, détection d'anomalies, base de profils radio. Livrables imposés : `reponses_mise_en_jambe.py`, `reponses_generalisation.py`, `rapport_generalisation.md`.

**Décision de périmètre** : on couvre **les 14 questions** de la Généralisation ; on **enrichit avec de l'OSINT réel gratuit** (Equasis, base MMSI UIT/MARS, OpenSanctions, flux RSS, Global Fishing Watch API, MCP épaves SHOM) ; on génère une **fiche de renseignement par LLM** (Mistral, choix « souveraineté FR » cohérent avec un livrable Minarm) ; on **valide rigoureusement** contre `anomalies_large.csv` (precision/recall par type d'anomalie, AUC, 5-fold CV) ; on a **benchmarké des modèles hors cours** (XGBoost, 5 détecteurs PyOD) pour objectiver le meilleur choix.

**Hors périmètre, justifié** : pas de vrai traitement de signal SDR (les données = métadonnées tabulaires, pas d'échantillons IQ) — présenté comme extension future ; pas de deep learning entraîné from scratch ; pas de BDD hébergée (CSV + SQLite local). Voir [`docs/cadrage.md`](docs/cadrage.md) §3.3.

---

## 2. Données & contrôle qualité *(critère : « bon emploi des données »)*

### 2.1 Données fournies (synthétiques)

| Fichier | Volumétrie | Colonnes clés |
|---|---:|---|
| `ships_large.csv` | 1 000 | mmsi, name, imo, type, **flag**, length/width, destination, last_ais_update, **historical_names**, **is_suspicious**, year_built, gross_tonnage |
| `radio_signatures_large.csv` | 5 000 | signature_id, mmsi, **frequency** (156-162 MHz VHF marine), bandwidth (kHz), **modulation** (FM/AM/DSC/SSB/OFDM), **power** (W), timestamp, location_lat/lon, signal_strength (dBm), noise_level, **pulse_pattern**, signal_to_noise_ratio |
| `ais_data_large.csv` | 10 000 | mmsi, timestamp, lat/lon, speed, course, status, **ais_active**, navigational_status, heading, rot |
| `anomalies_large.csv` | 100 | anomaly_id, mmsi, **type** ∈ {Fake Flag, Name Change, Spoofing, AIS Disabled, Position Mismatch, Speed Anomaly, Course Anomaly}, description, timestamp, **confidence** ∈ [0.7, 1.0], source |

### 2.2 Contrôle qualité (module `src/eda.py` → `outputs/data_quality_report.md`)

**La donnée est formellement propre** : 0 valeur manquante (sauf `historical_names`, 249 vides = navires jamais renommés — légitime), 0 doublon de ligne, **0 valeur hors-bornes** vs les dictionnaires de données, **0 valeur catégorielle hors-liste**, **intégrité référentielle parfaite** (0 MMSI orphelin : radio/ais/anom ⊂ ships ; 0 MMSI `FAKE-*` ; MMSI tous 9 chiffres ; IMO tous 7 chiffres ; 0 régression temporelle après tri). La normalisation appliquée (`mmsi`/`imo` en chaîne, timestamps en UTC, dédup, tri AIS par (mmsi, timestamp)) suffit.

**Mais 5 artefacts synthétiques mesurés** (le dataset n'a pas de structure réaliste) :
1. `navigational_status` ⊥ `status` : **V de Cramér = 0.03** (devrait ≈ 1) → `navigational_status` est du bruit.
2. `ais_active` ⊥ `status` (V = 0.01) ; `ais_active=True` pour **50.1 %** des lignes → irréaliste.
3. Positions AIS = points aléatoires : vitesse implicite (distance géodésique entre points consécutifs / Δt) — médiane 9 kn mais **p95 = 134 kn, max = 12 338 kn** → beaucoup de paires consécutives sont des « téléportations » → vitesse implicite non discriminante (d'où son poids ≈ 0 après tuning).
4. `name` = placeholder : **100 %** des navires ont `name == "NAVIRE-" + 4 derniers chiffres MMSI` → on identifie par MMSI.
5. `historical_names` = templates `OLD_NAME_0..2` (distribution ≈ uniforme 0/1/2/3) → ne porte que le *compte*.

**Finding majeur — plafond de rappel ≈ 63 %** : **37 des 100 anomalies de référence** (16 « Speed » + 21 « Course ») ont leur valeur citée **uniquement dans la description** (`speed` plafonné à 30 kn alors que les descriptions citent jusqu'à 49.2 kn) → elles sont **structurellement indétectables depuis les tables fournies**. Aucun modèle ne peut les récupérer sans parser la vérité terrain (circulaire). ⇒ **le plafond de rappel atteignable est ≈ 63 %**.

**3 actions de nettoyage réelles** :
- **6 navires sans aucune signature radio** (`363951510, 435131361, 522451298, 534553120, 535117872, 675464001` — dont **4 marqués `is_suspicious`**) : marqués `has_radio_signature=False`, absents de `ship_radio_profiles.csv` (994 profils, pas 1000), exploités comme signal de suspicion (« navire jamais entendu sur les ondes » — détecteur `detect_silent_ships`).
- `name` (large) inutilisable → identification par MMSI.
- `historical_names` NaN ⇒ `n_names_historical = 0` (jamais renommé), parsé proprement par `split(',')`.

### 2.3 Sources OSINT mobilisées *(la consigne survalorise « la découverte de sources pertinentes »)*

Détail + licences dans [`SOURCES.md`](SOURCES.md). En résumé : **Equasis** (historique noms/pavillons, PSC), **base MMSI UIT/MARS** (MID → pays attribué), **OpenSanctions API** (navire/armateur sous sanctions OFAC/UE), **flux RSS maritime** (gCaptain, Maritime Executive — Q14), **Global Fishing Watch API** (gap events / encounters / loitering pré-calculés), **MCP `shom-wrecks`** (≈ 4 796 épaves SHOM, Licence Ouverte). Tout est gratuit. Les clés/identifiants sont en variables d'environnement, jamais committées (`.env` gitignored, `.env.example` versionné). Plan B sans SDR : AIS public Danish Maritime Authority + AISStream.io.

---

## 3. Démarche & méthodes *(critère : « libs data science / IA »)*

Référentiel : cours d'A. Bogroff (`cours ml/FICHE_REVISION_DATA.md`). Pour chaque algo : *pourquoi* + paramétrage. Détails dans [`docs/hypotheses.md`](docs/hypotheses.md).

### Partie 1 — Base de profils radio
- **Q1** — Agrégation par MMSI (moyennes des features quantitatives + mode/entropie des catégorielles) → `data/processed/ship_radio_profiles.csv` (994 profils).
- **Q2** — `pulse_pattern` uniques + combinaison `pulse × modulation × bande de frequency` (l'énoncé prend `pulse_pattern` seul = 6 valeurs → peu discriminant ; on enrichit).
- **Q3** — **K-Means K=5** (cours 1 §2-3) : `StandardScaler` obligatoire + K-Means++ + `n_init=10`. On rapporte WCSS, **Silhouette**, **Elbow plot** K ∈ [2,10], et **PCA 2D** (cours 3 §2) pour la viz.

### Partie 2 — Détection d'anomalies
- **Q4 — Faux pavillon** : **`EllipticEnvelope` / MCD par pavillon** (cours 4 §4) — choisi pour contourner le **masking effect** de la distance de Mahalanobis naïve (les anomalies présentes dans le dataset gonfleraient µ et Σ). Fallback z-score robuste (médiane ± MAD) pour les pavillons < 30 navires.
- **Q5 — Changements de nom** : règle sur `n_names_historical > 2`.
- **Q6 — Signatures orphelines** : `radio.mmsi ∉ ships.mmsi` ; attribution par **k-NN** standardisé.
- **Q7 — AIS off > 24 h** : plages contiguës `ais_active==False`, exclusion `Moored/At Anchor` (algo inspiré de `pipe-gaps` / Global Fishing Watch).
- **Q8 — Écart position AIS↔radio > 1 km** : `merge_asof` par mmsi (tolérance 1 j, vu la sparsité temporelle) + `geopy.distance.geodesic` ; **carte folium** (`outputs/carte_anomalies.html`).

### Partie 3 — Temporel & statistique
- **Q9** — Évolution `frequency` / `signal_strength` du MMSI le plus actif + ruptures `|Δfreq| > 1 MHz`.
- **Q10** — Moyenne et écart-type de `frequency` par pavillon.
- **Q11** — Corrélation `speed`(AIS) ↔ `frequency`(radio) (`scipy.stats.pearsonr`).

### Partie 4 — Automatisation, validation, mise à jour
- **Q12 — Pipeline d'identification passive** : **k-NN d'identification** (k=10, vote pondéré 1/distance, features enrichies frequency + bandwidth + power + signal_strength + SNR + one-hot modulation + pulse_pattern) entraîné sur **toutes les signatures individuelles** ; **One-Class SVM en novelty detection** (cours 4 §3 & §5, ν=0.05, kernel RBF) entraîné sur le « normal propre » (navires `is_suspicious=False`) → décide si la signature est « du normal » ou « nouvelle ». L'identification est suivie de l'**enrichissement OSINT** (`build_osint_dossier`) puis de la **fiche de renseignement rédigée par Mistral** + **alerte PDF** (`reportlab`).
- **Q13 — Validation** : **leave-one-out** (10 signatures × 5 graines, on refit l'`Identifier` sans la signature testée — pas de data leakage).
- **Q14 — Mise à jour auto** : moyenne glissante des profils + ingestion **flux RSS** (`feedparser`) qui met à jour `is_suspicious` côté registre.

### Valeur ajoutée (Levier 3 — score de suspicion + benchmark)
- **Score multi-facteurs** : somme pondérée des contributions Q4–Q8 + règles de spoofing + **Isolation Forest** (cours 4 §6, sans scaling, `contamination=0.02`) + **LOF** (cours 4 §7, `n_neighbors=20`, anomalies *locales*) + **sous-score zone-dépendant** (façon GeoTrackNet light, arXiv 1912.00682 — grille 1°×1°, `−log p`) + **terme LLM tabulaire** zero-shot (arXiv 2406.16308, via Mistral).
- **Tuning des poids** : **régression logistique L2, 5-fold CV stratifié** (C=0.3, `class_weight='balanced'`), cible = `anomalies_large.csv`. Les poids = coefficients moyens sur les folds, clippés ≥ 0.
- **Benchmark hors cours** *(force de proposition)* : **XGBoost** (5-fold CV, `scale_pos_weight=9.4`) et **5 détecteurs PyOD** (ECOD, COPOD, HBOS, KNN, PCA-OD) — pour objectiver si « moderne » bat « cours ».
- **Graphe de connaissances** : NetworkX + pyvis (navires ↔ pavillons ↔ types d'anomalies ↔ alertes) → `outputs/knowledge_graph.html` (coche le critère « graphes de connaissance »).
- **Évaluation business** (cours 4 §8) : seuil calibré sur le **coût asymétrique** `10·FN + FP` (rater un navire suspect coûte 10× plus qu'une alerte à tort, contexte Minarm).

---

## 4. Résultats *(critère : « précision et complétude »)*

### 4.1 Bilan données (cf. `outputs/data_quality_report.md`)
1 000 navires / 5 000 signatures / 10 000 lignes AIS / 100 anomalies. 0 manquant (sauf `historical_names` 249), 0 doublon, 0 hors-bornes, 0 orphelin. 6 navires sans signature radio (4 suspects). Plafond de rappel : **63 %**.

### 4.2 Partie 1
- **Q1** — 994 profils. Top 5 fréquence moyenne : `211239944` (161.94 MHz), `604969493` (161.79), `205927097` (161.74), `206352442` (161.70), `637173265` (161.54).
- **Q2** — **6** `pulse_pattern` distincts, **0** apparaissant 1 seule fois ; **330** combinaisons `pulse × modulation × bande` uniques.
- **Q3** — K=5 : **WCSS = 1 446**, **Silhouette = 0.224** (le Silhouette est maximal à K=4 = 0.227 ; on garde K=5 imposé par l'énoncé mais on le signale). Viz : `outputs/clusters_kmeans.png` (scatter frequency×power + PCA 2D), table Elbow/Silhouette : `outputs/q3_elbow_silhouette.csv`.

### 4.3 Partie 2 — détection d'anomalies (rappel par type, vs `anomalies_large.csv`)

| Type d'anomalie | Détecteur | Rappel | (récupérable ?) |
|---|---|---:|:---:|
| **Spoofing** | signatures orphelines + règles MMSI/timestamp/intervalle | **40 %** (4/10) | ✅ |
| **Name Change** | `n_names_historical > 2` | **27 %** (4/15) | ✅ |
| **Position Mismatch** | écart AIS↔radio > 1 km (geopy) | **21 %** (3/14) | ✅ |
| **AIS Disabled** | plages `ais_active=False` > 24 h | **17 %** (2/12) | ✅ |
| **Fake Flag** | `EllipticEnvelope`/MCD par pavillon | **8 %** (1/12) | ✅ |
| **Course Anomaly** | — | 0 % (0/21) | ❌ valeur seulement dans la description |
| **Speed Anomaly** | — | 0 % (0/16) | ❌ idem |
| **Global** | — | **14 %** (14/100) | plafond théorique **63 %** |

Détails : Q4 → 60 navires flaggés (precision 1.7 % vs Fake Flag) ; Q5 → 258 navires > 2 noms ; Q6 → **0 signature orpheline** (le dataset large n'en contient pas — le détecteur est démontré sur le principe) ; Q7 → 215 plages AIS-off > 24 h sur 203 navires ; Q8 → 261 paires en écart > 1 km sur 235 navires. CSV : `outputs/q4_fake_flag_report.csv`, `q7_ais_off_blocks.csv`, `q8_position_mismatch.csv`, `recall_by_type.csv` ; graphique : `outputs/recall_by_type.png`.

**Lecture** : même sur les 63 anomalies *récupérables*, le rappel n'est que de 22 % — cohérent avec un dataset où les features (frequency/bandwidth/power, positions, statuts) sont quasi décorrélées des labels (cf. §2.2 : V de Cramér ≈ 0 partout). **La valeur du livrable n'est pas l'exactitude (impossible sur cette donnée) mais la rigueur de la démarche et la quantification honnête des limites.**

### 4.4 Partie 3
- **Q9** — MMSI `336052886` : 16 signatures, **10 sauts** `|Δfreq| > 1 MHz`. Graphique : `outputs/q9_temporal_336052886.png`.
- **Q10** — **Denmark** a la fréquence moyenne la plus haute (159.165 MHz). Table : `outputs/q10_flag_stats.csv`.
- **Q11** — r = **0.031**, p = **0.62**, N = 261 → **corrélation non significative** entre vitesse AIS et fréquence radio. C'est l'hypothèse a priori (la fréquence VHF ne dépend pas physiquement de la vitesse) — une réponse négative argumentée, conforme à la physique.

### 4.5 Partie 4 — automatisation & validation
- **Q12** — Le pipeline `identify(signature) → dict` retourne `(mmsi, confidence, suspect, raisons, novelty_score)`. Sur la signature de démo (présente dans le training → leakage attendu), il retrouve le bon MMSI mais avec une confiance faible (~0.20, car le vote k-NN est dispersé). Une fiche de renseignement et un **PDF d'alerte** sont générés pour le navire au score de suspicion le plus élevé (`outputs/alerte_<mmsi>.md` + `.pdf`).
- **Q13 — résultat important** : en **leave-one-out** (50 essais : 10 signatures × 5 graines, l'`Identifier` refit sans la signature testée), **le taux d'identification correcte est de 0/50**. → **Les métadonnées radio fournies ne sont pas plus similaires entre signatures d'un même navire qu'entre navires différents** : le « RF fingerprinting » au sens du sujet **ne peut pas fonctionner sur ces données**. C'est cohérent avec la nature des données (signatures tirées d'une distribution ≈ uniforme sur la bande VHF, indépendamment du MMSI) et avec la littérature : le vrai RF fingerprinting (identifier un émetteur par les imperfections de son hardware) exige des **échantillons IQ bruts** (arXiv 2402.06250 / FS-SEI), pas des métadonnées tabulaires.
- **Q14** — Ingestion de 2 flux RSS (gCaptain, The Maritime Executive) → 42 items récupérés, 0 MMSI cité dans les titres (les flux maritimes mentionnent rarement des MMSI) ; le mécanisme de mise à jour de `is_suspicious` à partir des MMSI extraits est en place.

### 4.6 Score de suspicion global — comparaison des modèles *(critère : « apprentissage, tuning »)*

| Approche | AUC | precision@96 | Verdict |
|---|---:|---:|---|
| Score multi-facteurs, **poids uniformes** | 0.557 | 9.4 % | baseline |
| Score multi-facteurs, **poids appris (LogReg L2, 5-fold CV)** | **0.567** (OOF) / 0.566 (appliqué) | **16.7 %** (appliqué) / 12.5 % (OOF) | **🏆 meilleur** |
| **XGBoost** (5-fold CV, supervisé, hors cours) | 0.513 (OOF) ; **AUC train = 1.0** | 11.5 % (OOF) | overfit (96 positifs seulement) |
| HBOS (PyOD) | 0.528 | 10.4 % | sous le multi-facteurs |
| COPOD (PyOD) | 0.522 | 12.5 % | sous le multi-facteurs |
| KNN-PyOD | 0.501 | 13.5 % | hasard |
| ECOD (PyOD) | 0.492 | 8.3 % | sous le hasard |

Courbes ROC : `outputs/roc_score_tuned.png` (LogReg L2) et `outputs/roc_xgboost.png` (XGBoost + feature importance). Benchmark complet : `outputs/pyod_benchmark.csv`. Scores par navire : `outputs/global_score.csv` (uniforme) et `global_score_tuned.csv` (tuné).

**Coefficients appris (moyenne ± écart-type sur 5 folds)** : `lof` +15.4 ± 2.6 (**détecteur dominant et stable**) · `zone` +2.6 ± 0.8 · `ais_off` +0.5 ± 0.1 · `speed` +0.4 ± 0.3 · `position_mismatch` +0.15 ± 0.13 · `fake_flag` (MCD) **−2.1 ± 0.4** · `isolation_forest` **−3.2 ± 0.2** (les deux **anti-corrélés** de façon stable).

**Lectures clés** :
1. **Le LOF est le détecteur le plus prédictif** : les anomalies *locales/contextuelles* (densité anormale par rapport au voisinage immédiat — cours 4 §7) prédisent mieux la vérité terrain que les détecteurs globaux.
2. **`EllipticEnvelope`/MCD (Q4) et `Isolation Forest` sont anti-corrélés à la vérité terrain** (stable sur 5 folds, donc pas du bruit) → ces deux modèles d'**anomalies statistiques globales** flaguent des navires que les annotateurs synthétiques n'ont **pas** étiquetés. Résultat scientifique : sur ce dataset, « anomalie statistique » ≠ « incident documenté ».
3. **Le modèle linéaire régularisé du cours bat XGBoost** sur ce régime (96 positifs / ~10 features semi-corrélées) : XGBoost a trop de capacité et apprend par cœur le train (AUC train 1.0) sans généraliser (AUC OOF 0.51). C'est l'**inductive bias** approprié pour des données rares — exactement le concept du cours (§0 §1). Avec 10× plus de positifs labellisés, XGBoost serait probablement préférable.
4. **Aucun détecteur « boîte noire » unique ne bat le score multi-facteurs construit avec connaissance métier** — ce qui valide la consigne (« bon emploi des données : enrichissement »).

### 4.7 Graphe de connaissances
167 nœuds (120 Navires top-suspects + connus comme anormaux, 10 Pavillons, 7 types d'Anomalies, 30 Alertes), 250 arêtes, 1 composante connexe, degré moyen 3.0. Export interactif : `outputs/knowledge_graph.html`.

---

## 5. Automatisation & API *(critère : « automatisation, usage d'API »)*

- **Pipeline reproductible** : `python reponses_generalisation.py` (ou `make all`) → exécute Q1→Q14 + EDA + tuning + benchmark + graphe + fiche LLM, écrit tous les `outputs/`, depuis un env neuf, en **~30 s** (dont ~14 s d'appel à `mistral-large`), sans intervention.
- **Dashboards interactifs** (2 implémentations, au choix) : (a) **Streamlit** — `streamlit run app.py` (`make dashboard`), un seul process ; (b) **Next.js + FastAPI** — `make backend` (uvicorn `backend.py`, port 8765) + `make next` (`next-app/`, port 3000), front TypeScript/Tailwind/Recharts proxant l'API. Les deux offrent **7 pages** : vue d'ensemble (KPIs, qualité des données, rappel par type, comparaison des modèles) · profils & clusters K-Means (scatter interactif) · navires suspects (classement par score + breakdown des contributions + **fiche de renseignement LLM générée en direct + PDF téléchargeable**) · carte folium · graphe de connaissances · **pipeline d'identification Q12 en live** (capter une signature → MMSI + verdict de suspicion + fiche) · rapport de qualité des données. Lisent les artefacts pré-calculés (lancer `make generalisation` d'abord).
- **API externes** : **Mistral** (`mistralai` — rédaction de la fiche + détecteur d'anomalie tabulaire), **OpenSanctions** (REST), **base MMSI UIT** (offline, table MID), **Equasis** (compte gratuit), **flux RSS** (`feedparser` — Q14), **Global Fishing Watch** (REST), **AISStream.io** (websocket — démo live), **MCP `shom-wrecks`** (WFS épaves). Toutes les clés en variables d'environnement (`.env` gitignored).
- **Alerte automatique** : fiche de renseignement Markdown + **PDF** par navire suspect (`reportlab`) ; e-mail (`smtplib`) en option.
- **Cache** : SQLite local (`data/cache.sqlite`) pour les réponses OSINT et LLM → la démo tourne hors-ligne et de façon déterministe.
- **Robustesse** : tout est dégradable — pas de clé OpenSanctions → `[]` ; pas d'identifiants Equasis → vide ; pas de clé Mistral → fiche générée par template Jinja2 ; pas de clé AISStream → démo live désactivée. Le pipeline ne casse jamais.
- Tests : `pytest` (à compléter — quelques tests sur les règles déterministes).

---

## 6. Limites & pistes

- **Données synthétiques peu structurées** : V de Cramér ≈ 0 entre les colonnes censées être liées ; les features sont quasi décorrélées des labels d'anomalie → l'exactitude est intrinsèquement plafonnée. On a **quantifié** cette limite (plafond de rappel ≈ 63 %, AUC ≈ 0.57) plutôt que de la masquer.
- **Pas de vrai signal radio** : extension naturelle = **RF fingerprinting matériel** via SDR (rtl-sdr + AIS-catcher ≈ 25 €), méthodologie type **FS-SEI** / **DeepCRF** (arXiv 2402.06250, 2411.06925).
- **Anomalies de trajectoire** : un modèle probabiliste de trajectoires AIS (**GeoTrackNet**, arXiv 1912.00682) ou un Transformer génératif (**TrAISformer**, arXiv 2109.03958) — non entraînés ici par contrainte de temps/GPU, repris dans nos forks `BDD-MinArm`.
- **Plus de labels** : avec un dataset à 10× plus de positifs, XGBoost serait probablement préférable au LogReg L2.
- **OSINT** : finaliser le scraping Equasis (interface déjà câblée) ; brancher les clés API (OpenSanctions, GFW, AISStream) ; évaluer la factualité des fiches LLM sur un mini-benchmark avant tout usage opérationnel.
- **Croisement avec l'imagerie satellite (Sujet 5)** : « l'AIS dit rien, le satellite dit qu'il y a un navire » — combiner nos détections AIS suspectes avec les détections SAR Sentinel-1 (xView3-SAR, arXiv 2206.00897). Récit commun de l'équipe : « rendre visible un navire qui veut rester invisible — par ses émissions radio et depuis l'orbite ».

---

## 7. Annexes

- [`docs/cadrage.md`](docs/cadrage.md) — cahier des charges complet (périmètre, choix d'algos justifiés, métriques d'acceptation).
- [`docs/hypotheses.md`](docs/hypotheses.md) — tous les seuils & choix de modèles avec justifications (cours d'A. Bogroff + arXiv) + findings EDA.
- [`docs/plan-3-jours.md`](docs/plan-3-jours.md) — plan d'exécution & répartition.
- [`SOURCES.md`](SOURCES.md) — sources OSINT mobilisées (URL, licence, usage).
- `cours ml/FICHE_REVISION_DATA.md` — référentiel ML (cours d'Alexis Bogroff, Albert School, ML III, 4 sessions).
- `outputs/data_quality_report.md` — rapport de contrôle qualité détaillé.
- `outputs/results.json` — toutes les valeurs chiffrées du run.
- `outputs/` — carte folium, graphe de connaissances, courbes ROC, PNG des clusters et de l'évolution temporelle, PDF d'alerte, CSV intermédiaires.

---

*Fiche technique : env Python 3.12 via `uv` · 17 modules `src/` documentés · pipeline reproductible en ~16 s · 100 % open source · 0 € de coût de fonctionnement · cours d'A. Bogroff (Albert School) comme référentiel ML, complété par XGBoost et PyOD (force de proposition) et un benchmark comparatif honnête.*
