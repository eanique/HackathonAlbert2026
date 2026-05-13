# Hypothèses, seuils & choix algorithmiques — Sujet 3

> Document **vivant**. Toute hypothèse / tout seuil / tout choix d'algo est noté ici avec sa justification (cours Bogroff ou arXiv). Repris tel quel en annexe du `rapport_generalisation.md`. Créé le 2026-05-11.

> **Convention de citation** : `[Cours N §M]` = cours d'Alexis Bogroff, Albert School (`cours ml/FICHE_REVISION_DATA.md`) ; `[arXiv ID]` = papier du doc de recherche.

---

## A. Conventions de données

| Convention | Détail |
|---|---|
| Types | `mmsi`, `imo` traités comme **chaînes** (jamais en int — perte de zéros, comparaisons faussées). |
| MMSI | 9 chiffres ; les **3 premiers = MID** (code pays attribué par l'UIT). MMSI hors MID connu ou non numérique → flag « MMSI invalide ». |
| Timestamps | Tous convertis en `datetime` **UTC**. |
| Tri AIS | **Obligatoire** : `sort_values(['mmsi','timestamp'])` avant tout calcul temporel (vitesse implicite, gaps, sauts). |
| Dédup | Lignes strictement identiques → drop. Doublons `(mmsi, timestamp)` → on garde la dernière. |
| Manquants | Loggés (combien, quelle colonne, traitement). Aucune imputation silencieuse. |
| Scaling | `StandardScaler` (`z = (x − µ)/σ`) avant tout algo basé distance/variance (K-Means, GMM, hiérarchique, DBSCAN, LOF, PCA, EllipticEnvelope) [Cours 0 §4]. ⚠️ **Exception : Isolation Forest ne demande PAS de scaling** [Cours 4 §6]. |

---

## B. Seuils & paramètres par question

### Partie 1 — Base de profils radio

| ID | Hypothèse / seuil | Justification |
|---|---|---|
| **Q1** Agrégation | Pour chaque navire : moyenne de `frequency`, `bandwidth`, `power`, `signal_strength`, `signal_to_noise_ratio` + `n_signatures`. Pour `pulse_pattern` et `modulation` (catégoriels) : mode + entropie. | Énoncé impose la moyenne. L'entropie sur le catégoriel capture la « variabilité » d'un navire qui change de mode. |
| **Q2** Pattern unique | `value_counts() == 1` sur `pulse_pattern` seul **ET** sur la combinaison `pulse_pattern × modulation × bande de frequency` | Le `pulse_pattern` seul = 6 valeurs possibles → peu discriminant ; la combinaison apporte une vraie unicité [piège du CLAUDE.md]. |
| **Q3** K-Means | **K = 5** imposé par l'énoncé. Pipeline : `StandardScaler` → `KMeans(n_clusters=5, n_init=10, init='k-means++', random_state=42)`. Features : `(frequency, bandwidth, power)`. On rapporte **inertie/WCSS**, **Silhouette score** ∈ [−1, 1], et **Elbow plot** sur K ∈ [2, 10] pour discuter. Viz : scatter `frequency × power` coloré par cluster + **PCA 2D** (cours 3 §2). | [Cours 1 §2-3] K-Means avec K-Means++ pour éviter les minima locaux ; `n_init=10` standard. Justification du biais sphérique : les 3 features sont à peu près convexes (à vérifier par la viz PCA). Silhouette pour quantifier la séparation [Cours 1 §3(c)]. |

### Partie 2 — Anomalies

| ID | Hypothèse / seuil | Justification |
|---|---|---|
| **Q4** Pavillon ≠ profil | Pour chaque `flag` (≥ 30 navires), fitter **`EllipticEnvelope` (MCD)** [Cours 4 §4] sur (frequency, bandwidth, power) des navires **non `is_suspicious`** de ce pavillon. `contamination=0.05` (par défaut). Pour chaque navire, calculer la **distance de Mahalanobis robuste** au cœur de son pavillon ; flaguer si `predict()==−1`. `confidence` du rapport = `(d − seuil) / seuil` normalisée puis bornée à [0, 1]. → **top 10 par `confidence` × MAD-z**. | [Cours 4 §4] **MCD** est *exactement* le bon outil : contourne le **masking effect** de Mahalanobis naïve (les anomalies présentes dans le dataset gonflent µ et Σ et se cachent). On utilise un sous-ensemble de `h ≈ n/2` points à déterminant de covariance minimal. ⚠️ Si un pavillon a < 30 navires, fallback : z-score robuste sur médiane ± MAD avec seuil `|z| > 3`. |
| **Q5** Changements de nom | `historical_names.str.split(',').str.len() > 2` → flag. Joindre avec `anomalies[type=='Name Change']` (sur `mmsi`). | Énoncé (« plus de 2 noms historiques »). |
| **Q6** Orphelines | `radio_signatures.mmsi ∉ ships.mmsi` → orphelines. Attribution proposée : **k-NN** dans l'espace standardisé `(frequency, bandwidth, power)`, k=5, distance Euclidienne. Score de confiance = `1/(1+d_mean)`. | [Cours 1 §2 inductive bias] La distance Euclidienne est cohérente avec l'usage du K-Means dans le même espace. |
| **Q7** AIS off | Sur l'AIS trié par mmsi+timestamp : détection des plages contiguës `ais_active==False` ; **durée = dernier − premier timestamp** de la plage. Flag si **> 24 h**. ⚠️ Exception : on **exclut** les plages où le navire est `status ∈ {Moored, At Anchor}` (mouillage légitime). Si pas de filtre possible, on rapporte la limite. | Énoncé. Algorithme inspiré de `pipe-gaps` (Global Fishing Watch). Réduit faux positifs en distinguant AIS-off au port vs en mer. |
| **Q8** Écart AIS↔radio | `merge_asof` par mmsi (tolérance ±10 min) → apparier chaque point radio au point AIS le plus proche. `geopy.distance.geodesic` (km). Flag si **distance > 1 km**. Rapporter écart **médian** et **max** par navire. | Énoncé. `merge_asof` car les timestamps AIS et radio ne sont pas alignés exactement. |

### Partie 3 — Temporel/stats

| ID | Hypothèse / seuil | Justification |
|---|---|---|
| **Q9** Changement brutal | Pour un MMSI, tracer `frequency(t)` et `signal_strength(t)`. Flag : **\|Δfrequency\| > 1 MHz** OU **\|Δsignal_strength\| > 15 dBm** entre deux points consécutifs. Détection complémentaire : rupture **CUSUM** ou Bayesian Online Changepoint Detection (lib `ruptures`) en bonus. | Énoncé donne l'exemple 156.8 → 160.0 MHz (Δ = 3.2 MHz). Le seuil 1 MHz cadre les sauts inter-canaux VHF maritime. |
| **Q10** Stats par pavillon | `radio.merge(ships[['mmsi','flag']]).groupby('flag').frequency.agg(['mean','std','count'])`. `.idxmax()` sur la moyenne. Barplot avec barres d'erreur (écart-type / √n). | Énoncé. |
| **Q11** Corrélation speed↔freq | Apparier AIS et radio (`merge_asof` par mmsi+timestamp). `scipy.stats.pearsonr(speed, frequency)` → r + p-value. « Significatif » si **p < 0.05 ET \|r\| > 0.1**. | Hypothèse a priori : **non significative** (la fréquence VHF ne dépend pas physiquement de la vitesse) → une réponse négative argumentée est valide et attendue. |

### Partie 4 — Automatisation & validation

| ID | Hypothèse / seuil | Justification |
|---|---|---|
| **Q12** Pipeline | Identification = k-NN sur `ship_radio_profiles.csv` standardisé, k=1 (MMSI le plus proche) avec score `1/(1+d)`. Décision « suspect » = **One-Class SVM** [Cours 4 §5] entraîné sur le « normal propre » (navires `is_suspicious==False` + signatures hors `anomalies_large`), kernel RBF, **`nu=0.05`** (5 % d'outliers tolérés dans le training [Cours 4 §5]). Si OCSVM dit −1 OU score k-NN < seuil OU score multi-facteurs > seuil → alerte. | [Cours 4 §3] Distinction outlier (training pollué) vs **novelty** (training propre, cas idéal). Ici on simule un training propre ⇒ OCSVM est l'outil de référence. `ν` à calibrer en cross-val sur F1 ou rappel. |
| **Q13** Validation | 10 signatures aléatoires (3 graines différentes → moyenne ± écart). Matrice de confusion (MMSI prédit vs réel, suspect/non-suspect). | Énoncé. |
| **Q14** Mise à jour auto | Moyenne glissante des profils (`profile_new = (n·profile_old + sig)/(n+1)`, n = nb de signatures). Re-fit OCSVM toutes les K nouvelles obs. Ingestion RSS via `feedparser` : extraire les MMSI/IMO cités → set `is_suspicious=True`. | POC fonctionnel de l'un + description écrite de l'autre. |

### Leviers de valeur ajoutée

| Élément | Choix | Justification |
|---|---|---|
| **Isolation Forest global (L3)** | `IsolationForest(contamination=0.02, n_estimators=200, random_state=42)` sur `radio_signatures_large.csv` enrichi (one-hot de `pulse_pattern` et `modulation`). ⚠️ **PAS** de StandardScaler. | [Cours 4 §6] iForest scalable, robuste aux échelles, parfait pour la haute dimension après one-hot. `contamination=0.02` = 100/5000 ≈ taux observé dans `anomalies_large.csv`. `s ≈ 1` → anomalie ; `s < 0.5` → normal. |
| **LOF complémentaire (L3)** | `LocalOutlierFactor(n_neighbors=20, contamination=0.02)`, après StandardScaler. Mode outlier (sur le training pollué). | [Cours 4 §7] LOF capte les anomalies **locales/contextuelles** (un point qui paraît normal globalement mais l'est moins dans son voisinage). `n_neighbors=20` standard ; range typique 10-50. |
| **Règles de spoofing (L3)** | (a) **MMSI dupliqué** = même MMSI émettant depuis 2 positions incompatibles au même instant (Δt < 5 min, distance > 10 km) ; (b) **timestamp incohérent** = futur ou antérieur au précédent du même MMSI ; (c) **intervalle de transmission anormal** = écart médian local hors plage attendue pour `navigational_status` ; (d) **vitesse implicite anormale** = distance géodésique entre 2 positions AIS consécutives / Δt > 50 km/h (~27 nœuds) sauf navires militaires. Chaque règle → flag + `confidence` ∈ [0,1]. | arXiv 2603.11055 (« Wide-Area GNSS Spoofing & Jamming Detection »), 2602.16257 (SeaSpoofFinder). Sauts de position implausibles → patterns récurrents Baltique/mer Noire/Mourmansk → récit géopolitique Minarm. |
| **Sous-score zone-dépendant (L3)** | Grille spatiale **cellules de 1°×1°** sur les positions AIS. Pour chaque cellule, distribution empirique (lissée KDE) de `(cap, vitesse, statut)`. Score = `−log p` du comportement observé, écrêté à `[0, 10]`. | Inspiré de **GeoTrackNet** (arXiv 1912.00682) en version light : un comportement normal au large d'un port l'est moins en plein océan. Pas d'entraînement DL. |
| **Anomalie tabulaire LLM (L3)** | Mistral `mistral-small-latest`, `temperature=0`, prompt few-shot : on présente 20 lignes « normales » de `radio_signatures` + la ligne à tester ; on demande un score ∈ [0,1] + une justification d'une phrase. Cache des réponses par hash de prompt. | arXiv 2406.16308. Complément zero-shot, faible poids dans le score global (5 % par défaut). |
| **Score de suspicion global** | `score = w_flag · z(flag_pavillon) + w_name · 1(>2 noms) + w_orph · 1(orpheline) + w_aisoff · z(durée_off) + w_pos · z(écart) + w_speed · z(vitesse_implicite) + w_iforest · score_iforest + w_lof · z(LOF) + w_spoofing · max(règles) + w_zone · score_zone + w_llm · score_llm`, normalisé [0,1]. Poids initiaux égaux puis **ajustés pour maximiser l'AUC vs `is_suspicious`** sur un split holdout 70/30. | Transparent, justifiable, traçable. AUC + precision@k rapportés. |
| **Métrique business** | Cadre **coût asymétrique** [Cours 4 §8] : `coût_total = α · FP + β · FN` avec `β/α ≈ 10` (rater un suspect = 10× plus grave qu'alerter à tort, contexte Minarm). On optimise les seuils sur ce coût et sur le **rappel**, pas sur l'accuracy. | [Cours 4 §8] Anomalies < 1 % → accuracy inutile. |

---

## C. Choix d'environnement / outils

| Choix | Décision | Justification |
|---|---|---|
| Python | **3.12 via `uv`** | Le système 3.9 est trop vieux pour les libs récentes. |
| Stockage | CSV + Parquet (pour intermédiaires lourds) + SQLite local pour cache OSINT/alertes | Pas de Supabase : 16k lignes total, pandas en mémoire ; pas de dépendance externe pendant la démo. |
| LLM | **Mistral** (`mistralai` SDK) ; fallback Anthropic → Ollama → template Jinja2 | Souveraineté FR (cohérent Minarm). |
| Carte | Folium (HTML statique, sans serveur) | Léger, portable, démo-friendly. |
| Graphe | **NetworkX** (+ pyvis pour viz) si besoin de cocher le critère « graphes de connaissance » | Inutile d'installer Neo4j pour 10k nœuds. |
| Lint/test | `ruff` + `pytest` | Légers, standards. |
| Reproductibilité | `Makefile` (`make install`, `make all`, `make report`) + `requirements.txt` figé | Pas d'usine à gaz. |

---

## D. Pièges du cours à éviter explicitement

1. ✅ **Scaler avant tout sauf iForest** [Cours 4 §6, Memento piège #1].
2. ✅ **K-Means** : `n_init=10` et K-Means++ par défaut [Cours 1 §2].
3. ✅ **Curse of dimensionality** [Cours 0 §3] : on garde les features à 3-6 dims max après one-hot ou on PCA d'abord (sans aller trop loin).
4. ✅ **t-SNE** : si on l'utilise pour viz, on rappelle dans le rapport que **la position, la distance entre clusters et la forme globale ne sont PAS interprétables** [Cours 3 §3, Memento piège #6].
5. ✅ **Masking effect** : EllipticEnvelope / MCD pour Q4, pas Mahalanobis naïve [Cours 4 §4, Memento piège #10].
6. ✅ **iForest** : score `s ≈ 1` = anomalie (chemin court), `s < 0.5` = normal [Memento piège #11].
7. ✅ **LOF** : `LOF ≫ 1` = anomalie ; seul détecteur d'anomalies locales [Memento piège #12].
8. ✅ **OCSVM** : `ν` bas = frontière large (risque FN) ; haut = frontière serrée (risque FP) [Memento piège #13].
9. ✅ **Évaluation à coût asymétrique** : pas d'accuracy, on calibre FP/FN selon le contexte business [Memento piège #14].
10. ✅ **PCA ≠ SVD** : SVD est le moteur, PCA est le but [Memento piège #7] — on cite correctement dans le rapport.

---

## E. Findings de l'EDA (module `src/eda.py` → `outputs/data_quality_report.md`)

Mesures objectivées (reproductibles via `python reponses_generalisation.py`) :

| Constat | Mesure | Conséquence sur la méthodo |
|---|---|---|
| Donnée formellement propre | 0 NaN (sauf `historical_names` 249 = légitime), 0 doublon, 0 hors-bornes vs dictionnaires, 0 catégorie hors-liste, 0 MMSI orphelin, 0 régression temporelle | la normalisation de `load.py` suffit ; pas de correction de données |
| `navigational_status` ⊥ `status` | V de Cramér = 0.03 (≈ 0 ; vrai AIS ≈ 1) | `navigational_status` non utilisé comme feature discriminante |
| `ais_active` ⊥ `status`, 50/50 | V de Cramér = 0.01 ; `ais_active=True` = 50.1 % | l'AIS-off (Q7) est utilisé mais on sait qu'il est artificiellement fréquent |
| Positions AIS = points aléatoires | vitesse implicite (géodésique/Δt) : médiane 9 kn, p95 134 kn, **max 12 338 kn** | la « vitesse implicite » comme règle de spoofing reste codée mais on s'attend à un poids ≈ 0 après tuning (confirmé) ; on le documente |
| `name` (large) = placeholder | 100 % == `NAVIRE-` + 4 derniers chiffres MMSI | identification par MMSI ; `name` non utilisé comme feature |
| `historical_names` = templates | distribution ≈ uniforme 0/1/2/3 (`OLD_NAME_k`) | seul le *compte* est exploitable (Q5) ; NaN ⇒ 0 (jamais renommé), pas une donnée manquante |
| **Plafond de rappel ≈ 63 %** | 16 « Speed » + 21 « Course » = 37/100 anomalies dont la valeur n'est que dans la description (`speed` plafonné à 30 kn, descriptions citant jusqu'à 49.2 kn) | on **ne tente pas** de détecter Speed/Course Anomaly depuis les tables (circulaire) ; on rapporte le plafond ; l'AUC ≈ 0.57 et le rappel global ≈ 14 % sont cohérents avec ce plafond + la décorrélation features↔labels |
| 6 navires sans signature radio | `363951510, 435131361, 522451298, 534553120, 535117872, 675464001` (4 `is_suspicious`) | marqués `has_radio_signature=False` ; absents de `ship_radio_profiles.csv` (994, pas 1000) ; exploités comme signal de suspicion (`detect_silent_ships`) |

**Choix de modèles validés par le benchmark** (rapport §4.6) : sur ce régime (96 positifs, ~10 features semi-corrélées), le **score multi-facteurs + LogReg L2 5-fold CV** (cours) bat **XGBoost** (overfit, AUC train 1.0 / OOF 0.51) et les 5 détecteurs **PyOD** (ECOD/COPOD/HBOS/KNN/PCA-OD, tous AUC ≤ 0.53). Le **LOF** (cours 4 §7) est le plus prédictif (coef L2 +15.4 ± 2.6) ; `EllipticEnvelope`/MCD (Q4) et `IsolationForest` sont **anti-corrélés** à la vérité terrain (coef −2.1 et −3.2, stables) → sur ce dataset, « anomalie statistique » ≠ « incident documenté ».
