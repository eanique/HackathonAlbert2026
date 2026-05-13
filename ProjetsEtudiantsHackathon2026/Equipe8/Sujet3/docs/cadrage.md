# Cahier des charges — Sujet 3 : Aide à l'identification des navires par analyse intelligente de leur radio signature

> Équipe **BDD-MinArm** · Hackathon Albert School mai 2026 · Commanditaire Ministère des Armées. Doc de cadrage **formel** (≈ PRD), issu du skill `cadrage-hackathon`. Daté du 2026-05-11. **À mettre à jour au fil de l'avancée.** Référentiel ML : cours « Unsupervised Learning Problems » d'Alexis Bogroff, Albert School (`cours ml/FICHE_REVISION_DATA.md`).

---

## 0. Mode d'exécution — « 2 agents IA » (Claude)

Le travail est exécuté par **deux streams parallèles tenus par Claude (le même modèle, joué sur 2 rôles)**, orchestrés via le tool `Agent` pour les briques indépendantes :

| Agent | Périmètre | Modules |
|---|---|---|
| **Agent A — « Data & Détection »** | Données, statistiques, modèles d'anomalie « cours », validation interne | `load.py`, `profiles.py`, `cluster.py`, `anomalies.py`, P3 stats (Q9–Q11), évaluation precision/recall vs `anomalies_large.csv`, visualisations |
| **Agent B — « Pipeline & Renseignement »** | Score global, règles de spoofing, pipeline d'identification, enrichissement OSINT, fiche LLM, automatisation, rapport | `spoofing_rules.py`, `anomaly_score.py`, `identify.py`, `osint_enrich.py`, `intel_report.py` (Mistral), `maps.py`, `Makefile`, `rapport_generalisation.md` |

**Règles d'orchestration** : (i) chaque agent travaille dans son périmètre, écrit des fonctions pures aux signatures clairement spécifiées (cf. § 9) ; (ii) point de synchronisation en fin de chaque phase = revue croisée + tests d'intégration ; (iii) un seul propriétaire par module (pas de double édition concurrente) ; (iv) chaque livraison passe par `ruff` + un mini test pytest avant intégration ; (v) le `rapport_generalisation.md` est assemblé en fin par Agent B à partir des notes d'analyse critique laissées par les deux agents dans chaque section du `reponses_generalisation.py`.

> **Garde-fou de rigueur** : tout choix de modèle, de seuil ou de métrique doit citer le cours (ex. `// cours 4 §6, Isolation Forest, contamination ≈ 2 %`) ou un papier arXiv (cf. `docs/recherche-arxiv-sujets-3-5.md` dans `BDD-MinArm/hackathon-2026`).

---

## 1. Contexte & problème

Identifier un navire de façon **fiable et passive** (sans dépendre de son AIS, qui peut être éteint, falsifié, manipulé) en exploitant sa **signature radio** : identifier (nom, type, pavillon, destination), localiser même sans AIS, détecter des anomalies (faux pavillon, changement de nom, spoofing, AIS off, écart de position), constituer une **base de profils radio** pour identification passive. Livrables imposés : `reponses_mise_en_jambe.py`, `reponses_generalisation.py`, `rapport_generalisation.md`.

**Critères de notation à viser** (consigne officielle) : (1) usage de libs data science / IA (NLP, LLM, automatisation, graphes, viz) · (2) qualité des données (normalisation, nettoyage, enrichissement, labellisation) · (3) précision & complétude des résultats · (4) automatisation & API · (5) collaboration · (6) clarté & professionnalisme de la présentation. **Le jury survalorise la découverte de sources OSINT pertinentes.**

---

## 2. Le « woah » visé (en une phrase)

> Une **carte de surveillance côtière** + une **fiche de renseignement générée par un LLM** à partir de sources OSINT libres gratuites — prouvant qu'on peut faire du **produit de renseignement réel** sans hardware coûteux, en s'appuyant sur **une méthodologie d'apprentissage non supervisé rigoureuse** (cours Albert School : EllipticEnvelope/MCD pour les anomalies par pavillon, Isolation Forest pour les anomalies globales, One-Class SVM pour le pipeline d'identification passive en novelty detection).

---

## 3. Périmètre IN / OUT (la restriction est explicitement valorisée)

### 3.1 IN — exigences fonctionnelles (les 14 questions de la Généralisation, à fond)

| ID | Question (synthèse) | Algo / méthode du cours mobilisée |
|---|---|---|
| **Q1** | Agrégation des signatures par navire → `ship_radio_profiles.csv` ; top 5 freq moyenne | `pandas.groupby` ; aucune ML. |
| **Q2** | `pulse_pattern` uniques ; nb de patterns distincts | `value_counts()` ; commentaire critique : seulement 6 valeurs → on rapporte aussi l'unicité de la **combinaison** `pulse_pattern × modulation × bande de frequency`. |
| **Q3** | Clustering K=5 sur (frequency, bandwidth, power) + viz 2D | **K-Means (cours 1)** avec `StandardScaler` obligatoire, `n_init=10`, K-Means++ par défaut. Justification de K=5 imposé par l'énoncé MAIS on rapporte aussi **Elbow plot** (inertie/WCSS) et **Silhouette score** (∈ [−1,1]) pour discuter (cours 1 §3). Viz : scatter `frequency × power` coloré par cluster + **PCA 2D** (cours 3) en complément. |
| **Q4** | Pavillon ≠ profil radio typique → top 10 par confidence | **EllipticEnvelope / MCD par pavillon (cours 4 §4)** — *exactement* le bon outil : pour chaque pavillon, fitter une covariance robuste sur (frequency, bandwidth, power) des navires non-suspects → flaguer les écarts. Contourne le **masking effect** de la distance de Mahalanobis naïve (cours 4 §4). Confidence = distance de Mahalanobis robuste normalisée. |
| **Q5** | `historical_names` > 2 noms ; lier aux anomalies `Name Change` | `pandas` + jointure ; eval vs `anomalies_large[type=='Name Change']`. |
| **Q6** | Signatures orphelines + méthode d'attribution | `set(radio.mmsi) − set(ships.mmsi)` ; attribution par **k-NN dans l'espace standardisé** (frequency, bandwidth, power) — cf. cours 1 (logique de distance Euclidienne). Eval vs `anomalies[type=='Spoofing']`. |
| **Q7** | AIS off > 24 h consécutives | Tri AIS par mmsi+timestamp ; détection de plages contiguës `ais_active==False` ; durée > 24 h. Eval vs `anomalies[type=='AIS Disabled']`. Plan B : règles de [`pipe-gaps`] de Global Fishing Watch ré-implémentées. |
| **Q8** | Écart position AIS↔radio > 1 km + carte folium | `merge_asof` par mmsi (apparier les points dans le temps) ; `geopy.distance.geodesic` ; carte folium. Eval vs `anomalies[type=='Position Mismatch']`. |
| **Q9** | Évolution frequency / signal_strength d'un MMSI ; changements brutaux | Séries temporelles ; seuil `|Δfreq| > 1 MHz` ; rupture CUSUM (lib `ruptures`) en bonus. |
| **Q10** | Moyenne & écart-type frequency par pavillon | `groupby.agg` ; barplot avec barres d'erreur. |
| **Q11** | Corrélation `speed`(AIS) ↔ `frequency`(radio) | `scipy.stats.pearsonr` après `merge_asof` ; hypothèse a priori : non significatif (la fréquence VHF ne dépend pas physiquement de la vitesse) → **une réponse négative argumentée est valide et attendue**. |
| **Q12** | Pipeline d'identification passive (signature → MMSI + suspect + alerte) | **Novelty Detection (cours 4 §3)** : entraîner sur le « normal propre » (`is_suspicious==False`) → on peut combiner k-NN d'identification + **One-Class SVM (cours 4 §5)** pour décider si la signature est « du normal » ou « nouvelle » ; ν calibré sur la performance. Fiche LLM générée (cf. § 5). |
| **Q13** | Validation sur 10 signatures aléatoires → taux d'ID correcte | Tirage répété (seed), matrice de confusion. **Métriques business à coût asymétrique (cours 4 §8)** : on optimise vers le **rappel** (rater un navire suspect = très coûteux ≫ alerter à tort un cargo normal). |
| **Q14** | Mise à jour auto + ingestion OSINT (flux RSS) | POC moyenne glissante des profils + `feedparser` qui met à jour `is_suspicious`. |

### 3.2 IN — Leviers (valeur ajoutée, 0 €)

- **Levier L2 (Enrichissement OSINT + LLM)** — sortie : la fiche de renseignement (PDF + Markdown) générée par Mistral. Sources gratuites : Equasis, base MMSI UIT/MARS, OpenSanctions, flux RSS maritime. Inscription requise (Personne / Agent B, J1).
- **Levier L3 (Score d'anomalie sérieux + règles de spoofing)** — cœur de l'évaluation :
  - **Score multi-facteurs** = somme pondérée × `confidence` des flags Q4–Q8 + règles spoofing + sous-score zone-dépendant.
  - **Isolation Forest (cours 4 §6)** entraîné sur `radio_signatures_large.csv` (features = frequency, bandwidth, power, signal_strength, snr, pulse_pattern one-hot, modulation one-hot) → score d'anomalie continu, intégré au score global. ⚠️ **Pas de scaling** (le cours insiste : iForest est invariant aux échelles, contrairement aux autres). `contamination ≈ 0.02` (≈ 100 anomalies / 5000 signatures, cf. `anomalies_large.csv`).
  - **LOF (cours 4 §7)** en complément, pour capter les **anomalies locales/contextuelles** (un navire dont le profil est OK globalement mais détonne dans son sous-groupe — ex. dans son pavillon). `n_neighbors=20`, `contamination=0.02`.
  - **Règles de spoofing** (arXiv 2603.11055 / SeaSpoofFinder) : MMSI dupliqués, timestamps incohérents, intervalles de transmission anormaux, sauts implausibles.
  - **Sous-score zone-dépendant** (façon GeoTrackNet light, arXiv 1912.00682) : grille spatiale ~1°×1° sur les positions AIS, distribution empirique de (cap, vitesse, statut) par cellule, score = `−log p` écrêté.
  - **Anomalie tabulaire par LLM** (arXiv 2406.16308) en complément zero-shot, faible poids, sur l'API Mistral.
- **Tuning du score global** : ajustement des poids des facteurs pour **maximiser l'AUC vs `is_suspicious`** (label de référence, jamais utilisé en entrée du détecteur, uniquement pour évaluer).

### 3.3 OUT — explicitement exclu et justifié dans le rapport

- ❌ **Vrai traitement de signal SDR / IQ samples** : `radio_signatures.csv` = métadonnées tabulaires (frequency, bandwidth, modulation, power, pulse_pattern, SNR), **pas d'échantillons IQ**. → présenté comme extension future (rtl-sdr + AIS-catcher ; RF fingerprinting matériel type FS-SEI / arXiv 2402.06250). **C'est le piège central du sujet : ne pas survendre.**
- ❌ **Deep learning lourd entraîné from scratch** (GeoTrackNet/TrAISformer complets, modèles SAR, …) : pas de temps, pas de GPU → cités, version light en pandas/numpy pour le sous-score zone-dépendant.
- ❌ **Neo4j / Supabase** : NetworkX en mémoire (+ pyvis pour la viz) suffit pour le mini-graphe optionnel navires↔anomalies↔alertes. Stockage = CSV + SQLite local pour le cache OSINT et le journal d'alertes.
- ❌ **API payantes** (MarineTraffic, Spire) : on reste sur du gratuit (et le jury valorise l'OSINT libre).
- ❌ **SDR matériel** : budget 0 € confirmé. Plan B = AIS public Danish Maritime Authority + AISStream.io websocket (clé gratuite).

---

## 4. Référentiel ML — choix d'algorithmes (rigueur cours Bogroff)

> Toutes les justifications dans `docs/hypotheses.md`. En résumé, le « pourquoi » de chaque modèle :

| Tâche | Algo retenu | Justification cours |
|---|---|---|
| **Clustering radio (Q3)** | **K-Means K=5** + StandardScaler + Silhouette | Imposé par l'énoncé. Biais inductif sphérique du K-Means **acceptable** ici car (frequency, bandwidth, power) sont à peu près convexes/sans formes exotiques (à vérifier par PCA 2D). |
| **Anomalies par pavillon (Q4)** | **EllipticEnvelope (MCD)** | Cours 4 §4 : permet d'estimer un « cœur dense » par pavillon **sans masquage** par les anomalies. *Mahalanobis naïve* aurait été contaminée par les navires suspects présents dans le dataset. |
| **Anomalies globales / score (L3)** | **Isolation Forest** + **LOF** (en complément local) | Cours 4 §6 : iForest scalable, sans scaling, parfait pour la haute dimension (avec one-hot du pulse_pattern et modulation). Cours 4 §7 : LOF rattrape les anomalies **contextuelles** (un point qui paraît normal globalement mais l'est moins dans son voisinage). |
| **Pipeline d'identification passive (Q12)** | k-NN d'identification + **One-Class SVM** en novelty detection | Cours 4 §5 et §3 : *novelty* parce qu'on a (au moins théoriquement) un training propre (`is_suspicious==False`). OCSVM borne le « normal » → tout ce qui sort = nouvelle signature suspecte. Kernel RBF, ν calibré sur F1 ou rappel. |
| **Viz 2D des clusters (Q3 / Q4)** | **PCA** (linéaire) | Cours 3 §2 : axes interprétables, contre la colinéarité ; on garde 2 PC pour la viz, on rapporte le scree plot (variance cumulée). |
| **Réduction de dim. pour clustering robuste (option)** | **UMAP** avant K-Means / HDBSCAN | Cours 3 §3 : si le clustering en dim. brute donne des résultats peu nets, UMAP préserve mieux la structure globale que t-SNE et permet de réduire à dim. arbitraire avant K-Means. |
| **Détection d'anomalies tabulaires par LLM** | Mistral zero-shot | arXiv 2406.16308 ; complément, faible poids dans le score. |

---

## 5. Architecture LLM (cf. choix : **Mistral**)

- **Backend** : Mistral (`mistralai` SDK Python), modèles `mistral-small-latest` (par défaut, économique) ou `mistral-large-latest` (rédaction soignée de la fiche).
- **Pourquoi Mistral** : choix narratif (souveraineté numérique française → cohérent avec un livrable Minarm) + qualité suffisante pour nos deux usages : (a) rédiger la fiche de renseignement à partir des données enrichies ; (b) détecteur d'anomalies tabulaires zero-shot (arXiv 2406.16308).
- **Variable d'env** : `MISTRAL_API_KEY` (dans un `.env` non committé, géré par `python-dotenv`). Backend configurable via `LLM_BACKEND=mistral|anthropic|ollama|template`.
- **Garde-fous** :
  - Aucun appel LLM dans une boucle non bornée ; cache des réponses dans `data/cache.sqlite` (par hash du prompt) pour rejouer la démo sans réseau.
  - `temperature ≤ 0.3` pour la fiche de renseignement (factuel) ; `temperature = 0` pour l'anomalie tabulaire (déterministe).
  - Le prompt système rappelle le contexte Minarm et la sobriété attendue.
  - Fallback hiérarchique : Mistral KO → Anthropic (si dispo) → Ollama (si dispo) → template Jinja2.

---

## 6. Données

### 6.1 Fournies (base notée)
- `ships_large.csv` (1 000) · `radio_signatures_large.csv` (5 000) · `ais_data_large.csv` (10 000) · `anomalies_large.csv` (100, **vérité terrain**) · `MiseEnJambe/*_small.csv` (~20) · dictionnaires JSON.

### 6.2 OSINT temps réel & enrichissement (gratuit)
| Source | Type | Clé | Usage |
|---|---|---|---|
| **Equasis** (`equasis.org`) | Web/scraping (compte gratuit) | identifiants en `.env` | historique noms/pavillons, PSC, armateur (Q4, Q5) |
| **UIT MMSI / MARS** (`itu.int/mmsi`) | Lookup web | — | validité MMSI, MID (3 premiers chiffres) → pays attribué (Q6) |
| **OpenSanctions** (`opensanctions.org`) | API REST | `OPENSANCTIONS_API_KEY` | navire/armateur sous sanctions OFAC/UE |
| **Flux RSS maritime** (gCaptain, Maritime Executive…) | RSS | — | Q14 : ingestion OSINT auto |
| **AISStream.io** | WebSocket temps réel | `AISSTREAM_API_KEY` | démo « live » : flux AIS réel d'une bbox → carte qui s'actualise + détecteurs |
| **Global Fishing Watch API** | REST + client Python | `GFW_API_TOKEN` | gap events / encounters / loitering **pré-calculés** sur l'AIS mondial — récit géopolitique Baltique/mer Noire |
| **MCP `shom-wrecks`** | déjà branché (4 796 épaves SHOM) | — | écarter faux positifs de position (épave connue ≠ navire) |
| **Danish Maritime Authority** (`web.ais.dk/aisdata`) | CSV gratuits | — | plan B : rejouer les détecteurs sur AIS réel historique |

**Cache local** : `data/cache.sqlite` (tables : `osint_equasis`, `osint_opensanctions`, `rss_items`, `gfw_events`, `llm_responses`, `alerts`).
**Sécurité** : toutes les clés via variables d'env, listées dans `README.md`, **jamais committées** ; `.env` dans `.gitignore`.

---

## 7. Métriques d'acceptation (les chiffres du pitch)

| Métrique | Cible | Comment on la mesure |
|---|---|---|
| **Rappel global du score de suspicion** vs `is_suspicious` | **≥ 80 %** à un taux de FP ≤ 20 % | courbe ROC + AUC ; precision@k ; tableau par type d'anomalie |
| **Precision/recall par détecteur** (Q4, Q5, Q6, Q7, Q8) vs `anomalies_large.csv` filtré par `type` | rapport détaillé | matrice de confusion par type |
| **Taux d'identification correcte (Q13)** | **≥ 8 / 10** | tirages multiples (seed), moyenne ± écart |
| **Silhouette score (Q3)** | rapporté | discussion de la pertinence de K=5 (cours 1) |
| **AUC du score global** | rapporté | courbe ROC vs `is_suspicious` |
| **Temps d'exécution du pipeline complet** | ≤ 60 s | mesuré dans `Makefile` |

**Cadre business (cours 4 §8)** : l'évaluation se fait à **coût asymétrique** — `coût(FN)` ≫ `coût(FP)` (rater un navire suspect dans un contexte Minarm = bien plus grave qu'une alerte à tort). Donc on **calibre vers le rappel**, et on rapporte explicitement ce trade-off dans le rapport.

---

## 8. Automatisation & API

- **Pipeline reproductible** : `python reponses_generalisation.py` (ou `make all`) exécute Q1→Q14 de bout en bout depuis un env neuf, lit `data/raw/`, écrit `data/processed/` + `outputs/`, **sans intervention**.
- **API externes** : Mistral (LLM), OpenSanctions (REST), Equasis (web), UIT (lookup), feedparser (RSS), AISStream.io (websocket), GFW (REST via `gfw-api-python-client`), MCP `shom-wrecks`.
- **Alerte automatique** : **PDF** par navire suspect (`reportlab`) avec la fiche de renseignement ; option **e-mail** (`smtplib`).
- **Cache** : SQLite local, TTL configurable par source.
- **Tests** : `pytest` sur les règles déterministes (spoofing, détecteurs Q4–Q8) → un mini test par règle assurant `precision ≥ x` sur un mini-corpus connu.

---

## 9. Architecture du projet (arborescence cible)

```
sujet3/
├── data/
│   ├── raw/                      ← CSV fournis (symlink/copie)
│   ├── processed/                ← ship_radio_profiles.csv, etc.
│   └── cache.sqlite              ← cache OSINT + journal d'alertes
├── src/
│   ├── __init__.py
│   ├── config.py                 ← chargement .env, constantes (chemins, seuils)
│   ├── load.py                   ← lecture + normalisation (Agent A)
│   ├── profiles.py               ← agrégation → profils navire — Q1 (Agent A)
│   ├── cluster.py                ← KMeans K=5 + Silhouette + PCA viz — Q3 (Agent A)
│   ├── anomalies.py              ← Q4 (EllipticEnvelope par flag), Q5, Q6, Q7, Q8 (Agent A)
│   ├── spoofing_rules.py         ← règles MMSI/timestamp/intervalle (Agent B)
│   ├── anomaly_score.py          ← IF + LOF + score multi-facteurs + sous-score zone (Agent B)
│   ├── identify.py               ← k-NN + OCSVM novelty — Q12 (Agent B)
│   ├── osint_enrich.py           ← Equasis / UIT / OpenSanctions / RSS / GFW (Agent B)
│   ├── intel_report.py           ← fiche LLM Mistral + PDF (Agent B)
│   ├── realtime_ais.py           ← client AISStream.io (Agent B, optionnel)
│   ├── maps.py                   ← helpers folium (Agent B)
│   └── llm.py                    ← backend LLM configurable (Mistral / Anthropic / Ollama / template)
├── notebooks/                    ← exploration (un par agent)
├── outputs/                      ← carte HTML, PNG, PDF d'alerte, tableau métriques
├── docs/                         ← ce fichier, hypotheses.md, plan-3-jours.md
├── reponses_mise_en_jambe.py     ← LIVRABLE : Mise en jambe Q1→Q12 (Agent A)
├── reponses_generalisation.py    ← LIVRABLE : Généralisation Q1→Q14
├── rapport_generalisation.md     ← LIVRABLE
├── README.md                     ← lancement, env vars, sources
├── Makefile                      ← `make install` / `make all` / `make report`
├── requirements.txt              ← pandas, numpy, scikit-learn, scipy, geopy, folium, matplotlib, plotly, feedparser, requests, joblib, reportlab, mistralai, python-dotenv, websockets, ruff, pytest, networkx, pyvis, shapely, ruptures (optionnel)
└── .env.example                  ← MISTRAL_API_KEY=... AISSTREAM_API_KEY=... GFW_API_TOKEN=... OPENSANCTIONS_API_KEY=... EQUASIS_USERNAME=... EQUASIS_PASSWORD=...
```

Env Python : **3.12 via `uv`** (`uv venv` + `uv pip install -r requirements.txt`) — le système 3.9 est trop vieux.

---

## 10. Plan d'exécution & livrables

→ voir [`plan-3-jours.md`](plan-3-jours.md) (3 phases logiques, mode 2 agents IA).

**Livrables finaux** déposés dans `ProjetsEtudiantsHackathon2026/<NomGroupe-Sujet3>/` : `EQUIPE.md`, `reponses_mise_en_jambe.py`, `reponses_generalisation.py`, `rapport_generalisation.md`, `README.md`, `requirements.txt`, `data/processed/ship_radio_profiles.csv`, `outputs/`.

---

## 11. Risques & parades

| Risque | Parade | Minimum viable garanti |
|---|---|---|
| Données synthétiques avec patterns trop faciles ou incohérents | Le dire dans le rapport ; valider rigoureusement contre `anomalies_large.csv` ; enrichir avec OSINT réel | — |
| Equasis indisponible / lent | Cache local SQLite ; plan B = se contenter de `historical_names` du CSV | la fiche LLM se génère quand même |
| Mistral KO / quota | Fallback hiérarchique : Anthropic → Ollama → template Jinja2 | la fiche se génère via template (perd le critère « LLM ») |
| 2 agents IA = travail non humain → risque d'erreur silencieuse | Tests pytest sur les règles déterministes ; revue croisée fin de phase ; chaque chiffre du rapport est traçable à une cellule du code | — |
| Démo live qui plante | HTML statique + screenshots dans `outputs/` ; vidéo de secours | screenshots |

---

## 12. Pitch (5 min)

Problème (30 s) → Démarche & périmètre (1 min, en citant le cours ML pour appuyer la rigueur) → **Démo live** (2 min : carte folium → clic sur un navire incohérent → popup → on passe une signature dans le pipeline → bon MMSI à 0.9x + fiche de renseignement générée par Mistral + alerte PDF) → Résultats chiffrés (1 min : rappel vs vérité terrain, AUC, identification ≥ 8/10, sources OSINT mobilisées) → Limites & extensions (30 s : SDR + RF fingerprinting matériel, modèles trajectoires type GeoTrackNet, croisement avec le Sujet 5).

---

## 13. Validation du cahier des charges

✅ Couvre les 14 questions de la Généralisation · ✅ choix d'algorithmes justifiés par le cours Bogroff · ✅ pour chaque algo : pourquoi + paramétrage + métrique · ✅ enrichissement OSINT réel gratuit · ✅ pipeline reproductible + alertes auto · ✅ métriques d'acceptation chiffrées + cadre business asymétrique (cours 4 §8) · ✅ risques + parades + minimum viable · ✅ 0 € · ✅ aligné sur les 6 critères de notation officiels et la consigne générale du hackathon.
