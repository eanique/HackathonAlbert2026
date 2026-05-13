# Plan d'exécution — Sujet 3 (mode 2 agents IA, 3 phases)

> Daté du 2026-05-11. Travail exécuté par **deux streams Claude parallèles** (Agent A = Données/Détection, Agent B = Pipeline/Renseignement), orchestrés via le tool `Agent`. Les « 3 jours » deviennent **3 phases logiques** ; chaque phase = un point de synchronisation + revue croisée.
>
> **Règle de coupe en cas de retard** : on coupe d'abord (1) l'anomalie tabulaire LLM, puis (2) le sous-score zone-dépendant, puis (3) AISStream temps réel. On **ne coupe pas** Q1→Q14, ni la carte folium, ni la validation precision/recall/AUC.

---

## Phase 1 — Données, base de profils radio, accès OSINT

### Préalable (commun aux 2 agents, ~20 min)
- Relire `CLAUDE.md` (énoncé, données, pièges) + `cadrage.md` + `hypotheses.md`.
- Initialiser l'environnement : `uv venv && uv pip install -r requirements.txt`.
- Copier `SujetsHackathon2026/Sujet3/Généralisation/*.csv` dans `sujet3/data/raw/`.
- Créer `.env` (vide ou avec les clés déjà disponibles).

### Agent A — Données & Q1–Q3
1. **`src/load.py`** : lecture des 4 CSV, normalisation (mmsi/imo str, timestamps UTC, dédup, manquants loggés, tri AIS), validation MMSI (9 chiffres + MID UIT), dictionnaire de données généré.
2. **Mise en jambe** (`reponses_mise_en_jambe.py`) : Q1→Q12 sur les `*_small.csv` (~1 h, échauffement et premier livrable).
3. **`src/profiles.py`** : Q1 — agrégation par navire → `data/processed/ship_radio_profiles.csv` ; top 5 fréquence ; Q2 — `pulse_pattern` uniques + combinaison `pulse_pattern × modulation × bande de frequency`.
4. **`src/cluster.py`** : Q3 — `StandardScaler` + `KMeans(K=5, n_init=10, init='k-means++')` ; rapporter **WCSS, Silhouette, Elbow plot sur K ∈ [2,10]** ; PCA 2D ; viz `frequency × power` + biplot.

### Agent B — Infra, OSINT, LLM, viz
1. Créer les comptes/clés OSINT gratuits : **Equasis** (inscription), **OpenSanctions** (API key), **AISStream.io** (API key), **GFW** (API token), choisir 1–2 **flux RSS** maritime. → variables d'env dans `.env`, listées dans `README.md` et `.env.example`.
2. **`src/config.py`** : chargement `.env` (python-dotenv), constantes (chemins, seuils, listes MID UIT par pays).
3. **`src/llm.py`** : interface unique `generate_intel_note(...)` avec backend Mistral (`mistralai` SDK), fallbacks Anthropic / Ollama / template Jinja2 via variable d'env `LLM_BACKEND`. Cache des réponses dans `data/cache.sqlite` (clé = hash du prompt).
4. **`src/osint_enrich.py`** (stubs avec interfaces fixées, implémentation finalisée en Phase 2) : `lookup_equasis(mmsi) → dict`, `validate_mmsi_uit(mmsi) → dict`, `query_opensanctions(name, imo, mmsi) → list`, `fetch_rss(url) → list[item]`.
5. **`src/maps.py`** : helpers folium (fond de carte, couches `ships/anomalies/zones/alerts`, popup standardisé).
6. **`SOURCES.md`** + **`README.md`** initiaux (lancement, variables d'env, sources créditées avec URL et licence).

### Sync fin de Phase 1
- ✅ `ship_radio_profiles.csv` produit.
- ✅ Clusters K=5 visualisés (`outputs/clusters_kmeans.png` + biplot PCA).
- ✅ Mise en jambe livrée.
- ✅ Au moins 1 source OSINT répond (test live).
- ✅ Backend LLM choisi et `generate_intel_note("hello")` fonctionne.
- ✅ Squelette du repo en place, `make all` ne plante pas (sections vides ok).
- ✅ Revue croisée : A vérifie le code OSINT/LLM/maps de B, B vérifie load/profiles/cluster de A.

---

## Phase 2 — Détection d'anomalies, score de suspicion, identification (le cœur noté)

### Agent A — Détecteurs « cours » (Q4–Q11)
1. **`src/anomalies.py`** :
   - **Q4** : pour chaque `flag` ≥ 30 navires → `EllipticEnvelope (MCD)` sur (frequency, bandwidth, power) des non-suspects ; flaguer le reste ; `confidence` à partir de la distance robuste ; **top 10**. *[Cours 4 §4]*
   - **Q5** : `historical_names.str.split(',').len() > 2` ; join `anomalies[type=='Name Change']`.
   - **Q6** : signatures orphelines + k-NN d'attribution (k=5, standardisé).
   - **Q7** : plages contiguës `ais_active==False`, durée > 24h, exclusion `Moored/At Anchor`.
   - **Q8** : `merge_asof` + `geopy` ; navires avec écart > 1km ; couche folium `lines` (AIS↔radio).
2. **P3 (Q9–Q11)** : évolution temporelle d'un MMSI + détection de ruptures `|Δfreq|>1MHz` (+ `ruptures` bonus) ; stats par pavillon ; corrélation `pearsonr(speed, frequency)` (hypothèse a priori : non significatif).
3. **Évaluation rigoureuse** vs `anomalies_large.csv` : matrice de confusion **par `type`** ; precision/recall/F1 par détecteur ; tableau récap.

### Agent B — Score global, règles, pipeline
1. **`src/spoofing_rules.py`** : règles MMSI dupliqué, timestamp incohérent, intervalle anormal, vitesse implicite (geodesic / Δt) — toutes déterministes, chacune avec un `confidence`.
2. **`src/anomaly_score.py`** :
   - **Isolation Forest** sur `radio_signatures_large` enrichi (one-hot pulse_pattern + modulation), `contamination=0.02`, **sans scaling**. *[Cours 4 §6]*
   - **LOF** standardisé, `n_neighbors=20`, `contamination=0.02`. *[Cours 4 §7]*
   - **Sous-score zone-dépendant** : grille 1°×1°, KDE par cellule sur (cap, vitesse, statut), `−log p`. *(GeoTrackNet light)*
   - **Anomalie LLM tabulaire** : `score_llm(row)` via Mistral, prompt few-shot 20 exemples normaux. *[arXiv 2406.16308]*
   - **Score global multi-facteurs** : somme pondérée × confidence ; **tuning des poids** pour maximiser AUC vs `is_suspicious` (split 70/30) ; courbe ROC + precision@k.
3. **`src/identify.py`** (Q12) : k-NN sur `ship_radio_profiles` + **One-Class SVM** *novelty* `(kernel='rbf', nu=0.05)` entraîné sur le « normal propre ». *[Cours 4 §5]* Retourne `dict` structuré (mmsi, confidence, suspect, raisons).
4. **`src/osint_enrich.py`** (implémentation complète) + **`src/intel_report.py`** : assemble OSINT + score → **fiche de renseignement** rédigée par Mistral + **PDF** (`reportlab`).
5. **`src/maps.py`** : carte folium finalisée (couches navires / anomalies / écarts AIS↔radio / alertes).

### Sync fin de Phase 2
- ✅ Tous les détecteurs Q4–Q8 codés et évalués (precision/recall/F1 par type).
- ✅ Score global validé : AUC vs `is_suspicious` rapporté, poids ajustés.
- ✅ Pipeline `identify(signature) → dict` fonctionnel.
- ✅ Fiche PDF générée pour ≥ 1 navire suspect (test live).
- ✅ Carte folium ouvre dans un navigateur.
- ✅ Revue croisée + commit propre.

---

## Phase 3 — Validation, automatisation, rapport, pitch

### Matin — Agent B
1. **Q13** : `identify` sur 10 signatures aléatoires × 3 graines → taux moyen ± écart, matrice de confusion.
2. **Q14** : POC mise à jour incrémentale (moyenne glissante) + ingestion **RSS** (`feedparser`) qui set `is_suspicious`.
3. **Assemblage `reponses_generalisation.py`** : une section par question Q1→Q14 (commentaire `### Q1`, `### Q2`, …), chacune affiche/sauvegarde son résultat + note d'analyse critique en commentaire. Vérifier que `python reponses_generalisation.py` tourne **de bout en bout** depuis un env neuf. `Makefile` (`make all`).
4. **`src/realtime_ais.py`** (optionnel) : client AISStream pour démo « live ».

### Matin — Agent A
1. Finaliser les visualisations dans `outputs/` : `clusters_kmeans.png`, `pca_biplot.png`, `carte_anomalies.html`, `temporal_mmsi.png`, `flag_stats.png`, `roc_curve.png`, `confusion_per_type.png`, `metrics_table.csv`, `alerte_<mmsi>.pdf`.
2. Rédiger **`rapport_generalisation.md`** (plan du skill `livrable-hackathon`, aligné sur les 6 critères) :
   1. **Contexte & périmètre** (avec décisions verrouillées et OUT justifiés).
   2. **Données & nettoyage** *(critère 2)* — sources fournies + OSINT, URLs, licences, schémas, étapes.
   3. **Démarche & méthodes** *(critère 1)* — Q1→Q14, chaque algo cité (cours Bogroff + arXiv) avec le *pourquoi*.
   4. **Résultats** *(critère 3)* — tous les chiffres : top-N, nb d'anomalies, AUC, precision/recall par type, taux d'ID, ROC, viz.
   5. **Automatisation & API** *(critère 4)* — pipeline, OSINT, alertes, perfs.
   6. **Limites & pistes** — synthétique vs réel, SDR/RF fingerprinting, modèles trajectoires, …
   7. **Annexes** — dictionnaire de données, hypothèses (`hypotheses.md`), liens vers `outputs/`.

### Après-midi — ensemble (synchro)
- **Geler** les livrables : copie dans `ProjetsEtudiantsHackathon2026/<NomGroupe-Sujet3>/` (EQUIPE.md, reponses_*.py, rapport_*.md, README.md, requirements.txt, data/processed/, outputs/). Dérouler la checklist du skill **`livrable-hackathon`**.
- **Vérifications finales** : (i) `pip install -r requirements.txt` depuis zéro + `python reponses_generalisation.py` produit tout ; (ii) aucun fichier > 50 Mo committé ; (iii) aucune clé / token dans le code ; (iv) toutes les questions traitées ou listées « hors périmètre » dans le rapport ; (v) sources OSINT créditées (nom + URL + licence) ; (vi) `ruff check` propre ; (vii) `pytest` passe.
- **Slides + répétition pitch** (4–5 slides + démo) : Problème → Démarche & périmètre → **Démo live** (carte → popup → `identify()` → PDF) → Résultats chiffrés → Limites & extensions. **Fallback** : screenshots + vidéo prêts.
- **Sync avec l'équipe Sujet 5** sur le récit commun (« rendre visible un navire qui veut rester invisible ») et l'ordre de passage.

---

## Récapitulatif (vue d'avion)

| Phase | Agent A — Données/Détection | Agent B — Pipeline/Renseignement |
|---|---|---|
| **1** | `load.py`, Mise en jambe, `profiles.py` (Q1–Q2), `cluster.py` (Q3) | OSINT credentials + `config.py` + `llm.py` (Mistral) + `osint_enrich.py` stubs + `maps.py` helpers + SOURCES.md/README |
| **2** | `anomalies.py` (Q4 EllipticEnvelope MCD, Q5–Q8), P3 (Q9–Q11), métriques par type | `spoofing_rules.py`, `anomaly_score.py` (iForest + LOF + zone + LLM + score global tuné sur AUC), `identify.py` (k-NN + OCSVM novelty), OSINT complet, `intel_report.py` (PDF), carte folium finale |
| **3** | Visualisations finales, `rapport_generalisation.md` | Q13 validation, Q14 RSS, assemblage `reponses_generalisation.py`, `Makefile`, `realtime_ais.py` (option) |
| **3 PM** | **ENSEMBLE** : gel des livrables, checklist `livrable-hackathon`, slides, répétition pitch, sync avec équipe S5 | |

---

## Conventions de versionnage (interne aux agents)

- Un commit par module finalisé, message au format `feat(QX): …` ou `chore: …`.
- Pas de push tant que `ruff check` n'est pas propre et qu'au moins 1 test pytest sur le module passe.
- Le `rapport_generalisation.md` est versionné en continu (chaque chiffre du rapport doit être traçable à une cellule du code).
