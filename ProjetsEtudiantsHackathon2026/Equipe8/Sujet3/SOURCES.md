# Sources de données — Sujet 3 (BDD-MinArm)

> Toutes les sources sont **gratuites**. Les clés/identifiants se mettent dans `.env` (gitignored) ; le pipeline tourne sans elles (fallbacks). Conformément à la consigne, chaque source est créditée (nom · URL · licence · usage). La consigne du hackathon survalorise explicitement « la découverte de sources pertinentes en sources libres ».

## 1. Données fournies (le socle noté — synthétiques)

| Fichier | Source | Licence | Usage |
|---|---|---|---|
| `ships_large.csv`, `radio_signatures_large.csv`, `ais_data_large.csv`, `anomalies_large.csv` (+ `*_small.csv`, dictionnaires JSON) | Repo du hackathon `eanique/HackathonAlbert2026`, dossier `SujetsHackathon2026/Sujet3/` | usage hackathon | toutes les questions Q1→Q14 ; `anomalies_large.csv` = vérité terrain pour l'évaluation |

## 2. Sources OSINT d'enrichissement (Levier 2)

| Source | URL | Accès | Licence | Ce que ça donne | Où dans le code |
|---|---|---|---|---|---|
| **Base MMSI UIT / MARS** | `itu.int/en/ITU-R/terrestrial/fmd/Pages/mid.aspx` | public, offline | UIT-R M.1371 (public) | MID (3 premiers chiffres du MMSI) → pays attribué ; validité d'un MMSI | `src/config.py` (table `UIT_MID`), `src/osint_enrich.lookup_mmsi` |
| **OpenSanctions** | `opensanctions.org/api` | clé API gratuite | CC-BY 4.0 | navire / armateur / société sous sanctions (OFAC SDN, UE consolidated…) | `src/osint_enrich.query_opensanctions` (`OPENSANCTIONS_API_KEY`) |
| **Equasis** | `equasis.org` | compte gratuit | données publiques, usage avec compte | historique des noms et des pavillons, inspections PSC, armateur/manager, particularités du navire | `src/osint_enrich.lookup_equasis` (`EQUASIS_USERNAME` / `EQUASIS_PASSWORD`) — scraping post-login à finaliser |
| **Flux RSS maritime** — gCaptain | `gcaptain.com/feed/` | gratuit | éditeur (lien + extrait) | actualité navires / incidents → ingestion OSINT (Q14) | `src/osint_enrich.fetch_rss` |
| **Flux RSS maritime** — The Maritime Executive | `maritime-executive.com/articles.rss` | gratuit | éditeur (lien + extrait) | idem | `src/osint_enrich.fetch_rss` |
| **Global Fishing Watch API** | `globalfishingwatch.org/our-apis` | token API gratuit (inscription) | CC-BY 4.0 (GFW) | événements **pré-calculés** sur l'AIS mondial : AIS gap events, encounters (transbordements), loitering | `src/osint_enrich.query_gfw_events` (`GFW_API_TOKEN`) |
| **SHOM — épaves** (via MCP `shom-wrecks`) | `services.data.shom.fr/INSPIRE/wfs` (repo `dorian-erkens/mcp-shom-wrecks`) | public (WFS) | Licence Ouverte Etalab | ≈ 4 796 épaves françaises (position, brassiage, type, circonstances du naufrage) → écarter les faux positifs de position | MCP `shom-wrecks` (outils `search_wreck_by_name`, `get_nearby_wrecks`, `search_wrecks_bbox`, `get_wreck_details`) |

## 3. AIS réel (plan B au SDR — temps réel & historique)

| Source | URL | Accès | Licence | Usage |
|---|---|---|---|---|
| **AISStream.io** | `aisstream.io` | clé API gratuite | gratuit (CGU AISStream) | flux AIS **temps réel** par WebSocket (bounding box) → démo « live » : carte qui s'actualise + détecteurs sur du vrai AIS | `src/realtime_ais.stream` (`AISSTREAM_API_KEY`) |
| **Danish Maritime Authority** | `web.ais.dk/aisdata/` | gratuit, sans clé | données publiques danoises | dumps AIS historiques journaliers (CSV) — rejouer les détecteurs sur de l'AIS réel | (non câblé — ajout trivial si besoin) |
| **NOAA MarineCadastre** | `marinecadastre.gov/ais/` | gratuit | domaine public US | AIS historique US par année/zone | (non câblé) |
| **IMO GISIS** | `gisis.imo.org` | compte gratuit | public | Ship & Company Particulars (numéros IMO) | (non câblé — complète Equasis) |
| **EU Fleet Register** | `webgate.ec.europa.eu/fleet-europa` | gratuit | données ouvertes UE | registre communautaire des navires de pêche | (non câblé — si on traite des chalutiers) |

## 4. LLM (Levier 2 — fiche de renseignement + détecteur d'anomalie tabulaire)

| Service | URL | Accès | Pourquoi | Où dans le code |
|---|---|---|---|---|
| **Mistral** | `console.mistral.ai` | clé API | choix **souveraineté numérique française** — cohérent avec un livrable Ministère des Armées ; qualité suffisante pour rédiger une fiche de renseignement et faire de la détection d'anomalie tabulaire zero-shot (arXiv 2406.16308) | `src/llm.py` (`MISTRAL_API_KEY`, `MISTRAL_MODEL`), `src/intel_report.py` |
| Anthropic / Ollama / Jinja2 | — | fallbacks | si Mistral indisponible : `LLM_BACKEND=anthropic\|ollama\|template` ; le pipeline ne casse jamais | `src/llm.py` |

## 5. Référentiels méthodologiques (cités dans le rapport)

| Référence | Type | Usage |
|---|---|---|
| Cours d'**Alexis Bogroff**, Albert School — ML III « Unsupervised Learning Problems », 4 sessions (`cours ml/FICHE_REVISION_DATA.md`) | cours | référentiel des algos : K-Means/GMM, hiérarchique/DBSCAN, PCA/t-SNE/UMAP, détection d'anomalies (EllipticEnvelope/MCD, One-Class SVM, Isolation Forest, LOF) ; concepts (inductive bias, scaling, masking effect, outlier vs novelty, coût asymétrique) |
| **GeoTrackNet** — Nguyen et al. — arXiv [1912.00682](https://arxiv.org/abs/1912.00682) — code `CIA-Oceanix/GeoTrackNet` (fork `BDD-MinArm`) | papier + code | inspiration du sous-score zone-dépendant (anomalie d'AIS dépendante de la zone) |
| **TrAISformer** — arXiv [2109.03958](https://arxiv.org/abs/2109.03958) — code `CIA-Oceanix/TrAISformer` (fork `BDD-MinArm`) | papier + code | cité comme extension (prédiction de trajectoire → écart attendu) |
| **pipe-gaps** — `GlobalFishingWatch/pipe-gaps` (fork `BDD-MinArm`) | code | algo ré-implémenté pour Q7 (détection de « AIS gap events ») |
| **SeaSpoofFinder** / GNSS spoofing detection — arXiv [2602.16257](https://arxiv.org/abs/2602.16257), [2603.11055](https://arxiv.org/abs/2603.11055) | papiers | inspiration des règles de spoofing (MMSI dupliqués, timestamps, intervalles) |
| **Anomaly Detection of Tabular Data Using LLMs** — Li et al. — arXiv [2406.16308](https://arxiv.org/abs/2406.16308) | papier | terme « anomalie tabulaire par LLM zero-shot » dans le score |
| **RF Fingerprinting of Bluetooth devices** — arXiv [2402.06250](https://arxiv.org/abs/2402.06250) ; **FS-SEI** (`BeechburgPieStar/FS-SEI`) ; **DeepCRF** — arXiv [2411.06925](https://arxiv.org/abs/2411.06925) | papiers + code | cités comme méthodologie de référence pour l'extension « vrai RF fingerprinting sur échantillons IQ via SDR » |

## 6. Outils & librairies open source

`pandas`, `numpy`, `scipy`, `scikit-learn`, `umap-learn`, `ruptures`, `geopy`, `shapely`, `folium`, `matplotlib`, `plotly`, `networkx`, `pyvis`, `requests`, `feedparser`, `websockets`, `joblib`, `python-dotenv`, `mistralai`, `jinja2`, `reportlab`, `markdown`, `xgboost`, `pyod`, `ruff`, `pytest` — toutes installées via `uv` (Python 3.12). Voir `requirements.txt`.

---

*Mise à jour : ajouter ici toute nouvelle source utilisée. Ne jamais committer de clé API (les valeurs vont dans `.env`, qui est dans `.gitignore`).*
