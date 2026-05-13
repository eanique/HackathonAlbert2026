"""Sujet 3 — Identification navale par signature radio.

Modules :
    config        — chargement .env, constantes (chemins, seuils, listes MID UIT).
    load          — lecture + normalisation des CSV fournis.
    profiles      — agrégation des signatures → profils navire (Q1).
    cluster       — K-Means K=5 + Silhouette + PCA (Q3).
    anomalies     — détecteurs Q4 (EllipticEnvelope/MCD par flag), Q5–Q8.
    spoofing_rules— règles MMSI/timestamp/intervalle (arXiv 2603.11055).
    anomaly_score — Isolation Forest + LOF + sous-score zone + score global.
    identify      — pipeline d'identification passive (k-NN + OCSVM novelty, Q12).
    osint_enrich  — Equasis / UIT MMSI / OpenSanctions / RSS / GFW.
    intel_report  — fiche de renseignement LLM (Mistral) + PDF.
    realtime_ais  — client AISStream.io (optionnel, démo live).
    maps          — helpers folium.
    llm           — backend LLM configurable (Mistral / Anthropic / Ollama / template).
"""

__version__ = "0.1.0"
