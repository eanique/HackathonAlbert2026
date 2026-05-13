"""Interface LLM unifiée — Mistral REST + fallback template Jinja2.

Le SDK officiel `mistralai` est en quarantaine sur PyPI (mai 2026) → on appelle
l'API REST directement avec `requests`. L'endpoint est compatible OpenAI
(POST /v1/chat/completions, body `{model, messages}`), choix par défaut Mistral
pour la souveraineté française (aligné Sujet 3, cf. cadrage §6.5).

Signature `generate_intel_note(facts: dict) -> str` identique à celle prévue
pour `sujet3/src/llm.py` → mutualisable. Cache SQLite : clé = sha256(prompt+model)
pour éviter de re-payer / re-attendre une fiche déjà générée.

Fallback : si `LLM_BACKEND=template` ou si l'API plante / pas de clé, on rend
une fiche Jinja2 (déterministe, hors-ligne). C'est ce qui tourne par défaut
sans `.env` rempli — utile pour les tests et la démo offline.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from typing import Any

import requests
from jinja2 import Template

from . import config as C

_MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
_DEFAULT_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
_TIMEOUT_S = 30


# ----------------------------------------------------------------------
# Cache SQLite (évite de re-générer la même fiche)

def _cache_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(C.CACHE_DB)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS llm_cache "
        "(key TEXT PRIMARY KEY, response TEXT, created_at REAL)"
    )
    return conn


def _cache_get(key: str) -> str | None:
    with _cache_conn() as conn:
        row = conn.execute(
            "SELECT response FROM llm_cache WHERE key = ?", (key,)
        ).fetchone()
    return row[0] if row else None


def _cache_put(key: str, response: str) -> None:
    with _cache_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO llm_cache(key, response, created_at) VALUES (?, ?, ?)",
            (key, response, time.time()),
        )


def _cache_key(prompt: str, model: str) -> str:
    return hashlib.sha256(f"{model}::{prompt}".encode()).hexdigest()


# ----------------------------------------------------------------------
# Backend 1 — Mistral REST (compatible OpenAI)

def _call_mistral(prompt: str, model: str, max_tokens: int = 800) -> str:
    """POST /v1/chat/completions ; lève si pas de clé ou erreur HTTP."""
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        raise RuntimeError("MISTRAL_API_KEY absent du .env")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Tu es analyste renseignement maritime pour le Ministère des "
                    "Armées (DGA/DRM). Tu écris des fiches courtes, factuelles, "
                    "en français, sans embellissement. Ne fabrique aucun fait : "
                    "appuie-toi UNIQUEMENT sur les données fournies."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    r = requests.post(_MISTRAL_URL, json=payload, headers=headers, timeout=_TIMEOUT_S)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


# ----------------------------------------------------------------------
# Backend 2 — Template Jinja2 (toujours dispo, déterministe, hors-ligne)

_TEMPLATE_INTEL_NOTE = Template(
    """## Fiche de renseignement — Détection navale

**Identifiant** : {{ detection_id }}
**Date scène** : {{ scene_datetime }}
**Source imagerie** : {{ source }}

### Géolocalisation
- Coordonnées : {{ "%.4f"|format(lat) }}, {{ "%.4f"|format(lon) }}
- Port le plus proche : **{{ nearest_port or "(inconnu)" }}**
{% if nearest_mil_zone -%}
- Zone militaire la plus proche : **{{ nearest_mil_zone }}**
  ({{ "%.1f"|format(nearest_mil_zone_dist_km) }} km, risk={{ nearest_mil_zone_risk }})
{%- endif %}

### Détection
- Classe estimée : **{{ category or "?" }}**{% if is_military %} (militaire){% endif %}
- Confiance : {{ "%.2f"|format(confidence) if confidence else "?" }}
- Bbox (px) : {{ bbox_px or "?" }}

### Contexte environnemental
{% if weather -%}
- Météo : {{ weather.description or "?" }}, {{ weather.temp_c or "?" }} °C, vent {{ weather.wind_ms or "?" }} m/s
{%- else -%}
- Météo : non récupérée (clé OPENWEATHER_API_KEY manquante)
{%- endif %}

### Croisement AIS
{% if is_dark -%}
- ⚠️ **NAVIRE SOMBRE** : aucun MMSI AIS détecté dans la fenêtre
  ({{ ais_window_min }} min, {{ ais_window_km }} km). Un satellite voit ce que
  l'AIS cache — alerte forte si confirmé par un second passage.
{%- elif nearest_mmsi -%}
- MMSI le plus proche : {{ nearest_mmsi }} à {{ nearest_mmsi_dist_km }} km
{%- else -%}
- AIS non interrogé pour cette détection
{%- endif %}

### Infrastructure critique sous-marine
{% set _infra_d = nearest_infra_dist_km if (nearest_infra_dist_km is defined) else none -%}
{% set _infra_thr = (infra_alert_threshold_km if (infra_alert_threshold_km is defined) else none) or 2.0 -%}
{% set _infra_kind = nearest_infra_kind if (nearest_infra_kind is defined) else none -%}
{% set _infra_name = nearest_infra_name if (nearest_infra_name is defined) else none -%}
{% if _infra_d is not none -%}
{% if _infra_d < _infra_thr -%}
- 🚨 **PROXIMITÉ INFRA CRITIQUE** : {{ _infra_kind }} «{{ _infra_name }}»
  à **{{ "%.2f"|format(_infra_d) }} km** (seuil {{ _infra_thr }} km).
  Précédents : Nord Stream (2022), Eagle S × EstLink (déc. 2024). Un cargo qui
  approche un câble ou un gazoduc à cette distance justifie une vérification
  systématique (cap, vitesse, pavillon).
{%- else -%}
- {{ _infra_kind }} le plus proche : «{{ _infra_name }}» à
  {{ "%.2f"|format(_infra_d) }} km (hors seuil d'alerte {{ _infra_thr }} km).
{%- endif %}
{%- else -%}
- Aucune infrastructure critique sous-marine OSM dans le rayon de recherche.
{%- endif %}

### Analyse
{% if alert -%}
🚨 **ALERTE** : {{ alert_reason }}
{%- else -%}
Aucune alerte automatique sur cette détection (critères non réunis).
{%- endif %}

---
*Fiche générée par le pipeline Sujet 5 (BDD-MinArm) — toutes les données sont
issues de sources OSINT (cf. SOURCES.md). Ce document ne contient aucune
spéculation : tout fait non vérifié est marqué (?).*
"""
)


def _call_template(facts: dict[str, Any]) -> str:
    return _TEMPLATE_INTEL_NOTE.render(**facts)


# ----------------------------------------------------------------------
# Façade publique

def generate_intel_note(
    facts: dict[str, Any],
    backend: str | None = None,
    model: str | None = None,
    use_cache: bool = True,
) -> dict:
    """Rédige une fiche markdown à partir d'un dict de faits.

    Args:
        facts: dictionnaire de faits structurés sur la détection (cf. signature
            attendue par le template). Pas de spéculation : chaque clé est
            soit présente, soit explicitement None.
        backend: 'mistral', 'template' (par défaut : `LLM_BACKEND` env, sinon
            'mistral' → bascule auto sur 'template' si pas de clé).
        model: override du modèle Mistral.
        use_cache: lit/écrit le cache SQLite (désactivable pour tests).

    Returns:
        dict {markdown, backend_used, cached, latency_s, model}
    """
    backend = backend or os.getenv("LLM_BACKEND", "mistral")
    model = model or _DEFAULT_MODEL
    t0 = time.perf_counter()

    if backend == "template" or not os.getenv("MISTRAL_API_KEY"):
        # Le template est toujours dispo : pas de cache nécessaire (déterministe)
        return {
            "markdown": _call_template(facts),
            "backend_used": "template",
            "cached": False,
            "latency_s": round(time.perf_counter() - t0, 3),
            "model": "jinja2",
        }

    # Backend Mistral : on demande au LLM de RÉCRIRE la fiche template avec une
    # analyse plus fine — le template fournit le squelette factuel, Mistral
    # ajoute l'analyse contextuelle. On ne lui laisse PAS inventer de faits.
    skeleton = _call_template(facts)
    prompt = (
        "Voici la fiche brute (faits structurés extraits du pipeline). "
        "Rééécris-la en français, **sans ajouter aucun fait nouveau** : tu peux "
        "reformuler, regrouper, prioriser, ajouter une section « Analyse » de "
        "3-5 lignes basée UNIQUEMENT sur les éléments ci-dessous. Garde la "
        "structure markdown (titres, listes).\n\n"
        f"---\n{skeleton}\n---"
    )

    key = _cache_key(prompt, model)
    if use_cache:
        cached = _cache_get(key)
        if cached:
            return {
                "markdown": cached,
                "backend_used": "mistral",
                "cached": True,
                "latency_s": round(time.perf_counter() - t0, 3),
                "model": model,
            }

    try:
        md = _call_mistral(prompt, model)
        if use_cache:
            _cache_put(key, md)
        return {
            "markdown": md,
            "backend_used": "mistral",
            "cached": False,
            "latency_s": round(time.perf_counter() - t0, 3),
            "model": model,
        }
    except (requests.RequestException, RuntimeError, KeyError) as e:
        # Bascule sur template (jamais d'erreur fatale pour la démo)
        return {
            "markdown": _call_template(facts),
            "backend_used": f"template (fallback: {type(e).__name__})",
            "cached": False,
            "latency_s": round(time.perf_counter() - t0, 3),
            "model": "jinja2",
            "error": str(e),
        }


# ----------------------------------------------------------------------
# Helper : auto-fill des champs manquants depuis les sources OSINT

def enrich_facts(
    detection: dict,
    *,
    osint=None,
    ais_status: dict | None = None,
) -> dict:
    """Enrichit un dict de détection brut avec port + météo + statut AIS.

    Si `osint` est None, on importe `src.osint_enrich` paresseusement (évite
    une dépendance circulaire avec `intel_report`).
    """
    if osint is None:
        from . import osint_enrich as osint  # noqa: N813

    facts = dict(detection)
    lat, lon = detection.get("lat"), detection.get("lon")
    if lat is not None and lon is not None:
        try:
            facts["nearest_port"] = osint.nearest_port_name(lat, lon)
        except Exception:  # noqa: BLE001
            facts["nearest_port"] = None
        try:
            facts["weather"] = osint.weather_at(lat, lon, detection.get("timestamp"))
        except Exception:  # noqa: BLE001
            facts["weather"] = None

    facts.setdefault("ais_window_min", C.AIS_TIME_WINDOW_MIN)
    facts.setdefault("ais_window_km", C.AIS_DIST_WINDOW_KM)
    if ais_status:
        facts.update(ais_status)
    return facts


if __name__ == "__main__":
    # Démo locale : génère une fiche template (toujours dispo, sans clé).
    demo = {
        "detection_id": "DEMO-001",
        "scene_datetime": "2026-05-12T10:30:00Z",
        "source": "Sentinel-1 RTC (demo)",
        "lat": 43.108,
        "lon": 5.901,
        "nearest_port": "Toulon",
        "nearest_mil_zone": "Base navale de Toulon",
        "nearest_mil_zone_dist_km": 0.8,
        "nearest_mil_zone_risk": "Critical",
        "category": "Frégate (estimée)",
        "is_military": True,
        "confidence": 0.78,
        "bbox_px": [1024, 512, 1080, 568],
        "weather": {"description": "ciel dégagé", "temp_c": 18, "wind_ms": 4},
        "is_dark": True,
        "alert": True,
        "alert_reason": "Militaire en zone Critical sans MMSI AIS.",
    }
    out = generate_intel_note(demo, backend="template")
    print(out["markdown"])
    print(f"\n[backend={out['backend_used']}, latency={out['latency_s']}s]")
