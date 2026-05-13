"""Backend LLM configurable — Mistral en priorité (souveraineté FR pour Minarm).

Interface unique :

    generate(prompt: str, system: str | None = None, temperature: float = 0.2,
             max_tokens: int = 800, cache_key: str | None = None) -> str

Backends supportés via la variable d'env LLM_BACKEND :
    - "mistral"   → mistralai SDK (par défaut)
    - "anthropic" → anthropic SDK (fallback)
    - "ollama"    → modèle local (Llama 3 / Qwen) via API REST (fallback)
    - "template"  → pas de LLM, renvoie le prompt formaté (fallback ultime,
                    utilisé par intel_report.py pour générer la fiche via Jinja2)

Cache : si `cache_key` est fourni, la réponse est stockée dans data/cache.sqlite
        (table llm_responses) et rejouée à l'identique au prochain appel
        → la démo tourne hors-ligne et de façon déterministe.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Optional

from .config import (
    CACHE_DB,
    LLM_BACKEND,
    MISTRAL_API_KEY,
    MISTRAL_MODEL,
)

# ----------------------------------------------------------------------------
# Cache SQLite
# ----------------------------------------------------------------------------


def _ensure_cache() -> None:
    with sqlite3.connect(CACHE_DB) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS llm_responses (
                   key TEXT PRIMARY KEY,
                   backend TEXT NOT NULL,
                   model TEXT,
                   prompt TEXT NOT NULL,
                   response TEXT NOT NULL,
                   ts TEXT DEFAULT CURRENT_TIMESTAMP
               )"""
        )


def _cache_get(key: str) -> str | None:
    _ensure_cache()
    with sqlite3.connect(CACHE_DB) as conn:
        row = conn.execute(
            "SELECT response FROM llm_responses WHERE key = ?", (key,)
        ).fetchone()
    return row[0] if row else None


def _cache_set(key: str, backend: str, model: str, prompt: str, response: str) -> None:
    _ensure_cache()
    with sqlite3.connect(CACHE_DB) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO llm_responses (key, backend, model, prompt, response) VALUES (?,?,?,?,?)",
            (key, backend, model, prompt, response),
        )


def _hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:32]


# ----------------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------------


def _import_mistral_client():
    """Importe la classe `Mistral` quel que soit le layout du paquet `mistralai`.

    - SDK officiel v1.x : `from mistralai import Mistral`
    - layout v2.x (Speakeasy) : `from mistralai.client import Mistral`
    - ancien SDK : `from mistralai.client import MistralClient`
    """
    try:
        from mistralai import Mistral  # type: ignore
        return Mistral
    except ImportError:
        pass
    try:
        from mistralai.client import Mistral  # type: ignore
        return Mistral
    except ImportError:
        pass
    from mistralai.client import MistralClient  # type: ignore  # noqa
    return MistralClient


def _call_mistral(prompt: str, system: Optional[str], temperature: float,
                  max_tokens: int) -> str:
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY manquante — voir .env.example.")
    Mistral = _import_mistral_client()

    client = Mistral(api_key=MISTRAL_API_KEY)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    # API unifiée v1.x/v2.x : client.chat.complete(...)
    resp = client.chat.complete(
        model=MISTRAL_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def _call_anthropic(prompt: str, system: Optional[str], temperature: float,
                    max_tokens: int) -> str:
    import os

    import anthropic  # type: ignore

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=max_tokens,
        temperature=temperature,
        system=system or "",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text  # type: ignore[attr-defined]


def _call_ollama(prompt: str, system: Optional[str], temperature: float,
                 max_tokens: int) -> str:
    import os

    import requests

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
    payload = {
        "model": model,
        "prompt": (f"<|system|>\n{system}\n" if system else "") + prompt,
        "options": {"temperature": temperature, "num_predict": max_tokens},
        "stream": False,
    }
    r = requests.post(f"{host}/api/generate", json=payload, timeout=120)
    r.raise_for_status()
    return r.json().get("response", "")


def _call_template(prompt: str, system: Optional[str], temperature: float,
                   max_tokens: int) -> str:
    """Fallback ultime : pas d'appel LLM, juste un écho structuré.

    intel_report.py prend le relais pour rédiger via Jinja2 quand ce backend
    est actif (LLM_BACKEND=template). Ne perd pas le pipeline.
    """
    return (
        "[LLM_BACKEND=template — aucune génération automatique]\n\n"
        f"{system or ''}\n\n{prompt}"
    )


# ----------------------------------------------------------------------------
# Interface publique
# ----------------------------------------------------------------------------


def generate(prompt: str, system: Optional[str] = None, *, temperature: float = 0.2,
             max_tokens: int = 800, cache_key: Optional[str] = None) -> str:
    """Génère une réponse via le backend LLM configuré.

    - `cache_key` : si fourni, déduplique les appels identiques (idéal pour la
      démo et pour la rigueur expérimentale — pas de variance d'un run à l'autre).
    - Erreurs réseau / clé manquante / quota : on retombe sur le template
      pour ne pas casser le pipeline.
    """
    if cache_key:
        key = _hash(cache_key, system or "", prompt, LLM_BACKEND)
        cached = _cache_get(key)
        if cached is not None:
            return cached

    try:
        if LLM_BACKEND == "mistral":
            resp = _call_mistral(prompt, system, temperature, max_tokens)
            model = MISTRAL_MODEL
        elif LLM_BACKEND == "anthropic":
            resp = _call_anthropic(prompt, system, temperature, max_tokens)
            model = "claude-opus-4-7"
        elif LLM_BACKEND == "ollama":
            resp = _call_ollama(prompt, system, temperature, max_tokens)
            model = "ollama"
        else:
            resp = _call_template(prompt, system, temperature, max_tokens)
            model = "template"
    except Exception as exc:  # pragma: no cover — robustesse démo
        resp = _call_template(
            f"[LLM_BACKEND={LLM_BACKEND} failed: {exc}]\n\n{prompt}",
            system, temperature, max_tokens,
        )
        model = "template-fallback"

    # On ne met en cache QUE les réponses réussies — jamais un fallback / une
    # erreur (sinon une panne transitoire reste figée dans le cache).
    if cache_key and not resp.startswith("[LLM_BACKEND="):
        _cache_set(key, LLM_BACKEND, model, prompt, resp)
    return resp


def score_anomaly_tabular(row: dict, normal_examples: list[dict]) -> tuple[float, str]:
    """Détection d'anomalie tabulaire zero-shot par LLM (arXiv 2406.16308).

    On présente N exemples « normaux » + la ligne à tester ; le LLM renvoie un
    score ∈ [0,1] et une justification d'une phrase. Cache par hash des features.
    """
    system = (
        "Tu es un détecteur d'anomalies tabulaires expert en signaux maritimes. "
        "Réponds en JSON strict : "
        '{"score": <0..1>, "reason": "<une phrase>"} . '
        "0 = parfaitement normal, 1 = très anormal."
    )
    examples = "\n".join(json.dumps(e, ensure_ascii=False) for e in normal_examples[:20])
    prompt = (
        f"Voici 20 lignes normales :\n{examples}\n\n"
        f"Voici la ligne à tester :\n{json.dumps(row, ensure_ascii=False)}\n\n"
        "Score d'anomalie ?"
    )
    raw = generate(
        prompt, system=system, temperature=0.0, max_tokens=120,
        cache_key=f"anom:{json.dumps(row, sort_keys=True)}",
    )
    try:
        data = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
        return float(data["score"]), str(data.get("reason", ""))
    except Exception:
        return 0.0, "parse_error"
