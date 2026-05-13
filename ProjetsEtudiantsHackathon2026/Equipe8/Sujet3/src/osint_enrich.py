"""Enrichissement OSINT (Levier 2 — Sujet 3).

Toutes les sources sont **gratuites**. Aucune clé n'est jamais committée.

Sources :
  - Equasis (equasis.org)        : historique noms/pavillons, PSC, armateur.
  - UIT MMSI / MARS              : validation MMSI, MID → pays (config.mmsi_country).
  - OpenSanctions API            : navire/armateur sanctionné OFAC/UE.
  - Flux RSS maritime            : ingestion automatique (Q14).
  - GFW API (optionnel)          : gap events / encounters / loitering pré-calculés.
  - MCP shom-wrecks              : épaves SHOM pour écarter les faux positifs.

Toutes les fonctions ont un **cache SQLite** (data/cache.sqlite) pour rejouer
la démo hors-ligne et de façon déterministe.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Any

import requests

from .config import (
    CACHE_DB,
    EQUASIS_PASSWORD,
    EQUASIS_USERNAME,
    GFW_API_TOKEN,
    OPENSANCTIONS_API_KEY,
    mmsi_country,
)

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Cache
# ----------------------------------------------------------------------------


def _ensure_cache_tables() -> None:
    with sqlite3.connect(CACHE_DB) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS osint_equasis (
                imo_or_mmsi TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                ts TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS osint_opensanctions (
                query TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                ts TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS osint_gfw (
                query TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                ts TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS osint_rss (
                guid TEXT PRIMARY KEY,
                feed TEXT,
                title TEXT,
                link TEXT,
                summary TEXT,
                published TEXT,
                ts TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)


def _cache_get(table: str, key_col: str, key: str, ttl_hours: int = 24*7) -> Any | None:
    _ensure_cache_tables()
    with sqlite3.connect(CACHE_DB) as conn:
        row = conn.execute(
            f"SELECT payload, ts FROM {table} WHERE {key_col} = ?", (key,)
        ).fetchone()
    if not row:
        return None
    try:
        ts = datetime.fromisoformat(row[1])
        if ts < datetime.utcnow() - timedelta(hours=ttl_hours):
            return None
    except Exception:
        pass
    return json.loads(row[0])


def _cache_set(table: str, key_col: str, key: str, value: Any) -> None:
    _ensure_cache_tables()
    with sqlite3.connect(CACHE_DB) as conn:
        conn.execute(
            f"INSERT OR REPLACE INTO {table} ({key_col}, payload) VALUES (?, ?)",
            (key, json.dumps(value, default=str)),
        )


# ----------------------------------------------------------------------------
# UIT MMSI / MARS (offline — table MID dans config.py)
# ----------------------------------------------------------------------------


def lookup_mmsi(mmsi: str) -> dict:
    """Validation MMSI + pays attribué (offline, table UIT_MID)."""
    country = mmsi_country(mmsi)
    return {
        "mmsi": mmsi,
        "valid": bool(mmsi and len(mmsi) == 9 and mmsi.isdigit()),
        "country_attributed": country,
        "mid": mmsi[:3] if mmsi and len(mmsi) >= 3 else None,
    }


# ----------------------------------------------------------------------------
# OpenSanctions
# ----------------------------------------------------------------------------


def query_opensanctions(name: str | None = None, imo: str | None = None,
                        mmsi: str | None = None) -> list[dict]:
    """Recherche un navire / une organisation dans OpenSanctions (OFAC, UE…).

    Endpoint : https://api.opensanctions.org/search/default?q=...
    Sans clé : quota faible mais utilisable. Clé : header `Authorization: ApiKey ...`.
    """
    if not (name or imo or mmsi):
        return []
    q = " ".join(filter(None, [name, imo, mmsi]))
    cached = _cache_get("osint_opensanctions", "query", q)
    if cached is not None:
        return cached

    headers = {}
    if OPENSANCTIONS_API_KEY:
        headers["Authorization"] = f"ApiKey {OPENSANCTIONS_API_KEY}"
    try:
        r = requests.get(
            "https://api.opensanctions.org/search/default",
            params={"q": q, "schema": "Vessel"}, headers=headers, timeout=10,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
    except Exception as exc:
        log.warning("OpenSanctions KO (%s) — fallback []", exc)
        results = []
    _cache_set("osint_opensanctions", "query", q, results)
    return results


# ----------------------------------------------------------------------------
# Equasis (compte gratuit)
# ----------------------------------------------------------------------------


def lookup_equasis(imo_or_mmsi: str) -> dict:
    """Historique noms/pavillons, PSC, armateur. Stub HTTP (à finaliser en J1).

    Note : Equasis n'a pas d'API publique → scraping post-login. On garde un
    cache durable (TTL 30 j) car les fiches changent peu.
    """
    cached = _cache_get("osint_equasis", "imo_or_mmsi", imo_or_mmsi, ttl_hours=24 * 30)
    if cached is not None:
        return cached

    if not (EQUASIS_USERNAME and EQUASIS_PASSWORD):
        log.info("Equasis : pas d'identifiants → renvoi vide.")
        return {"available": False, "reason": "no_credentials"}

    # TODO J1 — Agent B : scraping (session POST + parsing HTML).
    payload = {"available": False, "reason": "not_implemented", "imo_or_mmsi": imo_or_mmsi}
    _cache_set("osint_equasis", "imo_or_mmsi", imo_or_mmsi, payload)
    return payload


# ----------------------------------------------------------------------------
# RSS (Q14)
# ----------------------------------------------------------------------------


def fetch_rss(feed_url: str, max_items: int = 30) -> list[dict]:
    """Lit un flux RSS et stocke les items en cache (déduplication par GUID).

    Retour : liste de dict {guid, title, link, summary, published}.
    """
    import feedparser

    parsed = feedparser.parse(feed_url)
    items: list[dict] = []
    for e in parsed.entries[:max_items]:
        guid = e.get("id") or e.get("link") or e.get("title")
        item = {
            "guid": guid, "feed": feed_url,
            "title": e.get("title", ""), "link": e.get("link", ""),
            "summary": e.get("summary", ""),
            "published": str(e.get("published", "")),
        }
        items.append(item)
        with sqlite3.connect(CACHE_DB) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO osint_rss (guid, feed, title, link, summary, published) VALUES (?, ?, ?, ?, ?, ?)",
                (item["guid"], item["feed"], item["title"], item["link"],
                 item["summary"], item["published"]),
            )
    return items


def extract_suspicious_mmsi_from_rss(items: list[dict]) -> set[str]:
    """Heuristique simple : on regarde si un MMSI (9 chiffres) ou un IMO est
    cité dans le titre/résumé. À enrichir par LLM (`src/llm.py`) en bonus.
    """
    import re

    mmsi_pat = re.compile(r"\b\d{9}\b")
    found: set[str] = set()
    for it in items:
        text = " ".join([it.get("title", ""), it.get("summary", "")])
        for m in mmsi_pat.findall(text):
            found.add(m)
    return found


# ----------------------------------------------------------------------------
# Global Fishing Watch (optionnel)
# ----------------------------------------------------------------------------


def query_gfw_events(bbox: tuple[float, float, float, float],
                    event_type: str = "GAP",
                    days: int = 30) -> list[dict]:
    """Récupère des événements (GAP / ENCOUNTER / LOITERING) sur l'AIS mondial.

    Doc : https://globalfishingwatch.org/our-apis/
    """
    if not GFW_API_TOKEN:
        return []
    cache_key = f"{event_type}:{bbox}:{days}"
    cached = _cache_get("osint_gfw", "query", cache_key, ttl_hours=12)
    if cached is not None:
        return cached
    try:
        r = requests.get(
            "https://gateway.api.globalfishingwatch.org/v3/events",
            params={"datasets[0]": f"public-global-{event_type.lower()}-events:latest",
                    "start-date": (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d"),
                    "end-date": datetime.utcnow().strftime("%Y-%m-%d"),
                    "bbox": ",".join(map(str, bbox))},
            headers={"Authorization": f"Bearer {GFW_API_TOKEN}"}, timeout=20,
        )
        r.raise_for_status()
        events = r.json().get("entries", [])
    except Exception as exc:
        log.warning("GFW KO (%s) — fallback []", exc)
        events = []
    _cache_set("osint_gfw", "query", cache_key, events)
    return events


# ----------------------------------------------------------------------------
# Aggrégat : dossier OSINT d'un navire
# ----------------------------------------------------------------------------


def build_osint_dossier(mmsi: str, ship_row: dict) -> dict:
    """Construit un dossier OSINT consolidé pour un navire donné."""
    uit = lookup_mmsi(mmsi)
    sanctions = query_opensanctions(
        name=ship_row.get("name"),
        imo=ship_row.get("imo"),
        mmsi=mmsi,
    )
    equasis = lookup_equasis(ship_row.get("imo") or mmsi)

    dossier = {
        "mmsi": mmsi,
        "uit": uit,
        "flag_declared": ship_row.get("flag"),
        "flag_mismatch_uit": (uit.get("country_attributed")
                              and ship_row.get("flag")
                              and uit["country_attributed"] != ship_row.get("flag")),
        "name": ship_row.get("name"),
        "imo": ship_row.get("imo"),
        "historical_names": ship_row.get("historical_names_list"),
        "n_names_historical": ship_row.get("n_names_historical"),
        "is_suspicious": ship_row.get("is_suspicious"),
        "sanctions": sanctions[:5],
        "sanctioned": len(sanctions) > 0,
        "equasis": equasis,
    }
    return dossier
