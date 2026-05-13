"""Enrichissement OSINT — Q15 (port + météo + zones militaires OSM + infra critique)
ET flux de presse navale (RSS) / vérité terrain Wikipedia / interface Telegram.

Fournisseurs gratuits, OSINT, cités OK par le sujet :
    - **Nominatim** (OpenStreetMap, sans clé) : reverse geocoding → nom de
      ville/port le plus proche. Limite communauté : ≤ 1 req/s, User-Agent
      obligatoire.
    - **OpenWeatherMap** (clé gratuite limitée) : météo « current » par lat/lon.
      L'historique nécessite un compte payant — on documente la limitation et
      on retourne la météo current par défaut.
    - **Overpass** (OSM) : zones militaires (`military=naval_base`/`barracks`) et
      infrastructure critique sous-marine (câbles, pipelines) autour d'un point.
    - **Flux RSS de presse navale** (Naval News, USNI, gCaptain, Maritime Executive,
      Mer et Marine, Defense News) — *ciblage* de la chasse Piste B + *corroboration*
      des détections. Libres, structurés ; pas d'API X/Twitter (payante & bridée).
    - **Wikipedia** (REST API summary, CC-BY-SA) : composition publique de flotte par
      base navale → vérité terrain qualitative pour le tableau de chasse.
    - **Telegram** (interface Telethon, lecture seule, *compte dédié*) : milblogging
      RU/UA — couche de corroboration *manuelle* (désinfo élevée), NON branchée en
      flux automatique dans le livrable (cf. rapport §OSINT).

Toutes les fonctions :
    - Cachent les réponses dans `data/cache.sqlite` (clé = sha256 ; TTL pour les flux).
    - Respectent un rate-limit minimal (1 req/s pour Nominatim, idem pour OpenWeather).
    - Retournent `None`/`[]` plutôt que de lever — un échec API ne casse pas le pipeline.

Côté dashboard, les flux RSS sont aussi exposés via la route Next.js
`dashboard/src/app/api/naval-news/route.js` (mêmes feeds, parsing serveur → JSON).
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import sqlite3
import time
from datetime import datetime
from typing import Any

import requests

from . import config as C

_USER_AGENT = "BDD-MinArm-Sujet5/1.0 (hackathon Albert 2026 ; non commercial)"
_NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
_OWM_URL = "https://api.openweathermap.org/data/2.5/weather"
_OVERPASS_URL = "https://overpass-api.de/api/interpreter"


# ----------------------------------------------------------------------
# Cache HTTP partagé (sqlite)

_LAST_CALL: dict[str, float] = {}


def _rate_limit(host: str, min_interval_s: float = 1.05) -> None:
    """Bloque jusqu'à ce que `min_interval_s` se soit écoulé depuis le dernier appel."""
    now = time.time()
    elapsed = now - _LAST_CALL.get(host, 0)
    if elapsed < min_interval_s:
        time.sleep(min_interval_s - elapsed)
    _LAST_CALL[host] = time.time()


def _cache_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(C.CACHE_DB)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS http_cache "
        "(key TEXT PRIMARY KEY, response TEXT, created_at REAL)"
    )
    return conn


def _cached_get(url: str, params: dict) -> dict | None:
    key = hashlib.sha256(f"{url}::{json.dumps(params, sort_keys=True)}".encode()).hexdigest()
    with _cache_conn() as conn:
        row = conn.execute(
            "SELECT response FROM http_cache WHERE key = ?", (key,)
        ).fetchone()
    return json.loads(row[0]) if row else None


def _cached_put(url: str, params: dict, response: dict) -> None:
    key = hashlib.sha256(f"{url}::{json.dumps(params, sort_keys=True)}".encode()).hexdigest()
    with _cache_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO http_cache(key, response, created_at) VALUES (?, ?, ?)",
            (key, json.dumps(response), time.time()),
        )


def _cached_get_fresh(url: str, params: dict, ttl_s: float) -> Any | None:
    """Comme `_cached_get` mais ignore l'entrée si plus vieille que `ttl_s` (flux d'actu)."""
    key = hashlib.sha256(f"{url}::{json.dumps(params, sort_keys=True)}".encode()).hexdigest()
    with _cache_conn() as conn:
        row = conn.execute(
            "SELECT response, created_at FROM http_cache WHERE key = ?", (key,)
        ).fetchone()
    if not row:
        return None
    response, created_at = row
    if ttl_s is not None and (time.time() - (created_at or 0)) > ttl_s:
        return None
    try:
        return json.loads(response)
    except (json.JSONDecodeError, TypeError):
        return None


def _http_get_json(
    url: str, params: dict, headers: dict | None = None, host: str = ""
) -> dict | None:
    cached = _cached_get(url, params)
    if cached is not None:
        return cached
    if host:
        _rate_limit(host)
    try:
        r = requests.get(
            url, params=params, headers=headers or {"User-Agent": _USER_AGENT}, timeout=15
        )
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError) as e:
        print(f"  [osint] {url} → {type(e).__name__}: {e}")
        return None
    _cached_put(url, params, data)
    return data


def _http_post_json(url: str, data: str, host: str = "") -> dict | None:
    cached = _cached_get(url, {"data": data})
    if cached is not None:
        return cached
    if host:
        _rate_limit(host)
    try:
        r = requests.post(
            url, data=data, headers={"User-Agent": _USER_AGENT}, timeout=30
        )
        r.raise_for_status()
        out = r.json()
    except (requests.RequestException, ValueError) as e:
        print(f"  [osint] POST {url} → {type(e).__name__}: {e}")
        return None
    _cached_put(url, {"data": data}, out)
    return out


# ----------------------------------------------------------------------
# Q15a — Nominatim reverse geocoding

def nearest_port_name(lat: float, lon: float) -> str | None:
    """Renvoie le nom de la commune/port la plus proche via OSM Nominatim.

    Pas exactement « le port le plus proche » au sens nautique : c'est la
    commune littorale OSM la plus proche. Pour le pitch c'est largement assez
    (Toulon, Brest, Norfolk, Severomorsk, Sébastopol → noms reconnus).
    """
    data = _http_get_json(
        _NOMINATIM_URL,
        {"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 10},
        host="nominatim",
    )
    if not data:
        return None
    addr = data.get("address") or {}
    return (
        addr.get("city")
        or addr.get("town")
        or addr.get("village")
        or addr.get("municipality")
        or addr.get("county")
        or data.get("display_name", "").split(",")[0]
        or None
    )


# ----------------------------------------------------------------------
# Q15b — OpenWeatherMap

def weather_at(lat: float, lon: float, dt: Any = None) -> dict | None:
    """Météo au point (lat, lon).

    NB : l'historique payant n'est pas appelé (le sujet veut une démo, pas
    un produit). On retourne la météo *current*. Si `dt` est fourni mais
    différent de today, on signale dans le dict que c'est une approximation.
    """
    key = os.getenv("OPENWEATHER_API_KEY")
    if not key:
        return None
    data = _http_get_json(
        _OWM_URL,
        {"lat": lat, "lon": lon, "appid": key, "units": "metric", "lang": "fr"},
        host="openweather",
    )
    if not data:
        return None
    weather_block = (data.get("weather") or [{}])[0]
    main = data.get("main") or {}
    wind = data.get("wind") or {}
    return {
        "description": weather_block.get("description"),
        "temp_c": main.get("temp"),
        "humidity_pct": main.get("humidity"),
        "wind_ms": wind.get("speed"),
        "wind_deg": wind.get("deg"),
        "observed_at": datetime.utcfromtimestamp(data.get("dt", 0)).isoformat() + "Z"
        if data.get("dt")
        else None,
        "_note": (
            "Météo current (gratuit) — l'historique exact à la date de la scène "
            "nécessiterait un compte payant OpenWeatherMap. Approximation jouable "
            "pour une démo si dt récent (< 1 j)."
        ),
    }


# ----------------------------------------------------------------------
# Q15c — Overpass : zones militaires OSM autour d'un point

def military_zones_osm(lat: float, lon: float, radius_km: float = 25.0) -> list[dict]:
    """Liste les éléments OSM `military=*` dans un rayon `radius_km`.

    Permet de croiser nos `military_zones.csv` (synthétique) avec OSM (réel).
    Sortie : liste de dict `{name, kind, lat, lon, country, distance_km}`.
    """
    radius_m = int(radius_km * 1000)
    query = (
        "[out:json][timeout:25];"
        "("
        f'node["military"](around:{radius_m},{lat},{lon});'
        f'way["military"](around:{radius_m},{lat},{lon});'
        f'relation["military"](around:{radius_m},{lat},{lon});'
        ");out center tags;"
    )
    data = _http_post_json(_OVERPASS_URL, query, host="overpass")
    if not data:
        return []

    from geopy.distance import geodesic  # importé tard pour ne pas alourdir l'import top

    out = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        # node : (lat, lon) direct ; way/relation : `center`
        elat = el.get("lat") or (el.get("center") or {}).get("lat")
        elon = el.get("lon") or (el.get("center") or {}).get("lon")
        if elat is None or elon is None:
            continue
        try:
            dist = geodesic((lat, lon), (elat, elon)).km
        except (ValueError, TypeError):
            continue
        out.append(
            {
                "name": tags.get("name", tags.get("alt_name", "?")),
                "kind": tags.get("military", "?"),
                "lat": elat,
                "lon": elon,
                "country": tags.get("addr:country"),
                "distance_km": round(dist, 2),
                "osm_id": el.get("id"),
                "osm_type": el.get("type"),
            }
        )
    out.sort(key=lambda d: d["distance_km"])
    return out


# ----------------------------------------------------------------------
# Q15d — Overpass : infrastructure critique sous-marine (Levier L4)
#
# On vise les câbles télécom/électriques sous-marins et les gazoducs/oléoducs
# sous-marins. Précédents : Nord Stream (sept. 2022), Eagle S × EstLink (déc.
# 2024). Un cargo qui passe à proximité = signal de guerre hybride.

_INFRA_KINDS_PRIORITY = (
    "Gazoduc",
    "Oléoduc",
    "Pipeline sous-marin",
    "Câble électrique sous-marin",
    "Câble télécom sous-marin",
    "Câble sous-marin",
)


def _classify_infra(tags: dict) -> str:
    """Mappe les tags OSM vers une étiquette lisible (FR)."""
    substance = (tags.get("substance") or "").lower()
    if tags.get("man_made") == "pipeline":
        if substance in {"gas", "natural_gas", "methane"}:
            return "Gazoduc"
        if substance in {"oil", "petroleum", "crude_oil"}:
            return "Oléoduc"
        return "Pipeline sous-marin"
    if tags.get("seamark:type") == "cable_submarine":
        cat = (tags.get("seamark:cable_submarine:category") or "").lower()
        if cat in {"power", "power_line"}:
            return "Câble électrique sous-marin"
        if cat in {"telephone", "telegraph", "telecom", "telecommunication"}:
            return "Câble télécom sous-marin"
        return "Câble sous-marin"
    if tags.get("submarine") == "yes":
        if tags.get("power") == "cable":
            return "Câble électrique sous-marin"
        if tags.get("communication") == "line":
            return "Câble télécom sous-marin"
        return "Câble sous-marin"
    return "Infrastructure sous-marine"


def _polyline_min_distance_km(
    lat: float, lon: float, geometry: list[dict]
) -> tuple[float, float, float]:
    """Distance minimale (km) d'un point à une polyline OSM, + (lat, lon) du
    nœud le plus proche. Approximation node-by-node : suffisante au seuil ≥ 1 km."""
    from geopy.distance import geodesic

    best = (float("inf"), lat, lon)
    for node in geometry or []:
        nlat, nlon = node.get("lat"), node.get("lon")
        if nlat is None or nlon is None:
            continue
        try:
            d = geodesic((lat, lon), (nlat, nlon)).km
        except (ValueError, TypeError):
            continue
        if d < best[0]:
            best = (d, nlat, nlon)
    return best


def submarine_infra_nearby(
    lat: float,
    lon: float,
    radius_km: float = 10.0,
) -> list[dict]:
    """Liste les câbles + pipelines sous-marins OSM dans `radius_km`.

    Returns:
        Liste triée par distance croissante :
            `{kind, name, osm_id, osm_type, distance_km, near_lat, near_lon, tags}`.

    Limites :
        OSM est notoirement lacunaire pour les câbles télécom (couverture
        partielle, surtout Europe / Atlantique Nord / mers asiatiques). Pour
        une couverture complète, on documente dans le rapport l'option
        commerciale TeleGeography. C'est l'approche jouable côté OSINT pur.
    """
    radius_m = int(radius_km * 1000)
    query = (
        "[out:json][timeout:30];"
        "("
        # Pipelines sous-marins (gaz/pétrole)
        f'way["man_made"="pipeline"]["location"~"underwater|submarine"](around:{radius_m},{lat},{lon});'
        f'way["man_made"="pipeline"]["seamark:type"](around:{radius_m},{lat},{lon});'
        # Câbles sous-marins (seamark canonique)
        f'way["seamark:type"="cable_submarine"](around:{radius_m},{lat},{lon});'
        # Variantes tags
        f'way["submarine"="yes"]["power"="cable"](around:{radius_m},{lat},{lon});'
        f'way["submarine"="yes"]["communication"="line"](around:{radius_m},{lat},{lon});'
        f'way["power"="cable"]["location"="submarine"](around:{radius_m},{lat},{lon});'
        ");out geom tags;"
    )
    data = _http_post_json(_OVERPASS_URL, query, host="overpass")
    if not data:
        return []

    out: list[dict] = []
    for el in data.get("elements", []):
        if el.get("type") != "way":
            continue
        tags = el.get("tags", {}) or {}
        geometry = el.get("geometry") or []
        dist_km, near_lat, near_lon = _polyline_min_distance_km(lat, lon, geometry)
        if dist_km == float("inf"):
            continue
        out.append(
            {
                "kind": _classify_infra(tags),
                "name": tags.get("name") or tags.get("ref") or tags.get("operator") or "(sans nom)",
                "osm_id": el.get("id"),
                "osm_type": el.get("type"),
                "distance_km": round(dist_km, 3),
                "near_lat": near_lat,
                "near_lon": near_lon,
                "substance": tags.get("substance"),
                "operator": tags.get("operator"),
                "geometry": geometry,
            }
        )
    out.sort(key=lambda d: d["distance_km"])
    return out


def nearest_submarine_infra(
    lat: float,
    lon: float,
    radius_km: float = 10.0,
) -> dict | None:
    """Raccourci : l'infra critique la plus proche (ou None si rayon vide)."""
    items = submarine_infra_nearby(lat, lon, radius_km=radius_km)
    return items[0] if items else None


# ----------------------------------------------------------------------
# Flux de presse navale (RSS) — ciblage Piste B + corroboration des détections

_RSS_USER_AGENT = "MaritimeWatch/1.0 (BDD-MinArm hackathon Albert 2026)"


def _is_naval(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in C.OSINT_NAVAL_KEYWORDS)


def _strip_html(s: str) -> str:
    # Décoder les entités (y compris &lt;p&gt; → <p>) PUIS retirer les balises.
    txt = html.unescape(s or "")
    txt = re.sub(r"<[^>]+>", "", txt)
    return re.sub(r"\s+", " ", txt).strip()


def _parse_feed_xml(xml: str, source_label: str) -> list[dict]:
    """Parse RSS 2.0 OU Atom → liste d'items `{source, title, link, published, summary}`.

    Utilise `feedparser` s'il est installé (robuste), sinon un parseur regex de
    secours (suffisant pour ces flux ; même approche que la route Next.js du dashboard).
    """
    try:
        import feedparser  # type: ignore

        parsed = feedparser.parse(xml)
        items = []
        for e in parsed.entries:
            items.append(
                {
                    "source": source_label,
                    "title": _strip_html(e.get("title", "")),
                    "link": e.get("link", ""),
                    "published": str(e.get("published") or e.get("updated") or ""),
                    "summary": _strip_html(e.get("summary") or e.get("description") or "")[:600],
                }
            )
        return items
    except ImportError:
        pass

    # ---- Parseur de secours (RSS <item> ou Atom <entry>) ----
    def _tag(block: str, name: str) -> str:
        m = re.search(rf"<{name}[^>]*>(.*?)</{name}>", block, re.S)
        if not m:
            return ""
        val = m.group(1).strip()
        cd = re.match(r"<!\[CDATA\[(.*?)\]\]>", val, re.S)
        return _strip_html(cd.group(1) if cd else val)

    def _link(block: str) -> str:
        m = re.search(r"<link[^>]*\bhref=\"([^\"]+)\"", block)  # Atom
        if m:
            return m.group(1)
        return _tag(block, "link")  # RSS

    items: list[dict] = []
    blocks = re.findall(r"<item[ >].*?</item>", xml, re.S) or re.findall(
        r"<entry[ >].*?</entry>", xml, re.S
    )
    for b in blocks:
        items.append(
            {
                "source": source_label,
                "title": _tag(b, "title"),
                "link": _link(b),
                "published": _tag(b, "pubDate") or _tag(b, "published") or _tag(b, "updated"),
                "summary": (_tag(b, "description") or _tag(b, "summary") or _tag(b, "content"))[:600],
            }
        )
    return items


def fetch_rss_feed(key: str, limit: int = 30) -> list[dict]:
    """Récupère un flux RSS navals identifié par sa clé dans `config.OSINT_RSS_FEEDS`.

    Returns:
        Liste d'items `{source, title, link, published, summary}` (vide si flux/clé KO).
    """
    if key not in C.OSINT_RSS_FEEDS:
        return []
    label, url = C.OSINT_RSS_FEEDS[key]
    cached = _cached_get_fresh(url, {"_": "rss"}, ttl_s=C.OSINT_FEED_TTL_S)
    if cached is not None:
        return cached[:limit]
    try:
        r = requests.get(url, headers={"User-Agent": _RSS_USER_AGENT}, timeout=15)
        r.raise_for_status()
        items = _parse_feed_xml(r.text, label)
    except (requests.RequestException, ValueError) as e:
        print(f"  [osint] RSS {label} → {type(e).__name__}: {e}")
        return []
    _cached_put(url, {"_": "rss"}, items)
    return items[:limit]


def fetch_naval_news(only_naval: bool = True, per_feed: int = 20) -> list[dict]:
    """Agrège tous les flux RSS navals de `config.OSINT_RSS_FEEDS`.

    Args:
        only_naval: ne garder que les items dont titre+résumé matchent un mot-clé
            naval (`config.OSINT_NAVAL_KEYWORDS`) — réduit le bruit.
        per_feed: nb max d'items lus par flux.

    Returns:
        Liste d'items triés (les plus récents d'abord quand la date est parsable),
        chacun enrichi de `is_naval` (bool). Vide si aucun flux joignable.
    """
    out: list[dict] = []
    for key in C.OSINT_RSS_FEEDS:
        items = fetch_rss_feed(key, limit=per_feed)
        for it in items:
            it["is_naval"] = _is_naval(f"{it.get('title','')} {it.get('summary','')}")
        out.extend(items)
    if only_naval:
        out = [it for it in out if it.get("is_naval")]

    def _ts(it: dict) -> float:
        raw = (it.get("published") or "").strip()
        if not raw:
            return 0.0
        # RSS 2.0 : "Mon, 12 May 2026 09:00:00 GMT"
        try:
            from email.utils import parsedate_to_datetime

            d = parsedate_to_datetime(raw)
            if d is not None:
                return d.timestamp()
        except (TypeError, ValueError):
            pass
        # Atom : ISO 8601 ("2026-05-11T10:00:00Z")
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0

    out.sort(key=_ts, reverse=True)
    return out


# ----------------------------------------------------------------------
# Wikipedia — vérité terrain « composition de flotte par base » (Levier L1)

_WIKI_REST = "https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"


def wikipedia_summary(title: str, lang: str = "en") -> dict | None:
    """Résumé d'une page Wikipedia via la REST API (CC-BY-SA).

    Returns:
        dict `{title, extract, url, lang}` ou None si page introuvable / réseau KO.
    """
    url = _WIKI_REST.format(lang=lang, title=requests.utils.quote(title.replace(" ", "_")))
    cached = _cached_get_fresh(url, {"_": "wiki"}, ttl_s=C.OSINT_WIKI_TTL_S)
    if cached is not None:
        return cached
    try:
        r = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=15)
        if r.status_code != 200:
            return None
        j = r.json()
    except (requests.RequestException, ValueError) as e:
        print(f"  [osint] Wikipedia '{title}' → {type(e).__name__}: {e}")
        return None
    out = {
        "title": j.get("title", title),
        "extract": j.get("extract", ""),
        "url": j.get("content_urls", {}).get("desktop", {}).get("page", ""),
        "lang": lang,
    }
    _cached_put(url, {"_": "wiki"}, out)
    return out


def wikipedia_fleet_for_base(base_name: str, lang: str = "en") -> dict | None:
    """Page Wikipedia de la flotte rattachée à une base de `config.BASES_NAVALES`."""
    page = C.OSINT_WIKI_FLEET_PAGES.get(base_name)
    if not page:
        return None
    summ = wikipedia_summary(page, lang=lang)
    if summ:
        summ["base"] = base_name
    return summ


# ----------------------------------------------------------------------
# Telegram — interface (Telethon). NON branché en flux automatique pour le rendu.


def fetch_telegram_channels(
    channels: dict | None = None,
    limit_per_channel: int = 50,
    since_days: int = 7,
) -> list[dict]:
    """Lit les derniers messages de chaînes Telegram publiques (lecture seule).

    Implémentation **optionnelle** : nécessite `telethon` + `TELEGRAM_API_ID` /
    `TELEGRAM_API_HASH` (https://my.telegram.org) dans `.env`, et un *compte dédié*
    (jamais un numéro perso). Dans le livrable, Telegram est traité comme couche de
    corroboration *manuelle* → cette fonction renvoie `[]` tant que la lib/les
    credentials ne sont pas présents, et l'implémentation effective est laissée en
    esquisse (cf. ci-dessous) pour ne pas dépendre d'un service bruité/contrôlé.

        from telethon.sync import TelegramClient
        api_id, api_hash = int(os.environ["TELEGRAM_API_ID"]), os.environ["TELEGRAM_API_HASH"]
        with TelegramClient("bdd_minarm_osint", api_id, api_hash) as client:
            for handle in channels:
                for msg in client.iter_messages(handle, limit=limit_per_channel):
                    ...  # filtrer fenêtre temporelle, _is_naval(msg.message),
                         # traduire RU/UA via src/llm.py, géolocaliser, recouper
    """
    channels = channels or C.OSINT_TELEGRAM_CHANNELS
    try:
        import telethon  # type: ignore  # noqa: F401
    except ImportError:
        print(
            "  [osint] telethon non installé — Telegram = source de corroboration "
            "manuelle (cf. rapport). `uv pip install telethon` + TELEGRAM_API_ID/HASH pour activer."
        )
        return []
    if not (os.getenv("TELEGRAM_API_ID") and os.getenv("TELEGRAM_API_HASH")):
        print("  [osint] TELEGRAM_API_ID / TELEGRAM_API_HASH absents de .env — Telegram ignoré.")
        return []
    raise NotImplementedError(
        "Activer en suivant l'esquisse de la docstring (Telethon + compte dédié). "
        "Non requis pour le livrable — voir rapport §OSINT."
    )


# ----------------------------------------------------------------------
# Démo standalone

if __name__ == "__main__":
    print("=== Flux RSS presse navale (items filtrés 'naval') ===")
    for it in fetch_naval_news()[:8]:
        print(f"  · [{it['source']}] {it['title']}")

    print("\n=== Wikipedia — flotte par base navale ===")
    for base in C.BASES_NAVALES:
        w = wikipedia_fleet_for_base(base)
        if w:
            print(f"  · {base:<20} {w['title']} — {(w['extract'] or '')[:90].strip()}…")

    print("\n=== Telegram (interface — non branché par défaut) ===")
    fetch_telegram_channels()

    print("\n=== Nominatim (Toulon) ===")
    print(nearest_port_name(43.10, 5.90))

    print("\n=== OpenWeatherMap (Brest) ===")
    print(weather_at(48.39, -4.49))

    print("\n=== Overpass military (Norfolk, 30 km) ===")
    for z in military_zones_osm(36.95, -76.31, radius_km=30)[:10]:
        print(f"  {z['distance_km']:5.1f} km  {z['kind']:<20} {z['name']}")

    # Baltique près d'EstLink (Eagle S, déc. 2024) — câbles HVDC FI-EE
    print("\n=== Overpass infra critique sous-marine (Golfe de Finlande, 20 km) ===")
    for inf in submarine_infra_nearby(59.95, 25.30, radius_km=20)[:10]:
        print(f"  {inf['distance_km']:5.2f} km  {inf['kind']:<32} {inf['name']}")
