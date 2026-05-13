"""Client AISStream.io — flux AIS temps réel par WebSocket (optionnel).

Sert pour la démo « live » du Sujet 3 : la carte se met à jour avec de vrais
navires captés en direct, et le pipeline d'identification tourne dessus.

Clé gratuite : https://aisstream.io
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from .config import AISSTREAM_API_KEY

log = logging.getLogger(__name__)


async def stream(bbox: tuple[float, float, float, float] = (40.0, -10.0, 60.0, 10.0),
                 max_messages: int = 1000) -> AsyncIterator[dict]:
    """Itère sur les messages AIS reçus en temps réel pour `bbox`.

    bbox = (min_lat, min_lon, max_lat, max_lon).
    Renvoie des dicts {mmsi, timestamp, latitude, longitude, speed, course, …}
    pré-normalisés au schéma de `ais_data_large.csv`.
    """
    if not AISSTREAM_API_KEY:
        log.warning("AISSTREAM_API_KEY manquante — démo live indisponible.")
        return

    import websockets  # lazy import — optionnel

    sub = {
        "APIKey": AISSTREAM_API_KEY,
        "BoundingBoxes": [[[bbox[0], bbox[1]], [bbox[2], bbox[3]]]],
        "FilterMessageTypes": ["PositionReport"],
    }

    n = 0
    async with websockets.connect("wss://stream.aisstream.io/v0/stream") as ws:
        await ws.send(json.dumps(sub))
        async for raw in ws:
            data = json.loads(raw)
            pos = data.get("Message", {}).get("PositionReport", {})
            if not pos:
                continue
            meta = data.get("MetaData", {})
            yield {
                "mmsi": str(meta.get("MMSI", "")),
                "timestamp": meta.get("time_utc"),
                "latitude": pos.get("Latitude"),
                "longitude": pos.get("Longitude"),
                "speed": pos.get("Sog"),
                "course": pos.get("Cog"),
                "heading": pos.get("TrueHeading"),
                "status": str(pos.get("NavigationalStatus", "")),
                "ais_active": True,
            }
            n += 1
            if n >= max_messages:
                break


def run_demo(bbox=(40.0, -10.0, 60.0, 10.0), n: int = 100) -> list[dict]:
    """Helper synchrone pour la démo : collecte `n` messages puis stoppe."""
    async def _collect():
        out = []
        async for msg in stream(bbox=bbox, max_messages=n):
            out.append(msg)
        return out

    return asyncio.run(_collect())
