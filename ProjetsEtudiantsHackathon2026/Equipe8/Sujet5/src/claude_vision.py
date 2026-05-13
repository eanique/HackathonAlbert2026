"""Claude Vision — détecteur LLM-visuel (4e baseline du benchmark Q16).

Repris et durci depuis `Sujet5_Navires/agent/detector.py` puis adapté à la
chaîne BDD-MinArm. **Le LLM-vision n'est pas un détecteur classique** (pas de
bbox pixel-précise), mais c'est une baseline intéressante pour comparer :
    - YOLOv8n (CNN classique, COCO)
    - RT-DETR-l (transformer object detection)
    - Faster R-CNN (two-stage detector)
    - **Claude Vision (LLM multimodal)** ← *ici*

Apport unique : Claude classifie en une passe **type de navire + niveau de
risque géopolitique + raison d'alerte** — utile pour le pipeline d'intel,
mais coûteux en latence et $$ (API). C'est un *complément*, pas un remplaçant.

Tarif/rate-limit : on respecte une pause de 2 s entre appels (cf. Sujet5_Navires).
Fallback : si pas de `ANTHROPIC_API_KEY` ou erreur → retourne un dict structuré
avec `vessel_detected=None, error="…"` pour ne pas casser le pipeline.

Usage:
    from src.claude_vision import classify_with_claude
    r = classify_with_claude("data/images_real/toulon.jpg", lat=43.1, lon=5.9)
    print(r["vessel_category"], r["geopolitical_risk_level"], r["confidence"])
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any

from . import config as C


_DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
_RATE_LIMIT_S = 2.0
_PROMPT_VERSION = "maritime-v1.1-bdd-minarm"


_PROMPT_TEMPLATE = """Tu es analyste senior en surveillance maritime pour le Ministère
des Armées (équivalent DGA/DRM). Tu observes une image satellite optique ou SAR
prise aux coordonnées GPS : LAT {lat}, LON {lon}.

Tu réponds **UNIQUEMENT** en JSON strict (pas de markdown, pas de commentaire,
pas de prose autour). Schéma exact :

{{
  "vessel_detected": true|false,
  "n_vessels_estimated": int,
  "vessel_category": "cargo|tanker|fishing|military|frigate|destroyer|submarine|pleasure|unknown",
  "is_military": true|false,
  "vessel_length_estimate_m": null | float,
  "confidence": 0.0..1.0,
  "geographic_zone": "string libre court (ex. 'Rade de Toulon')",
  "zone_type": "EEZ|strait|high_sea|port|naval_base|contested|coast",
  "geopolitical_risk_level": "low|medium|high|critical",
  "geopolitical_context": "1-2 phrases factuelles, pas de spéculation",
  "is_dark_candidate": true|false,
  "alert": true|false,
  "alert_reason": null | "string",
  "limits": "ce que tu N'as PAS pu déterminer (ex. 'pas de MMSI dans l'image')"
}}

Règles strictes :
- Si tu ne vois rien : `vessel_detected=false`, `confidence` reflète ta certitude
  sur l'absence (0.8+ si l'eau est vide et claire).
- Ne fabrique JAMAIS de MMSI, de pavillon, de nom de navire.
- `alert=true` UNIQUEMENT si : militaire identifié ET zone tendue (high/critical)
  ET comportement notable (hors port, mouvement nocturne).
- Si l'image est SAR (noir et blanc, texture granuleuse), tu peux uniquement
  estimer la *présence* et la *longueur* — pas le *type précis* — sauf signature
  évidente (porte-avions = très long).
"""


def _b64(image_path: str | Path) -> tuple[str, str]:
    """Lit l'image et renvoie (base64, mime)."""
    p = Path(image_path)
    data = base64.standard_b64encode(p.read_bytes()).decode("ascii")
    ext = p.suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
    return data, mime


def _empty_result(reason: str) -> dict:
    return {
        "vessel_detected": None,
        "vessel_category": "unknown",
        "is_military": False,
        "confidence": 0.0,
        "geopolitical_risk_level": "low",
        "alert": False,
        "error": reason,
        "model_version": _PROMPT_VERSION,
    }


def classify_with_claude(
    image_path: str | Path,
    *,
    lat: float,
    lon: float,
    model: str | None = None,
    timeout_s: int = 30,
) -> dict:
    """Une analyse Claude Vision d'une image satellite.

    Returns:
        Dict avec les clés du schéma (voir `_PROMPT_TEMPLATE`).
        En cas d'erreur (clé manquante, image inaccessible, API down),
        renvoie un dict avec `error="…"` — **jamais d'exception**.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return _empty_result("ANTHROPIC_API_KEY absent du .env")

    if not Path(image_path).exists():
        return _empty_result(f"image introuvable : {image_path}")

    try:
        import anthropic  # lazy import (lourd)
    except ImportError:
        return _empty_result("package `anthropic` non installé")

    model = model or _DEFAULT_MODEL
    img_b64, mime = _b64(image_path)
    prompt = _PROMPT_TEMPLATE.format(lat=lat, lon=lon)

    # rate-limit déterministe (2 s entre appels)
    time.sleep(_RATE_LIMIT_S)

    try:
        client = anthropic.Anthropic(api_key=api_key, timeout=timeout_s)
        resp = client.messages.create(
            model=model,
            max_tokens=900,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": mime, "data": img_b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        raw = resp.content[0].text.strip()
        # Robustesse : retire éventuels ```json … ```
        for marker in ("```json", "```"):
            raw = raw.replace(marker, "")
        out = json.loads(raw.strip())
        out["model_version"] = _PROMPT_VERSION
        out["model"] = model
        return out
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        return _empty_result(f"parse error: {type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001 — API errors, network, etc.
        return _empty_result(f"{type(e).__name__}: {e}")


# ----------------------------------------------------------------------
# Benchmark wrapper (compatible Q16 — p6_benchmark)

def bench_claude_vision(image_path: str | Path, lat: float = 43.1, lon: float = 5.9, n_iter: int = 1) -> dict:
    """Mesure la latence d'un appel Claude Vision pour Q16.

    NB : on n'itère QU'UNE FOIS par défaut — un appel Claude = ~$0.01-0.05 +
    2 s de rate-limit. Pas de warm-up multi-appels.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        return {"model": "Claude Vision (Anthropic)", "error": "no ANTHROPIC_API_KEY"}

    latencies_ms = []
    last = None
    for _ in range(n_iter):
        t0 = time.perf_counter()
        last = classify_with_claude(image_path, lat=lat, lon=lon)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
    latencies_ms.sort()
    return {
        "model": f"Claude Vision ({last.get('model', _DEFAULT_MODEL)})",
        "n_iter": n_iter,
        "median_ms": round(latencies_ms[len(latencies_ms)//2], 0),
        "p10_ms": round(latencies_ms[max(0, len(latencies_ms)//10)], 0),
        "p90_ms": round(latencies_ms[min(len(latencies_ms)-1, (9*len(latencies_ms))//10)], 0),
        "n_detections": (
            last.get("n_vessels_estimated") if last and last.get("vessel_detected") else 0
        ),
        "category": last.get("vessel_category"),
        "is_military": last.get("is_military"),
        "geopolitical_risk_level": last.get("geopolitical_risk_level"),
        "error": last.get("error"),
    }


if __name__ == "__main__":
    # Démo : si l'image Toulon Sentinel-2 existe, on l'analyse.
    candidates = list((C.ROOT / "data" / "images_real").glob("*.preview.jpg"))
    if candidates and os.getenv("ANTHROPIC_API_KEY"):
        img = candidates[0]
        print(f"=== Claude Vision sur {img.name} ===")
        out = classify_with_claude(img, lat=43.1, lon=5.9)
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print("Pas d'image preview Sentinel-2 ou pas de ANTHROPIC_API_KEY.")
        print("Démo sans appel (structure de retour) :")
        print(json.dumps(_empty_result("dry-run"), indent=2, ensure_ascii=False))
