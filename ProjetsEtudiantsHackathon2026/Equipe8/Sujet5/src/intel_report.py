"""Q15/L3 — Fiches de renseignement (markdown + PDF) par détection alertée.

Pour chaque détection militaire en zone sensible (ou navire sombre confirmé),
on assemble une fiche structurée :
    - faits : id, position, zone, classe estimée, confidence, timestamp
    - port le plus proche (OSM Nominatim)
    - météo à la position (OpenWeatherMap)
    - statut AIS (sombre / MMSI le plus proche)
    - analyse rédigée par LLM (Mistral) ou template Jinja2 (fallback)

Sortie :
    - `outputs/intel_<detection_id>.md`
    - `outputs/intel_<detection_id>.pdf`  (si reportlab dispo)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from . import config as C
from . import llm

# reportlab est lourd → import paresseux + fallback markdown seul
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
    )

    _HAS_PDF = True
except ImportError:
    _HAS_PDF = False


# ----------------------------------------------------------------------
# Markdown → PDF (rendu très simple, suffit pour la démo jury)

def _markdown_to_pdf(md: str, out_path: Path) -> Path | None:
    """Rendu simple titres/listes → ReportLab. Pas de syntax complexe."""
    if not _HAS_PDF:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    story = []
    for raw in md.splitlines():
        line = raw.rstrip()
        if not line:
            story.append(Spacer(1, 6))
        elif line.startswith("## "):
            story.append(Paragraph(line[3:], styles["Heading2"]))
        elif line.startswith("### "):
            story.append(Paragraph(line[4:], styles["Heading3"]))
        elif line.startswith("- "):
            story.append(Paragraph("• " + line[2:], styles["BodyText"]))
        elif line.startswith("**"):
            story.append(Paragraph(line, styles["BodyText"]))
        elif line.startswith("---"):
            story.append(Spacer(1, 12))
        else:
            story.append(Paragraph(line, styles["BodyText"]))
    doc.build(story)
    return out_path


# ----------------------------------------------------------------------
# Assemblage des faits depuis une ligne du DataFrame d'alertes

def _row_to_facts(row: pd.Series, *, enrich_osint: bool = True) -> dict:
    """Convertit une ligne `df_alerts` en dict prêt pour `llm.generate_intel_note`."""
    facts = {
        "detection_id": str(row.get("detection_id", "?")),
        "image_id": row.get("image_id"),
        "scene_datetime": str(row.get("timestamp", "?")),
        "source": row.get("source") or "(?)",
        "lat": float(row["lat"]) if pd.notna(row.get("lat")) else None,
        "lon": float(row["lon"]) if pd.notna(row.get("lon")) else None,
        "category": row.get("category"),
        "is_military": bool(row.get("is_military", False)),
        "confidence": float(row["confidence"]) if pd.notna(row.get("confidence")) else None,
        "bbox_px": None,  # CSV synthétique = bbox normalisé ; pas de px tant qu'on n'a pas l'image
        "nearest_mil_zone": row.get("nearest_mil_zone_name"),
        "nearest_mil_zone_dist_km": float(row["nearest_mil_zone_dist_km"])
        if pd.notna(row.get("nearest_mil_zone_dist_km"))
        else None,
        "nearest_mil_zone_risk": row.get("nearest_mil_zone_risk"),
        "is_dark": row.get("is_dark"),  # peut être None si pas de pont AIS
        "nearest_mmsi": row.get("nearest_mmsi"),
        "nearest_mmsi_dist_km": row.get("nearest_mmsi_dist_km"),
        "alert": bool(row.get("alert", False)),
        "alert_reason": row.get("alert_reason"),
        "weather": None,
        "nearest_port": None,
        "ais_window_min": C.AIS_TIME_WINDOW_MIN,
        "ais_window_km": C.AIS_DIST_WINDOW_KM,
        # Infrastructure critique sous-marine (câbles, gazoducs)
        "nearest_infra_kind": row.get("nearest_infra_kind"),
        "nearest_infra_name": row.get("nearest_infra_name"),
        "nearest_infra_dist_km": float(row["nearest_infra_dist_km"])
        if pd.notna(row.get("nearest_infra_dist_km"))
        else None,
        "infra_alert_threshold_km": C.INFRA_DIST_KM,
    }

    if enrich_osint and facts["lat"] is not None and facts["lon"] is not None:
        from . import osint_enrich

        try:
            facts["nearest_port"] = osint_enrich.nearest_port_name(facts["lat"], facts["lon"])
        except Exception:  # noqa: BLE001
            pass
        try:
            facts["weather"] = osint_enrich.weather_at(facts["lat"], facts["lon"])
        except Exception:  # noqa: BLE001
            pass
        # Infra critique : seulement si pas déjà renseigné par le pipeline amont
        if facts["nearest_infra_dist_km"] is None:
            try:
                inf = osint_enrich.nearest_submarine_infra(
                    facts["lat"], facts["lon"], radius_km=C.INFRA_SEARCH_RADIUS_KM
                )
                if inf is not None:
                    facts["nearest_infra_kind"] = inf["kind"]
                    facts["nearest_infra_name"] = inf["name"]
                    facts["nearest_infra_dist_km"] = inf["distance_km"]
            except Exception:  # noqa: BLE001
                pass

    return facts


# ----------------------------------------------------------------------
# API publique

def render_intel_note(
    row: pd.Series,
    *,
    out_dir: Path = C.OUTPUTS,
    enrich_osint: bool = True,
    write_pdf: bool = True,
) -> dict:
    """Génère une fiche markdown (+PDF) pour une détection alertée.

    Returns:
        dict {markdown_path, pdf_path, backend_used, latency_s}.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    facts = _row_to_facts(row, enrich_osint=enrich_osint)
    note = llm.generate_intel_note(facts)
    md = note["markdown"]
    det_id = facts["detection_id"].replace("/", "_")
    md_path = out_dir / f"intel_{det_id}.md"
    md_path.write_text(md, encoding="utf-8")

    pdf_path = None
    if write_pdf and _HAS_PDF:
        pdf_path = _markdown_to_pdf(md, out_dir / f"intel_{det_id}.pdf")

    return {
        "detection_id": facts["detection_id"],
        "markdown_path": str(md_path.relative_to(C.ROOT)),
        "pdf_path": str(pdf_path.relative_to(C.ROOT)) if pdf_path else None,
        "backend_used": note["backend_used"],
        "latency_s": note["latency_s"],
    }


def render_all_alerts(
    df_alerts: pd.DataFrame,
    *,
    max_notes: int = 10,
    enrich_osint: bool = False,
    write_pdf: bool = True,
) -> pd.DataFrame:
    """Boucle sur les alertes (limité à `max_notes` pour rester rapide en démo).

    `enrich_osint=False` par défaut : Nominatim et OpenWeatherMap sont rate-limités
    (≥ 1 req/s chacun) ; sur 10 alertes ça reste sous les 30 s. Pour la démo
    on passe `True` ; pour les tests, `False`.
    """
    rows = []
    n = min(len(df_alerts), max_notes)
    for i in range(n):
        r = df_alerts.iloc[i]
        info = render_intel_note(r, enrich_osint=enrich_osint, write_pdf=write_pdf)
        rows.append(info)
    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        idx_path = C.OUTPUTS / "intel_index.csv"
        df_out.to_csv(idx_path, index=False)
    return df_out


# ----------------------------------------------------------------------
# Compat pour `make report` (cf. Makefile)

def regenerate_report() -> None:
    """Hook `make report` : régénère rapport global + fiches.

    Aujourd'hui : régénère uniquement le rapport markdown (le rapport principal
    `rapport_generalisation_detection_navires.md` reste écrit à la main).
    Cette fonction est un placeholder pour intégrer plus tard une boucle
    automatique « run pipeline → write rapport ».
    """
    print("[intel_report] make report : exécuter d'abord `make generalisation`,")
    print("              puis éditer `rapport_generalisation_detection_navires.md`.")


if __name__ == "__main__":
    # Démo : pipeline → 1 fiche
    from . import p5_pipeline

    out = p5_pipeline.run_p5()
    df_alerts = out["df_alerts"]
    if df_alerts.empty:
        print("Aucune alerte → pas de fiche à générer.")
    else:
        # 1 fiche de test (sans OSINT pour rester rapide)
        info = render_intel_note(df_alerts.iloc[0], enrich_osint=False)
        print(f"Fiche : {info['markdown_path']}")
        print(f"PDF   : {info['pdf_path']}")
        print(f"Backend : {info['backend_used']} ({info['latency_s']} s)")
