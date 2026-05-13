"""Fiche de renseignement LLM (Mistral) + PDF d'alerte (Sujet 3, Levier 2).

Pipeline :
    score multi-facteurs (anomaly_score.py)
        + dossier OSINT (osint_enrich.py)
        → prompt structuré → LLM (Mistral) → fiche Markdown + PDF.

Le PDF est généré avec `reportlab`. Fallback ultime : template Jinja2 si
aucun LLM n'est disponible (le pipeline ne casse jamais).
"""

from __future__ import annotations

import textwrap
from datetime import datetime
from pathlib import Path

from jinja2 import Template
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from .config import LLM_BACKEND, OUTPUTS
from .llm import generate

# ----------------------------------------------------------------------------
# Templates
# ----------------------------------------------------------------------------


_LLM_SYSTEM = textwrap.dedent("""\
    Tu es un officier de renseignement maritime. Tu rédiges des **notes de
    renseignement** factuelles, sobres et structurées à destination du
    Ministère des Armées. Tu cites systématiquement les sources OSINT
    utilisées (Equasis, UIT MMSI, OpenSanctions, RSS, GFW). Tu **n'inventes
    rien** : si une information n'est pas fournie, tu écris « non disponible » ;
    tu ne donnes JAMAIS de correspondance MID→pays que tu n'as pas reçue.
    Glossaire des « raisons » du score (à utiliser tel quel, ne pas réinterpréter) :
      - `lof` = Local Outlier Factor (anomalie de densité locale dans l'espace des signatures radio) ;
      - `zone` = comportement AIS atypique pour la zone géographique ;
      - `ais_off` = AIS désactivé plus de 24 h consécutives ;
      - `position_mismatch` = écart entre position AIS et position radio ;
      - `speed` = vitesse implicite anormale ;
      - `spoofing_rules` = règles de spoofing (MMSI dupliqué, timestamps incohérents…) ;
      - `fake_flag` = signature radio atypique pour le pavillon déclaré ;
      - `name_change` = plus de 2 noms historiques.
    Tu rends ta réponse **directement en Markdown** (pas de bloc de code ```),
    ~250 mots, en français.""")


_LLM_PROMPT_TEMPLATE = textwrap.dedent("""\
    # Identifiant du navire
    MMSI : {mmsi}
    Nom déclaré : {name}
    Pavillon déclaré : {flag_declared}
    IMO : {imo}
    Type : {ship_type}

    # Vérifications OSINT
    MID UIT (3 premiers chiffres MMSI) → pays attribué : {country_attributed}
    Incohérence MID UIT vs pavillon : {flag_mismatch_uit}
    Noms historiques : {historical_names}
    Nb de changements de nom : {n_names_historical}
    Sanctions OpenSanctions : {sanctioned} ({n_sanctions} résultat(s))
    Marqué `is_suspicious` dans le registre : {is_suspicious}

    # Score d'anomalie (sortie modèles ML)
    Score global : {score:.2f} / 1.0
    Principales raisons : {top_reasons}

    # Demande
    Rédige une **fiche de renseignement** structurée :
    1. **Synthèse** (3 lignes max).
    2. **Indices de suspicion** (liste à puces, basée sur les vérifications OSINT
       et le score d'anomalie).
    3. **Recommandation opérationnelle** (1 phrase).
    4. **Sources mobilisées** (liste avec URLs).""")


_TEMPLATE_FALLBACK = Template(textwrap.dedent("""\
    # Fiche de renseignement — MMSI {{ mmsi }}

    *Générée par template Jinja2 (LLM indisponible).*

    ## Synthèse
    Navire **{{ name }}** (MMSI {{ mmsi }}, IMO {{ imo }}, pavillon **{{ flag_declared }}**).
    Score global d'anomalie : **{{ '%.2f' % score }}**.

    ## Indices de suspicion
    {% if flag_mismatch_uit %}- ⚠️ Pavillon déclaré (**{{ flag_declared }}**) ≠ pays MID UIT (**{{ country_attributed }}**).{% endif %}
    {% if n_names_historical and n_names_historical > 2 %}- ⚠️ **{{ n_names_historical }} noms historiques** (changements répétés).{% endif %}
    {% if sanctioned %}- ⚠️ Présence dans OpenSanctions : **{{ n_sanctions }} résultat(s)**.{% endif %}
    {% if is_suspicious %}- ⚠️ Marqué `is_suspicious` dans le registre de référence.{% endif %}
    - Top raisons (score) : **{{ top_reasons }}**.

    ## Recommandation
    Investigation OSINT approfondie recommandée si score ≥ 0.5, action consulaire
    si score ≥ 0.7 et sanctions confirmées.

    ## Sources mobilisées
    - Equasis · UIT MMSI / MARS · OpenSanctions · Cours A. Bogroff (Albert School).
    """))


# ----------------------------------------------------------------------------
# Génération
# ----------------------------------------------------------------------------


def render_intel_note(*, mmsi: str, name: str | None, imo: str | None,
                      flag_declared: str | None, ship_type: str | None,
                      country_attributed: str | None, flag_mismatch_uit: bool,
                      historical_names: list[str] | None,
                      n_names_historical: int | None,
                      sanctioned: bool, n_sanctions: int,
                      is_suspicious: bool, score: float,
                      top_reasons: str) -> str:
    """Renvoie la fiche Markdown — via LLM (Mistral) ou fallback Jinja2."""
    ctx = {
        "mmsi": mmsi or "?", "name": name or "?", "imo": imo or "?",
        "flag_declared": flag_declared or "?", "ship_type": ship_type or "?",
        "country_attributed": country_attributed or "?",
        "flag_mismatch_uit": bool(flag_mismatch_uit),
        "historical_names": ", ".join(historical_names or []) or "—",
        "n_names_historical": int(n_names_historical or 0),
        "sanctioned": bool(sanctioned), "n_sanctions": int(n_sanctions),
        "is_suspicious": bool(is_suspicious),
        "score": float(score), "top_reasons": top_reasons or "—",
    }

    if LLM_BACKEND == "template":
        return _TEMPLATE_FALLBACK.render(**ctx)

    prompt = _LLM_PROMPT_TEMPLATE.format(**ctx)
    md = generate(prompt, system=_LLM_SYSTEM, temperature=0.2,
                  max_tokens=600, cache_key=f"intel:{mmsi}:{score:.2f}")
    if not md or "[LLM_BACKEND=" in md[:32]:
        return _TEMPLATE_FALLBACK.render(**ctx)
    # Nettoyage : enlever d'éventuelles balises de bloc de code ```markdown
    md = md.strip()
    if md.startswith("```"):
        md = md.split("\n", 1)[-1] if "\n" in md else md
        if md.rstrip().endswith("```"):
            md = md.rstrip()[:-3].rstrip()
    return md


# ----------------------------------------------------------------------------
# PDF
# ----------------------------------------------------------------------------


def to_pdf(markdown_note: str, mmsi: str, out: Path | None = None) -> Path:
    """Convertit la fiche Markdown en PDF (mise en forme minimale, robuste)."""
    out = out or OUTPUTS / f"alerte_{mmsi}.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=A4,
                            title=f"Alerte navire — MMSI {mmsi}")
    styles = getSampleStyleSheet()
    story = [
        Paragraph(f"<b>ALERTE NAVIRE — MMSI {mmsi}</b>", styles["Title"]),
        Paragraph(f"Générée le {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
                  styles["Italic"]),
        Spacer(1, 14),
    ]
    for line in markdown_note.splitlines():
        if not line.strip():
            story.append(Spacer(1, 6))
            continue
        # Mini-rendu Markdown : titres, gras, listes
        if line.startswith("# "):
            story.append(Paragraph(f"<b>{line[2:]}</b>", styles["Heading1"]))
        elif line.startswith("## "):
            story.append(Paragraph(f"<b>{line[3:]}</b>", styles["Heading2"]))
        elif line.startswith("### "):
            story.append(Paragraph(f"<b>{line[4:]}</b>", styles["Heading3"]))
        elif line.startswith(("- ", "* ")):
            story.append(Paragraph(f"• {line[2:]}", styles["BodyText"]))
        else:
            story.append(Paragraph(line, styles["BodyText"]))
    doc.build(story)
    return out


def regenerate_report() -> None:
    """Hook appelé par `make report` (à câbler en J3 — Agent A)."""
    print("⚠️ `make report` — Agent A : ré-assemble rapport_generalisation.md "
          "à partir des sorties de outputs/. Voir docs/plan-3-jours.md Phase 3.")
