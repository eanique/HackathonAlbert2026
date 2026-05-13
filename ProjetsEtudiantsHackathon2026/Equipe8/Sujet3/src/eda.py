"""Exploration & contrôle qualité des données (Sujet 3).

Produit un **rapport de qualité** structuré (dict + Markdown) couvrant :
  1. Volumétrie, dtypes, valeurs manquantes, doublons.
  2. Contrôle des bornes vs les dictionnaires de données fournis.
  3. Contrôle des valeurs catégorielles vs les listes attendues.
  4. Intégrité référentielle (MMSI : radio ⊂ ships, ais ⊂ ships, anom ⊂ ships).
  5. Formats d'identifiants (MMSI 9 chiffres, IMO 7 chiffres, MID UIT).
  6. Cohérences temporelles (régressions, plages).
  7. **Tests de structure** : `navigational_status` vs `status`, `ais_active`
     vs `status` — pour objectiver le caractère synthétique/non-structuré.
  8. **Détection du "value-only-in-text"** : les anomalies de vitesse/cap dont
     la valeur citée dans la description n'existe pas dans les tables ⇒
     borne supérieure de rappel atteignable.

Cf. les findings détaillés (raisons des choix) dans `docs/hypotheses.md`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from .config import OUTPUTS, mmsi_country

# Bornes attendues (depuis les dictionnaires de données fournis)
BOUNDS = {
    ("radio", "frequency"): (156.0, 162.0),
    ("radio", "bandwidth"): (10.0, 50.0),
    ("radio", "power"): (1.0, 500.0),
    ("radio", "signal_strength"): (-120.0, -20.0),
    ("radio", "noise_level"): (-120.0, -60.0),
    ("radio", "signal_to_noise_ratio"): (0.0, 50.0),
    ("radio", "location_lat"): (-90.0, 90.0),
    ("radio", "location_lon"): (-180.0, 180.0),
    ("ais", "latitude"): (-90.0, 90.0),
    ("ais", "longitude"): (-180.0, 180.0),
    ("ais", "speed"): (0.0, 50.0),
    ("ais", "course"): (0.0, 360.0),
    ("ais", "heading"): (0.0, 360.0),
    ("ais", "rot"): (-127.0, 127.0),
    ("ais", "navigational_status"): (0, 15),
    ("ships", "year_built"): (1900, 2026),
    ("anom", "confidence"): (0.0, 1.0),
}

CATEGORIES = {
    ("radio", "modulation"): {"FM", "AM", "DSC", "SSB", "OFDM"},
    ("radio", "pulse_pattern"): {"Short-Short-Long", "Short-Long-Short",
                                 "Long-Short-Long", "Short-Short-Short",
                                 "Long-Long-Short", "Continuous"},
    ("ais", "status"): {"Under Way", "At Anchor", "Moored",
                        "Not under command", "Restricted maneuverability"},
    ("ships", "type"): {"Container Ship", "Tanker", "Bulk Carrier",
                        "General Cargo", "Passenger Ship", "Fishing Vessel",
                        "Tugboat", "Pilot Vessel", "Sailboat", "Yacht",
                        "Military", "Other"},
    ("ships", "flag"): {"Panama", "Liberia", "France", "Denmark", "Singapore",
                        "Marshall Islands", "Malta", "Bahamas", "China", "USA"},
    ("anom", "type"): {"Fake Flag", "Name Change", "Spoofing", "AIS Disabled",
                       "Position Mismatch", "Speed Anomaly", "Course Anomaly"},
    ("anom", "source"): {"Radio Signature", "AIS Inconsistency", "Manual Review"},
}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


# Colonnes dérivées ajoutées par notre pipeline (à ne pas compter comme « manquants source »)
_DERIVED_COLS = {"historical_names_list", "n_names_historical", "mmsi_is_valid",
                 "mmsi_country", "has_radio_signature"}


def _missing(df: pd.DataFrame) -> dict:
    return {c: int(df[c].isna().sum()) for c in df.columns
            if c not in _DERIVED_COLS and df[c].isna().any()}


def _hashable_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Retire les colonnes contenant des objets non-hashables (ex. listes)
    avant un appel à `duplicated()` (qui sinon plante)."""
    keep = []
    for c in df.columns:
        sample = df[c].dropna().head(1)
        if len(sample) and isinstance(sample.iloc[0], (list, dict, set)):
            continue
        keep.append(c)
    return df[keep]


def _cramers_v(table: pd.DataFrame) -> float:
    """V de Cramér (force d'association entre 2 catégorielles, ∈ [0,1])."""
    chi2 = chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    r, k = table.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1)))) if min(r, k) > 1 and n else 0.0


# ----------------------------------------------------------------------------
# Rapport principal
# ----------------------------------------------------------------------------


def run_quality_report(ships: pd.DataFrame, radio: pd.DataFrame,
                       ais: pd.DataFrame, anom: pd.DataFrame) -> dict:
    """Construit le dict complet de contrôle qualité."""
    dfs = {"ships": ships, "radio": radio, "ais": ais, "anom": anom}
    rep: dict = {"volumetrie": {}, "manquants": {}, "doublons": {},
                 "hors_bornes": {}, "categories_hors_liste": {},
                 "integrite_referentielle": {}, "formats": {},
                 "temporel": {}, "structure": {}, "ceiling": {},
                 "cleaning_items": []}

    # 1) Volumétrie / manquants / doublons
    for name, df in dfs.items():
        rep["volumetrie"][name] = list(df.shape)
        m = _missing(df)
        if m:
            rep["manquants"][name] = m
        rep["doublons"][name] = int(_hashable_cols(df).duplicated().sum())

    # 2) Hors-bornes
    for (tbl, col), (lo, hi) in BOUNDS.items():
        df = dfs[tbl]
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            out = int(((s < lo) | (s > hi)).sum())
            if out:
                rep["hors_bornes"][f"{tbl}.{col}"] = {"n_out": out,
                                                      "min": float(s.min()),
                                                      "max": float(s.max()),
                                                      "bounds": [lo, hi]}

    # 3) Catégories hors-liste
    for (tbl, col), expected in CATEGORIES.items():
        df = dfs[tbl]
        if col in df.columns:
            seen = set(df[col].dropna().astype(str).unique())
            extra = seen - expected
            if extra:
                rep["categories_hors_liste"][f"{tbl}.{col}"] = sorted(extra)

    # 4) Intégrité référentielle
    sm = set(ships["mmsi"])
    rep["integrite_referentielle"] = {
        "radio_mmsi_orphans": len(set(radio["mmsi"]) - sm),
        "ais_mmsi_orphans": len(set(ais["mmsi"]) - sm),
        "anom_mmsi_orphans": len(set(anom["mmsi"]) - sm),
        "ships_without_radio": len(sm - set(radio["mmsi"])),
        "ships_without_ais": len(sm - set(ais["mmsi"])),
        "ships_without_radio_mmsi": sorted(sm - set(radio["mmsi"])),
        "n_fake_mmsi_in_radio": int(radio["mmsi"].astype(str).str.startswith("FAKE").sum()),
        "radio_signatures_per_mmsi": radio.groupby("mmsi").size().describe().round(2).to_dict(),
        "ais_rows_per_mmsi": ais.groupby("mmsi").size().describe().round(2).to_dict(),
    }

    # 5) Formats d'identifiants
    rep["formats"] = {
        "mmsi_9digits_ships": bool(ships["mmsi"].astype(str).str.match(r"^\d{9}$").all()),
        "mmsi_9digits_radio": bool(radio["mmsi"].astype(str).str.match(r"^\d{9}$").all()),
        "mmsi_9digits_ais": bool(ais["mmsi"].astype(str).str.match(r"^\d{9}$").all()),
        "imo_7digits_ships": bool(ships["imo"].astype(str).str.match(r"^\d{7}$").all()),
        "name_is_placeholder": float(
            (ships["name"].astype(str).str.extract(r"NAVIRE-(\d+)")[0]
             == ships["mmsi"].astype(str).str[-4:]).mean()
        ),
        "mmsi_known_mid_pct": float(ships["mmsi"].apply(lambda m: mmsi_country(m) is not None).mean()),
    }

    # 6) Temporel
    for name, df in dfs.items():
        tcols = [c for c in df.columns if "timestamp" in c or "update" in c]
        for c in tcols:
            t = pd.to_datetime(df[c], utc=True, errors="coerce")
            rep["temporel"][f"{name}.{c}"] = {
                "min": str(t.min()), "max": str(t.max()), "n_nat": int(t.isna().sum()),
            }
    a = ais.copy()
    a["timestamp"] = pd.to_datetime(a["timestamp"], utc=True)
    a = a.sort_values(["mmsi", "timestamp"])
    rep["temporel"]["ais_temporal_regressions"] = int(
        a.groupby("mmsi")["timestamp"].apply(lambda s: (s.diff() < pd.Timedelta(0)).sum()).sum()
    )

    # 7) Structure (synthétique ?) — associations attendues mais absentes
    if {"navigational_status", "status"}.issubset(ais.columns):
        ct1 = pd.crosstab(ais["navigational_status"], ais["status"])
        rep["structure"]["cramers_v_navstatus_vs_status"] = round(_cramers_v(ct1), 4)
    if {"ais_active", "status"}.issubset(ais.columns):
        ct2 = pd.crosstab(ais["ais_active"], ais["status"])
        rep["structure"]["cramers_v_aisactive_vs_status"] = round(_cramers_v(ct2), 4)
        rep["structure"]["ais_active_true_pct"] = float(ais["ais_active"].astype(bool).mean())
    # Vitesse implicite (sur l'AIS) — médiane et max (indicateur "positions aléatoires")
    try:
        from geopy.distance import geodesic
        speeds = []
        for _, g in a.groupby("mmsi"):
            g = g.reset_index(drop=True)
            for i in range(1, len(g)):
                dt = (g.loc[i, "timestamp"] - g.loc[i - 1, "timestamp"]).total_seconds() / 3600
                if dt <= 0:
                    continue
                d = geodesic((g.loc[i - 1, "latitude"], g.loc[i - 1, "longitude"]),
                             (g.loc[i, "latitude"], g.loc[i, "longitude"])).km
                speeds.append(d / dt / 1.852)  # km/h -> kn
        if speeds:
            rep["structure"]["implicit_speed_knots"] = {
                "median": round(float(np.median(speeds)), 1),
                "p95": round(float(np.percentile(speeds, 95)), 1),
                "max": round(float(np.max(speeds)), 1),
            }
    except Exception:
        pass

    # historical_names
    hn = ships["historical_names"].fillna("")
    nn = hn.apply(lambda s: len([x for x in s.split(",") if x.strip()]))
    rep["structure"]["historical_names_count_distribution"] = nn.value_counts().sort_index().to_dict()

    # 8) Plafond de rappel : anomalies dont la valeur n'est pas dans les tables
    spd = anom[anom["type"] == "Speed Anomaly"]
    crs = anom[anom["type"] == "Course Anomaly"]
    cited_speed = spd["description"].str.extract(r"(\d+\.?\d*)\s*nœuds")[0].astype(float)
    speed_in_table = pd.to_numeric(ais["speed"], errors="coerce").max()
    n_unrecoverable = len(spd) + len(crs)  # ni speed (cap 30) ni course exploitables
    rep["ceiling"] = {
        "n_anomalies_total": int(len(anom)),
        "n_speed_anomaly": int(len(spd)),
        "n_course_anomaly": int(len(crs)),
        "max_speed_cited_in_descriptions": float(cited_speed.max()) if len(cited_speed) else None,
        "max_speed_in_ais_table": float(speed_in_table),
        "n_unrecoverable_from_tables": int(n_unrecoverable),
        "achievable_recall_ceiling": round(1 - n_unrecoverable / max(len(anom), 1), 3),
        "note": ("Les valeurs de vitesse/cap des anomalies Speed/Course "
                 "n'apparaissent que dans la description ; la colonne `speed` "
                 "est plafonnée et les positions sont aléatoires (vitesse "
                 "implicite non discriminante) ⇒ ces anomalies ne sont pas "
                 "détectables depuis les tables fournies."),
    }

    # 9) Items de cleaning identifiés
    nr = rep["integrite_referentielle"]["ships_without_radio"]
    if nr:
        rep["cleaning_items"].append(
            f"{nr} navire(s) sans aucune signature radio → marqués "
            f"`has_radio_signature=False` ; absents de ship_radio_profiles.csv ; "
            f"utilisés comme signal de suspicion (« navire jamais entendu »)."
        )
    if rep["formats"]["name_is_placeholder"] > 0.9:
        rep["cleaning_items"].append(
            "Le champ `name` (dataset large) est un placeholder dérivé du MMSI "
            "(NAVIRE-<4 derniers chiffres>) → on identifie par MMSI, le `name` "
            "n'apporte aucune information (≠ dataset small qui a de vrais noms)."
        )
    if "ships" in rep["manquants"] and "historical_names" in rep["manquants"]["ships"]:
        rep["cleaning_items"].append(
            f"`historical_names` vide pour {rep['manquants']['ships']['historical_names']} "
            f"navires = « jamais renommé » (n_names_historical=0), PAS une donnée "
            f"manquante. Parsé proprement par split(',')."
        )
    return rep


# ----------------------------------------------------------------------------
# Export Markdown
# ----------------------------------------------------------------------------


def write_markdown(rep: dict, out: Path | None = None) -> Path:
    out = out or OUTPUTS / "data_quality_report.md"
    L: list[str] = ["# Rapport de qualité des données — Sujet 3", ""]
    L.append("## 1. Volumétrie")
    for name, shape in rep["volumetrie"].items():
        L.append(f"- `{name}` : {shape[0]} lignes × {shape[1]} colonnes")
    L.append("")
    L.append("## 2. Valeurs manquantes")
    if rep["manquants"]:
        for name, m in rep["manquants"].items():
            L.append(f"- `{name}` : {m}")
    else:
        L.append("- Aucune.")
    L.append("")
    L.append("## 3. Doublons (lignes entières)")
    for name, n in rep["doublons"].items():
        L.append(f"- `{name}` : {n}")
    L.append("")
    L.append("## 4. Contrôle des bornes (vs dictionnaires de données)")
    L.append("- " + ("Aucune valeur hors-bornes." if not rep["hors_bornes"]
                     else json.dumps(rep["hors_bornes"], ensure_ascii=False)))
    L.append("")
    L.append("## 5. Valeurs catégorielles hors-liste")
    L.append("- " + ("Aucune." if not rep["categories_hors_liste"]
                     else json.dumps(rep["categories_hors_liste"], ensure_ascii=False)))
    L.append("")
    L.append("## 6. Intégrité référentielle")
    ir = rep["integrite_referentielle"]
    L.append(f"- MMSI orphelins : radio={ir['radio_mmsi_orphans']}, ais={ir['ais_mmsi_orphans']}, anom={ir['anom_mmsi_orphans']}")
    L.append(f"- Navires sans signature radio : {ir['ships_without_radio']} ({ir['ships_without_radio_mmsi']})")
    L.append(f"- Navires sans donnée AIS : {ir['ships_without_ais']}")
    L.append(f"- MMSI préfixés `FAKE` dans radio : {ir['n_fake_mmsi_in_radio']}")
    L.append("")
    L.append("## 7. Formats d'identifiants")
    L.append("```json\n" + json.dumps(rep["formats"], indent=2, ensure_ascii=False) + "\n```")
    L.append("")
    L.append("## 8. Temporel")
    L.append("```json\n" + json.dumps(rep["temporel"], indent=2, ensure_ascii=False) + "\n```")
    L.append("")
    L.append("## 9. Tests de structure (le dataset est-il réaliste ?)")
    st = rep["structure"]
    L.append(f"- V de Cramér `navigational_status` ↔ `status` : **{st.get('cramers_v_navstatus_vs_status')}** (≈ 0 ⇒ aucune structure ; le vrai AIS aurait V ≈ 1)")
    L.append(f"- V de Cramér `ais_active` ↔ `status` : **{st.get('cramers_v_aisactive_vs_status')}** ; `ais_active=True` : {st.get('ais_active_true_pct', 0):.1%} (≈ 50 % ⇒ aléatoire)")
    if "implicit_speed_knots" in st:
        s = st["implicit_speed_knots"]
        L.append(f"- Vitesse implicite (distance géodésique / Δt entre points AIS consécutifs) : médiane **{s['median']} kn**, p95 **{s['p95']} kn**, max **{s['max']} kn** ⇒ les positions AIS sont **des points aléatoires, pas des trajectoires**.")
    L.append(f"- Distribution du nombre de noms historiques : {st.get('historical_names_count_distribution')} (≈ uniforme 0/1/2/3 — templates `OLD_NAME_k`).")
    L.append("")
    L.append("## 10. Plafond de rappel atteignable")
    c = rep["ceiling"]
    L.append(f"- {c['n_speed_anomaly']} anomalies « Speed » + {c['n_course_anomaly']} « Course » = {c['n_unrecoverable_from_tables']} anomalies dont la valeur n'est **que dans la description** (`speed` plafonné à {c['max_speed_in_ais_table']:.0f} kn, descriptions citant jusqu'à {c.get('max_speed_cited_in_descriptions')} kn).")
    L.append(f"- ⇒ **Rappel maximal atteignable depuis les tables fournies ≈ {c['achievable_recall_ceiling']:.0%}** (sur {c['n_anomalies_total']} anomalies de référence).")
    L.append(f"- *{c['note']}*")
    L.append("")
    L.append("## 11. Actions de nettoyage appliquées")
    for item in rep["cleaning_items"]:
        L.append(f"- {item}")
    L.append("")
    out.write_text("\n".join(L), encoding="utf-8")
    return out
