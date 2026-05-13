"""Évaluation par type d'anomalie — « qu'est-ce qu'on attrape, qu'est-ce qu'on rate ».

Pour chaque type de `anomalies_large.csv` (Fake Flag, Name Change, Spoofing,
AIS Disabled, Position Mismatch, Speed Anomaly, Course Anomaly), on calcule le
**rappel** du (des) détecteur(s) correspondant(s) :

  Fake Flag        → detect_fake_flag (EllipticEnvelope/MCD)
  Name Change      → detect_name_change (>2 noms historiques)
  Spoofing         → detect_orphan_signatures + spoofing_rules
  AIS Disabled     → detect_ais_disabled (> 24 h consécutives)
  Position Mismatch→ detect_position_mismatch (> 1 km)
  Speed Anomaly    → (non récupérable des tables — cf. EDA, plafond de rappel)
  Course Anomaly   → (non récupérable des tables)

Produit un tableau + un graphique en barres `outputs/recall_by_type.png`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import OUTPUTS


def recall_by_anomaly_type(anomalies: pd.DataFrame,
                            detected_mmsi_by_type: dict[str, set],
                            ) -> pd.DataFrame:
    """
    Args:
        anomalies : `anomalies_large.csv`.
        detected_mmsi_by_type : pour un type d'anomalie donné, l'ensemble des
            MMSI flaggés par le(s) détecteur(s) correspondant(s).
            Clés attendues : "Fake Flag", "Name Change", "Spoofing",
            "AIS Disabled", "Position Mismatch". (Speed/Course non couverts.)

    Returns:
        DataFrame [type, n_truth, n_detected_overlap, recall, recoverable, note]
    """
    rows: list[dict] = []
    for atype, grp in anomalies.groupby("type"):
        truth = set(grp["mmsi"])
        recoverable = atype in detected_mmsi_by_type
        if recoverable:
            detected = detected_mmsi_by_type[atype]
            overlap = truth & detected
            recall = len(overlap) / max(len(truth), 1)
            note = f"détecteur : {len(detected)} MMSI flaggés au total"
        else:
            overlap = set()
            recall = 0.0
            note = "non récupérable des tables (valeur seulement dans la description)"
        rows.append({
            "type": atype,
            "n_truth": len(truth),
            "n_overlap": len(overlap),
            "recall": round(recall, 3),
            "recoverable": recoverable,
            "note": note,
        })
    df = pd.DataFrame(rows).sort_values("recall", ascending=False)
    return df


def plot_recall_by_type(df: pd.DataFrame, out: Path | None = None) -> Path:
    out = out or OUTPUTS / "recall_by_type.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#0a6" if r else "#c33" for r in df["recoverable"]]
    bars = ax.barh(df["type"], df["recall"], color=colors)
    for bar, n_t, n_o in zip(bars, df["n_truth"], df["n_overlap"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{n_o}/{n_t}", va="center", fontsize=9)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Rappel du détecteur dédié")
    ax.set_title("Rappel par type d'anomalie  (vert = récupérable des tables · rouge = non récupérable)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def pretty_print(df: pd.DataFrame) -> None:
    print("  Rappel par type d'anomalie (vs anomalies_large.csv) :")
    print(df[["type", "n_truth", "n_overlap", "recall", "recoverable"]].to_string(index=False))
    rec = df[df["recoverable"]]
    overall_recoverable = rec["n_truth"].sum()
    overall_caught = rec["n_overlap"].sum()
    total = df["n_truth"].sum()
    print(f"  → Sur les {overall_recoverable}/{total} anomalies *récupérables*, "
          f"on en attrape {overall_caught} (rappel {overall_caught / max(overall_recoverable,1):.0%}).")
    print(f"  → Rappel global vs les {total} anomalies : {overall_caught / max(total,1):.0%} "
          f"(plafond théorique : {overall_recoverable / max(total,1):.0%}).")
