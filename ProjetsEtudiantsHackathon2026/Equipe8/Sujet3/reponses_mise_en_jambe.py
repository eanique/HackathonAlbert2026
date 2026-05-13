"""LIVRABLE : Mise en jambe — Sujet 3 (~20 lignes de données chacune).

Échauffement : exploration des CSV `*_small.csv`. Sections numérotées Q1→Q12
comme dans `SujetsHackathon2026/Sujet3/MiseEnJambe/README.md`.

Lancement : `python reponses_mise_en_jambe.py`
"""

from __future__ import annotations

import logging

import pandas as pd

from src.config import OUTPUTS
from src.load import load_ais, load_radio, load_ships

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
pd.set_option("display.max_columns", 12)
pd.set_option("display.width", 140)


def main() -> None:
    print("\n" + "=" * 78)
    print("  SUJET 3 — Mise en jambe (petit dataset, ~20 lignes)")
    print("=" * 78)

    ships = load_ships(small=True)
    radio = load_radio(small=True)
    ais = load_ais(small=True)

    # ---------- Q1 — Structure des fichiers --------------------------------
    print("\n### Q1 — Structure des fichiers")
    common = sorted(set(ships.columns) & set(radio.columns))
    print(f"Champs communs ships ↔ radio : {common}")
    print(f"Colonnes ships : {len(ships.columns)} | radio : {len(radio.columns)} | "
          f"ais : {len(ais.columns)}")

    # ---------- Q2 — Analyse des navires -----------------------------------
    print("\n### Q2 — Navires")
    longest = ships.loc[ships["length"].idxmax()]
    print(f"Plus long : {longest['name']} (MMSI {longest['mmsi']}) — {longest['length']} m.")
    print(f"Pavillon Panama : {int((ships['flag'] == 'Panama').sum())} navire(s).")

    # ---------- Q3 — Signatures radio --------------------------------------
    print("\n### Q3 — Signatures radio")
    print(f"Fréquence max : {radio['frequency'].max()} MHz.")
    print(f"Modulation FM : {int((radio['modulation'] == 'FM').sum())} signature(s).")

    # ---------- Q4 — AIS ---------------------------------------------------
    print("\n### Q4 — AIS")
    ais_off = ais[~ais["ais_active"].astype(bool)]
    print(f"Lignes AIS désactivé : {len(ais_off)}")
    if not ais.empty and "speed" in ais.columns:
        idx = ais["speed"].idxmax()
        print(f"Navire le plus rapide : MMSI {ais.loc[idx, 'mmsi']} — "
              f"{ais.loc[idx, 'speed']} nœuds.")

    # ---------- Q5 — Navires ↔ signatures radio ----------------------------
    print("\n### Q5 — Liens navires ↔ signatures")
    ships_with_sig = ships["mmsi"].isin(radio["mmsi"]).sum()
    print(f"Navires avec ≥ 1 signature radio : {int(ships_with_sig)}/{len(ships)}")

    # ---------- Q6 — Navires ↔ AIS -----------------------------------------
    print("\n### Q6 — Liens navires ↔ AIS")
    ships_with_ais = ships["mmsi"].isin(ais["mmsi"]).sum()
    only_ais = ships[ships["mmsi"].isin(ais["mmsi"]) & ~ships["mmsi"].isin(radio["mmsi"])]
    print(f"Navires avec ≥ 1 ligne AIS : {int(ships_with_ais)}")
    print(f"Avec AIS mais sans radio : {only_ais['mmsi'].tolist()}")

    # ---------- Q7 — Écarts de position (petit dataset, seuil 0.001°) ------
    print("\n### Q7 — Écart AIS↔radio simple (seuil 0.001°)")
    merged = ais.merge(radio[["mmsi", "location_lat", "location_lon"]],
                       on="mmsi", how="inner")
    merged["dlat"] = (merged["latitude"] - merged["location_lat"]).abs()
    merged["dlon"] = (merged["longitude"] - merged["location_lon"]).abs()
    mismatch = merged[(merged["dlat"] > 0.001) | (merged["dlon"] > 0.001)]
    print(f"Lignes avec écart > 0.001° : {len(mismatch)} "
          f"({mismatch['mmsi'].nunique()} navires)")

    # ---------- Q8 — Doublons / orphelins ----------------------------------
    print("\n### Q8 — Données manquantes / orphelins")
    no_data = ships[~ships["mmsi"].isin(radio["mmsi"]) & ~ships["mmsi"].isin(ais["mmsi"])]
    dup_mmsi = ships[ships["mmsi"].duplicated(keep=False)]
    print(f"Navires sans signature ni AIS : {no_data['mmsi'].tolist()}")
    print(f"MMSI dupliqués dans ships : {dup_mmsi['mmsi'].tolist()}")

    # ---------- Q9 — Statistiques descriptives -----------------------------
    print("\n### Q9 — Stats descriptives")
    print(ships["length"].agg(["mean", "median", "std"]).round(2).to_string())
    print(radio["frequency"].agg(["mean", "std"]).round(3).to_string())

    # ---------- Q10 — Visualisation (optionnel) ----------------------------
    print("\n### Q10 — Histogramme vitesses + barres par pavillon")
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(ais["speed"].dropna(), bins=10)
        axes[0].set_title("Histogramme vitesses (AIS)")
        axes[0].set_xlabel("nœuds")
        ships["flag"].value_counts().plot.bar(ax=axes[1], rot=30)
        axes[1].set_title("Navires par pavillon")
        fig.tight_layout()
        out = OUTPUTS / "mise_en_jambe_q10.png"
        fig.savefig(out, dpi=110)
        print(f"→ {out}")
    except Exception as exc:
        print(f"viz Q10 sautée ({exc})")

    # ---------- Q11 — Hypothèses (questions ouvertes) ----------------------
    print("\n### Q11 — Hypothèses (rédigées dans rapport_generalisation.md)")
    print("- Pourquoi un AIS désactivé ? Pannes, économies d'énergie, mais surtout "
          "contournement de sanctions / pêche illégale / espionnage. Voir Q7 large.")
    print("- Écarts position AIS↔radio : retard de mise à jour, propagation, "
          "ou usurpation (l'émetteur n'est pas où l'AIS prétend).")

    # ---------- Q12 — Améliorations possibles ------------------------------
    print("\n### Q12 — Améliorations possibles")
    print("- Champs manquants : `transmitter_id` matériel (RF fingerprinting),"
          " `direction-finding bearing` côté capteur, météo, courants.")
    print("- Automatisation : pipeline `python reponses_generalisation.py` "
          "(cf. main du livrable Généralisation).")

    print("\n" + "=" * 78)
    print("  Mise en jambe ✅ — passer maintenant à `python reponses_generalisation.py`")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
