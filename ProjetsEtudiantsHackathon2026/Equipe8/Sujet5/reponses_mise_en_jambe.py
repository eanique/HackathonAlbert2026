"""Livrable Sujet 5 — Mise en jambe (Q1 → Q10).

Échauffement sur les `*_small.csv` / `annotations_small.json`. Taxonomie
**différente** (8 classes) — ne pas mélanger avec la Généralisation (13 classes).

Lancer : `python reponses_mise_en_jambe.py`
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src import config as C


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def main() -> None:
    df = pd.read_csv(C.F_MEJ_IMAGES)
    with C.F_MEJ_ANNOT.open() as f:
        coco = json.load(f)

    # --- P1 : Métadonnées --------------------------------------------------
    banner("P1 — Exploration des métadonnées")

    print("\n[Q1] Structure")
    print(f"  Champs : {list(df.columns)}")
    print(f"  Images avec > 1 navire : {(df['num_ships'] > 1).sum()} / {len(df)}")

    print("\n[Q2] Zones géographiques")
    top3 = df["zone_name"].value_counts().head(3)
    for k, v in top3.items():
        print(f"  {k:<40} {v}")
    # Le small CSV ne contient pas `risk_level` → heuristique sur les noms de zone
    # (l'énoncé donne explicitement Malacca + Golfe Persique comme exemples).
    risky_keywords = ("Malacca", "Persique", "Ormuz", "Chine", "Suez", "Norfolk", "Sébastopol")
    n_high = df["zone_name"].fillna("").str.contains("|".join(risky_keywords)).sum()
    print(f"  Images en zones à risque élevé (heuristique mots-clés) : {n_high}")

    print("\n[Q3] Résolution & qualité")
    top_res = df["resolution"].value_counts().idxmax()
    mean_cloud = df["cloud_cover"].mean()
    print(f"  Résolution la plus fréquente : {top_res}")
    print(f"  Couverture nuageuse moyenne : {mean_cloud:.1f} %")

    # --- P2 : Annotations --------------------------------------------------
    banner("P2 — Annotations COCO")

    cats = {c["id"]: c["name"] for c in coco["categories"]}
    df_ann = pd.DataFrame(coco["annotations"])

    print("\n[Q4] Structure COCO")
    print(f"  Catégories : {len(cats)} -> {list(cats.values())}")
    print(f"  Annotations totales : {len(df_ann)}")
    print(f"  Images référencées : {len(coco['images'])}")

    print("\n[Q5] Répartition des types")
    df_ann["category_name"] = df_ann["category_id"].map(cats)
    top_type = df_ann["category_name"].value_counts().idxmax()
    print(f"  Type le plus fréquent : {top_type}")
    militaires_mej = {"Frégate", "Destroyer", "Porte-avions"}
    n_mil = df_ann["category_name"].isin(militaires_mej).sum()
    print(f"  Navires militaires (énoncé MEJ) : {n_mil}")

    print("\n[Q6] Taille des bounding boxes")
    # bbox COCO en normalisé [0,1] ici aussi ; aire en % de l'image = w_n * h_n * 100
    df_ann["bbox_area_pct"] = df_ann["bbox"].map(lambda b: b[2] * b[3] * 100)
    mean_pct = df_ann["bbox_area_pct"].mean()
    print(f"  Taille moyenne bbox (% de l'image) : {mean_pct:.3f} %")
    biggest = (
        df_ann.groupby("category_name")["bbox_area_pct"].mean().sort_values(ascending=False)
    )
    print(f"  Type avec les plus grandes bbox : {biggest.index[0]} ({biggest.iloc[0]:.3f} %)")

    # --- P3 : Visualisation (Q7) — skip (pas d'images disponibles) ---------
    banner("P3 — Visualisation")
    print("\n[Q7-Q8] Affichage des bbox / comptage manuel")
    print("  ⚠️ Les images `satellite_*.jpg` référencées N'EXISTENT PAS dans le repo.")
    print("     Hypothèse : on documente la méthode, et on l'applique en Piste B")
    print("     sur des scènes Sentinel-1/2 réelles téléchargées via Copernicus.")
    print("     (cf. sujet5/src/hunt.py en Phase 2)")

    # --- P4 : Questions ouvertes ------------------------------------------
    banner("P4 — Questions ouvertes")

    print("\n[Q9] Limites des annotations")
    print(
        "  • Le champ `confidence` est présent → ce ne sont PAS des annotations manuelles\n"
        "    mais des sorties de modèle relabelisées en GT (data leakage potentiel).\n"
        "  • Taxonomie MEJ (8 classes) ≠ Généralisation (13 classes) : risque de confusion.\n"
        "  • bbox normalisée [0,1] alors que `area` est en pixels → incohérent, recalculer.\n"
        "  • Couple (source, resolution) physiquement impossible (Sentinel-2 jamais 1 m).\n"
        "  Amélioration : annotation humaine via CVAT / Roboflow ; review croisée 2 annotateurs."
    )

    print("\n[Q10] Améliorations détection automatique")
    print(
        "  • Champs à ajouter aux métadonnées : `sensor_mode` (SAR/optique), `incidence_angle`,\n"
        "    `pixel_spacing_m_real`, `tide_level`, `sea_state`, `polarisation` (VV/VH pour SAR).\n"
        "  • Automatisation : pré-détection par modèle (YOLO/Faster R-CNN), revue humaine,\n"
        "    consensus multi-annotateurs ; pipeline `pre-label → review → merge` (Roboflow / CVAT)."
    )


if __name__ == "__main__":
    main()
