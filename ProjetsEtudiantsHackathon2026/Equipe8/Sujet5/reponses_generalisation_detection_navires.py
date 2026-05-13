"""Livrable Sujet 5 — Généralisation (Q1 → Q18).

Script auto-suffisant : `python reponses_generalisation_detection_navires.py` exécute
toutes les questions de bout en bout, lit `data/raw/`, écrit `data/processed/` + `outputs/`.

Sections :
    P1  Q1-Q3   Exploration (DONE)
    P2  Q4-Q6   Détection YOLO sur dataset substitué         (cf. src/p2_detection.py)
    P3  Q7-Q9   Classification binaire/multi-classes         (cf. src/p3_classify.py)
    P4  Q10-Q12 Géospatial + temporel + anomalies (DONE)
    P5  Q13-Q15 Pipeline + ONNX + API externes               (cf. src/p5_pipeline.py)
    P6  Q16-Q18 Benchmark détecteurs/classifieurs + rapport  (cf. src/p6_benchmark.py)

À J1 / Phase 1, seuls P1 + P4 sont implémentés (livrable minimal viable).
Les autres sections s'ajouteront en Phase 2/3 sans casser la signature.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src import config as C
from src import p1_explore, p3_classify, p4_geospatial
from src.maps import anomalies_map, detections_map

# Imports lourds (PyTorch + STAC) protégés : si lib absente, on saute proprement.
try:
    from src import hunt as _hunt
    _HAS_HUNT = True
except ImportError:
    _HAS_HUNT = False
try:
    from src import p2_detection as _p2
    _HAS_YOLO = True
except ImportError:
    _HAS_YOLO = False
try:
    from src import p5_pipeline, p6_benchmark, intel_report
    _HAS_PHASE3 = True
except ImportError:
    _HAS_PHASE3 = False


def banner(title: str) -> None:
    print("\n" + "=" * 76)
    print(f"  {title}")
    print("=" * 76)


def save_table(df: pd.DataFrame, name: str) -> Path:
    """Sauvegarde tabulaire dans data/processed/ + écho terminal."""
    path = C.DATA_PROCESSED / name
    df.to_csv(path, index=False)
    print(f"  → écrit {path.relative_to(C.ROOT)}  ({len(df)} lignes)")
    return path


def main() -> dict:
    results: dict = {}

    # ------------------------------------------------------------------
    # P1 — Exploration des données fournies
    banner("P1 — Exploration (Q1-Q3)")
    p1 = p1_explore.run_p1()
    results["P1"] = {k: v for k, v in p1.items() if k != "_data"}

    print("\n[Q1] Métadonnées des images")
    q1 = p1["Q1"]
    print(f"  Top 3 sources : {q1['top3_sources']}")
    print(f"  Résolution moyenne : {q1['mean_resolution_m']:.2f} m")
    print(
        f"  Images avec cloud_cover > 30 % : {q1['n_cloud_gt_30']} "
        f"({q1['pct_cloud_gt_30']:.1f} %)"
    )
    print(f"  ⚠️ {q1['_warning_source_resolution_incoherence']}")

    print("\n[Q2] Annotations")
    q2 = p1["Q2"]
    print(
        f"  Navires militaires annotés : {q2['n_militaires']} / "
        f"{q2['n_total_annotations']} ({q2['pct_militaires']} %)"
    )
    print("  Répartition par type (%) :")
    for k, v in sorted(q2["repartition_pct"].items(), key=lambda kv: -kv[1]):
        print(f"    {k:<28} {v:>6.2f} %")
    save_table(
        q2["bbox_by_type"].reset_index(),
        "q2_bbox_by_type.csv",
    )

    print("\n[Q3] Fusion images × détections")
    q3 = p1["Q3"]
    print(f"  Détections moyennes / image : {q3['mean_det_per_image']:.2f}")
    print(f"  Médiane : {q3['median_det_per_image']:.1f}")
    print(
        f"  Image avec le plus de détections : {q3['max_det_image_id']} "
        f"({q3['max_det_count']} détections)"
    )
    save_table(q3["df_merged"], "q3_merged.csv")

    # ------------------------------------------------------------------
    # P2 — Détection (Q4-Q6) : sanity YOLOv8n + interface fine-tune
    banner("P2 — Détection YOLOv8 (Q4-Q6)")
    if _HAS_YOLO:
        # Image sanity : un fichier déjà téléchargé en data/processed/ ou rien
        sanity_img = C.DATA_PROCESSED / "_sanity_input.jpg"
        if not sanity_img.exists():
            print("  Sanity image absente — `python -m src.p2_detection` à part.")
            results["P2"] = {"status": "sanity_skipped (no image)"}
        else:
            print("  Sanity YOLOv8n pré-entraîné COCO sur image de test :")
            s = _p2.sanity_yolov8n(sanity_img)
            print(f"    device       : {s['device']}  (cuda=False, mps=Apple Silicon)")
            print(f"    n détections : {s['n_detections']}")
            print(f"    image annotée : {s['annotated_image']}")
            print(f"    ⚠️ {s['_note']}")
            results["P2"] = {
                "sanity": {
                    "device": s["device"],
                    "n_detections": s["n_detections"],
                    "annotated_image": s["annotated_image"],
                }
            }
        print(
            "\n  Q4-Q6 fine-tune : `src.p2_detection.train_yolo_on_substituted(data_yaml)`\n"
            "    dataset substitué : xView3-SAR (SAR) ou Airbus Ship (optique)\n"
            "    cible mAP@0.5 ≥ 0.60 sur test split 15 %  [hypotheses.md §B Q4]"
        )
    else:
        print("  ultralytics non installé — `uv pip install ultralytics torch torchvision`.")
        results["P2"] = {"status": "skipped (ultralytics absent)"}

    # ------------------------------------------------------------------
    # P3 — Classification militaire/civil (Q7 fait en Phase 2 ; Q8/Q9 = lots suivants)
    banner("P3 — Classification (Q7-Q9)")
    p3 = p3_classify.run_p3_q7()
    q7 = p3["Q7"]
    results["P3"] = {"Q7": {k: v for k, v in q7.items() if k != "df_metrics"}}

    print("\n[Q7] Binaire militaire/civil — 5-fold CV (3 classifieurs, métriques classe militaire) :")
    print(q7["df_metrics"].to_string(index=False))
    print(
        f"\n  Meilleur RAPPEL [Cours 4 §8, coût asymétrique] : "
        f"{q7['best_recall_clf']} -> {q7['best_recall']:.3f}"
    )
    print(f"  Meilleur F1     : {q7['best_f1_clf']} -> {q7['best_f1']:.3f}")
    if q7["feature_importance"]:
        print("\n  Top 5 features (RandomForest) :")
        for k, v in list(q7["feature_importance"].items())[:5]:
            print(f"    {k:<25} {v:.4f}")
    print(f"\n  Sorties : {q7['files']['metrics_csv']}, {q7['files']['roc_png']}")
    print(f"  ⚠️ {q7['_note']}")
    # AUC autour de 0.5 sur ce CSV synthétique = pas de signal dans les features bbox/zone
    # → justifie le passage aux embeddings ResNet50 en Q8 (à venir).
    if q7["df_metrics"]["roc_auc"].max() < 0.6:
        print(
            "\n  📝 FINDING : AUC ≈ 0.5 sur tous les classifieurs → les features tabulaires\n"
            "     (bbox + position + zone) n'ont aucun signal sur `is_military` dans ce CSV\n"
            "     synthétique. Justification empirique pour passer à Q8 (embeddings ResNet50\n"
            "     sur HRSC2016) qui exploite le contenu image, pas seulement les métadonnées."
        )

    print("\n  Q8 multi-classes (ResNet50 + HRSC2016) : Phase 2 suite — src/p3_classify.py")
    print("  Q9 t-SNE des erreurs                  : Phase 2 suite — src/p3_classify.py")

    # ------------------------------------------------------------------
    # P4 — Géospatial + temporel + anomalies
    banner("P4 — Géospatial (Q10-Q12)")
    df_det = p1["_data"]["df_det"]
    p4 = p4_geospatial.run_p4(df_det)
    results["P4"] = {
        "Q10": p4["Q10"],
        "Q11_peaks": p4["Q11"]["peaks"],
        "Q12_counts": {
            "n_type_a_mil_hors_zone": p4["Q12"]["n_type_a_mil_hors_zone"],
            "n_type_b_civil_en_zone_critique": p4["Q12"]["n_type_b_civil_en_zone_critique"],
        },
    }

    print("\n[Q10] Détections en zones militaires")
    q10 = p4["Q10"]
    print(
        f"  Militaires en zone : {q10['n_militaires_en_zone']} / "
        f"{q10['n_militaires_total']} ({q10['pct_militaires_en_zone']} %)"
    )
    print(f"  ⚠️ {q10['_note']}")
    save_table(
        p4["df_det_with_zones"],
        "q10_detections_with_zones.csv",
    )

    print("\n[Q11] Évolution temporelle par zone — pics > µ+2σ :")
    for zone, dates in p4["Q11"]["peaks"].items():
        if dates:
            print(f"    {zone:<40} pics : {dates}")
    if isinstance(p4["Q11"].get("series"), pd.DataFrame):
        save_table(
            p4["Q11"]["series"].reset_index(),
            "q11_temporal_series.csv",
        )

    print("\n[Q12] Anomalies géospatiales")
    q12 = p4["Q12"]
    print(f"  Type A (militaire hors zone) : {q12['n_type_a_mil_hors_zone']}")
    print(
        f"  Type B (civil en zone High/Critical) : "
        f"{q12['n_type_b_civil_en_zone_critique']}"
    )
    save_table(q12["df_type_a"], "q12_anomalies_type_a.csv")
    save_table(q12["df_type_b"], "q12_anomalies_type_b.csv")

    # Cartes folium
    print("\n  Génération des cartes folium…")
    map_main = C.OUTPUTS / "detections_carte.html"
    detections_map(
        p4["df_det_with_zones"], p4["df_zones"], out_path=map_main
    )
    print(f"  → écrit {map_main.relative_to(C.ROOT)}")

    map_anom = C.OUTPUTS / "anomalies_carte.html"
    anomalies_map(
        q12["df_type_a"], q12["df_type_b"], p4["df_zones"], out_path=map_anom
    )
    print(f"  → écrit {map_anom.relative_to(C.ROOT)}")

    # ------------------------------------------------------------------
    # PISTE B — La vraie chasse + croisement AIS (Leviers L1 + L2)
    banner("PISTE B — La vraie chasse (Levier L1) + Navires sombres (Levier L2)")
    if _HAS_HUNT:
        print("  Query Sentinel-1 RTC via Planetary Computer (sans inscription)...")
        df_scenes = _hunt.survey_all_bases()
        results["PISTE_B"] = {
            "n_scenes_total": len(df_scenes),
            "scenes_par_base": (
                df_scenes.groupby("base").size().to_dict() if not df_scenes.empty else {}
            ),
        }
        # Chasse réelle (download + YOLO) gated par variable d'env (long).
        # `HUNT_REAL=1 python reponses_*.py` -> télécharge + infère sur Toulon S2.
        import os as _os
        if _os.getenv("HUNT_REAL") == "1":
            print("\n  [HUNT_REAL=1] Chasse réelle Toulon (Sentinel-2 optique)...")
            try:
                df_t = _hunt.hunt_base(
                    "Toulon (FR)", max_scenes=1, collection=_hunt._COLLECTION_S2
                )
                n = 0 if df_t is None or df_t.empty else len(df_t)
                results["PISTE_B"]["chasse_toulon_s2"] = {
                    "n_detections_yolo_coco": n,
                    "note": (
                        "YOLO COCO ne reconnaît PAS les navires en vue zénithale "
                        "(classifié à tort bear/sheep/person). Fine-tune xView3 / "
                        "Airbus Ship sur Colab T4 = prochaine étape."
                    ),
                }
            except Exception as e:  # noqa: BLE001
                results["PISTE_B"]["chasse_toulon_s2"] = {"error": str(e)}
                print(f"    sauté : {type(e).__name__}: {e}")
        else:
            print(
                "\n  Chasse réelle (download + YOLO) gated : `HUNT_REAL=1 make generalisation`.\n"
                "    Sans HUNT_REAL : on garde uniquement la query STAC (~80 scènes, gratuit).\n"
                "    Pour la démo : `python -m src.hunt toulon` ou voir\n"
                "    `outputs/piste_b_toulon_yolo_coco_diag.jpg` + `outputs/piste_b_diagnostic.md`."
            )
        # GFW gap events (Levier L2 : navires sombres) — seulement si token
        from src import ais_cross
        if _os.getenv("GFW_API_TOKEN"):
            print("\n  Levier L2 — Gap events GFW sur les 5 bases :")
            df_gaps = ais_cross.survey_gap_events_all_bases(window_days=30)
            results["PISTE_B"]["n_gap_events"] = len(df_gaps)
        else:
            print(
                "\n  Levier L2 (navires sombres) : `GFW_API_TOKEN` non défini "
                "dans .env -> sauté. Cf. cadrage §6.4."
            )
    else:
        print("  pystac-client absent -- `uv pip install pystac-client planetary-computer`.")
        results["PISTE_B"] = {"status": "skipped (pystac-client absent)"}

    # ------------------------------------------------------------------
    # P5 — Pipeline complet + API externes
    banner("P5 — Pipeline & API (Q13-Q15)")
    if _HAS_PHASE3:
        print("  [Q13] Pipeline : YOLO + classif + zone + règle d'alerte.")
        p5 = p5_pipeline.run_p5()
        print(
            f"    → {p5['n_total_detections']} détections traitées · "
            f"{p5['n_militaires']} militaires · "
            f"{p5['n_en_zone_militaire']} en zone · "
            f"**{p5['n_alertes']} alertes**."
        )
        for k, v in p5["files"].items():
            print(f"    → {v}")
        results["P5"] = {"Q13": {
            "n_total": p5["n_total_detections"],
            "n_militaires": p5["n_militaires"],
            "n_en_zone": p5["n_en_zone_militaire"],
            "n_alertes": p5["n_alertes"],
        }}

        # Q14 — Export ONNX + benchmark CPU
        print("\n  [Q14] Export ONNX YOLOv8n + benchmark CPU (PyTorch vs ONNX) :")
        try:
            onnx_info = p5_pipeline.export_onnx("yolov8n.pt")
            print(f"    → ONNX : {onnx_info['onnx_path']}")
            sanity_img = C.DATA_PROCESSED / "_sanity_input.jpg"
            if not sanity_img.exists():
                # bus.jpg d'Ultralytics comme fallback
                from ultralytics.utils import ASSETS
                sanity_img = Path(ASSETS) / "bus.jpg"
            df_bench = p5_pipeline.benchmark_inference(
                sanity_img, pt_weights="yolov8n.pt",
                onnx_weights=onnx_info["onnx_path"], n_iter=5,
            )
            print(df_bench.to_string(index=False))
            df_bench.to_csv(C.OUTPUTS / "q14_onnx_benchmark.csv", index=False)
            results["P5"]["Q14"] = df_bench.to_dict(orient="records")
        except Exception as e:  # noqa: BLE001
            print(f"    [Q14] sauté : {type(e).__name__}: {e}")
            results["P5"]["Q14"] = {"error": str(e)}

        # Q15 — APIs externes : 1 fiche d'intel sur la première alerte (démo)
        print(
            "\n  [Q15] APIs OSINT + LLM intel report sur 1 alerte (démo) :"
        )
        if p5["df_alerts"].empty:
            print("    Pas d'alerte → pas de fiche générée.")
        else:
            try:
                info = intel_report.render_intel_note(
                    p5["df_alerts"].iloc[0], enrich_osint=False
                )
                print(f"    → markdown : {info['markdown_path']}")
                print(f"    → PDF      : {info['pdf_path']}")
                print(f"    → backend  : {info['backend_used']} ({info['latency_s']}s)")
                results["P5"]["Q15"] = info
            except Exception as e:  # noqa: BLE001
                print(f"    [Q15] sauté : {type(e).__name__}: {e}")
                results["P5"]["Q15"] = {"error": str(e)}
    else:
        print("  Phase 3 modules absents — `uv pip install -r requirements.txt`.")
        results["P5"] = {"status": "skipped (phase 3 imports failed)"}

    # ------------------------------------------------------------------
    # P6 — Benchmark détecteurs (Q16) + classifieurs (Q17) + synthèse (Q18)
    banner("P6 — Benchmark (Q16-Q18)")
    if _HAS_PHASE3:
        print("  [Q16] Détecteurs (inférence CPU, image sanity) :")
        try:
            df_det = p6_benchmark.benchmark_detectors(n_iter=3)
            print(df_det.to_string(index=False))
            results["P6"] = {"Q16": df_det.to_dict(orient="records")}
        except Exception as e:  # noqa: BLE001
            print(f"    sauté : {type(e).__name__}: {e}")
            df_det = None
            results["P6"] = {"Q16": {"error": str(e)}}

        print("\n  [Q17] Classifieurs binaires (features Q7, 5-fold CV) :")
        try:
            df_clf = p6_benchmark.benchmark_classifiers()
            print(df_clf.to_string(index=False))
            results["P6"]["Q17"] = df_clf.to_dict(orient="records")
        except Exception as e:  # noqa: BLE001
            print(f"    sauté : {type(e).__name__}: {e}")
            df_clf = None
            results["P6"]["Q17"] = {"error": str(e)}

        print("\n  [Q18] Synthèse :")
        try:
            q18 = p6_benchmark.run_q18(df_det, df_clf)
            print(f"    → {q18['summary_md']}")
            print(f"    → {q18['fig_detectors']}")
            print(f"    → {q18['fig_classifiers']}")
            results["P6"]["Q18"] = {
                "summary_md": q18["summary_md"],
                "fig_detectors": q18["fig_detectors"],
                "fig_classifiers": q18["fig_classifiers"],
            }
        except Exception as e:  # noqa: BLE001
            print(f"    sauté : {type(e).__name__}: {e}")
    else:
        print("  Phase 3 modules absents — `uv pip install -r requirements.txt`.")
        results["P6"] = {"status": "skipped (phase 3 imports failed)"}

    # ------------------------------------------------------------------
    # Récap
    banner("Récapitulatif")
    summary = {
        "phase_1_done": ["Q1", "Q2", "Q3", "Q10", "Q11", "Q12"],
        "phase_2_done": [
            "Q7 (binaire RF/SVM/MLP, finding: AUC≈0.5 sur CSV synthétique)",
            "P2 sanity YOLOv8n (CPU/MPS, ~6 détections COCO)",
            "Piste B survey Sentinel-1 (~80 scènes sur 5 bases sur 30 j)",
            "ais_cross interface + GFW gap events (Levier L2)",
        ],
        "phase_3_done": [
            "Q13 pipeline.predict_from_csv → règles d'alerte (zone × risk × militaire)",
            "Q14 export ONNX + benchmark CPU PyTorch vs ONNX",
            "Q15 OSINT (Nominatim port + OpenWeatherMap météo + Overpass military zones)",
            "Q15/L3 intel_report → fiche markdown + PDF (Mistral REST si clé, sinon Jinja2)",
            "Q16-Q18 benchmark détecteurs (YOLO vs RT-DETR vs Faster R-CNN) + classifieurs",
        ],
        "phase_2_3_todo": [
            "Q4-Q6 fine-tune YOLO sur xView3 ou Airbus Ship  (→ Colab T4)",
            "Q8-Q9 ResNet50 + HRSC2016  (→ Colab ou local long)",
            "Piste B réelle : fetch_scene_geotiff + detect_ships_on_scene (→ Phase 2 J2)",
            "ais_cross : DMA CSV plan B si pas de GFW_API_TOKEN",
        ],
        "outputs": [
            "data/processed/q2_bbox_by_type.csv",
            "data/processed/q3_merged.csv",
            "data/processed/q10_detections_with_zones.csv",
            "data/processed/q11_temporal_series.csv",
            "data/processed/q12_anomalies_type_a.csv",
            "data/processed/q12_anomalies_type_b.csv",
            "data/processed/chasse_scenes_disponibles.csv",
            "outputs/q7_metrics_binaire.csv",
            "outputs/q7_roc_binaire.png",
            "outputs/q7_feature_importance.png",
            "outputs/q7_feature_importance.csv",
            "outputs/q13_pipeline_detections.csv",
            "outputs/q13_pipeline_alertes.csv",
            "outputs/q14_onnx_benchmark.csv",
            "outputs/q16_detectors_benchmark.csv",
            "outputs/q17_classifiers_benchmark.csv",
            "outputs/q18_benchmark_global.md",
            "outputs/q18_detectors_latency.png",
            "outputs/q18_classifiers.png",
            "outputs/intel_*.md / intel_*.pdf",
            "outputs/detections_carte.html",
            "outputs/anomalies_carte.html",
        ],
    }
    summary_path = C.OUTPUTS / "summary_phase1.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  → écrit {summary_path.relative_to(C.ROOT)}")
    print("\nPhase 1 OK. Lancer la Phase 2 (P2 + P3 + Piste B) ensuite.")

    return results


if __name__ == "__main__":
    main()
