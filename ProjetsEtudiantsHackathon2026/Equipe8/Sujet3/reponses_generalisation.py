"""LIVRABLE PRINCIPAL — Sujet 3 / Généralisation (Q1 → Q14).

Sections numérotées comme l'énoncé. Chaque section :
  - calcule un résultat,
  - l'affiche (`print`),
  - sauvegarde un artefact (`data/processed/...` ou `outputs/...`),
  - laisse une note d'analyse critique en commentaire.

Lancement : `python reponses_generalisation.py` (ou `make generalisation`).
Pipeline reproductible : tourne de bout en bout depuis un env neuf en ≤ 60 s.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
from scipy import stats

from src import (
    anomalies,
    anomaly_score,
    cluster,
    eda,
    evaluation,
    gbm_classifier,
    graph,
    maps,
    profiles,
    pyod_benchmark,
    spoofing_rules,
    tuning,
)
from src.config import (
    AIS_OFF_HOURS,
    ASOF_TOLERANCE_MIN,
    DATA_PROCESSED,
    FREQ_JUMP_MHZ,
    OUTPUTS,
    POS_MISMATCH_KM,
    SIGNAL_JUMP_DBM,
)
from src.identify import Identifier
from src.intel_report import render_intel_note, to_pdf
from src.load import (
    annotate_mmsi_validity,
    load_ais,
    load_anomalies,
    load_radio,
    load_ships,
)
from src.osint_enrich import (
    build_osint_dossier,
    extract_suspicious_mmsi_from_rss,
    fetch_rss,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
pd.set_option("display.max_columns", 12)
pd.set_option("display.width", 140)

SECTION = "\n" + "─" * 78
RESULTS: dict = {}


def _section(title: str) -> None:
    print(f"{SECTION}\n  {title}\n" + "─" * 78)


# =============================================================================
def main() -> None:
    t0 = time.perf_counter()
    print("\n" + "=" * 78)
    print("  SUJET 3 — Généralisation (Q1 → Q14)")
    print("  Réponses produites par le pipeline ; toutes les hypothèses dans "
          "`docs/hypotheses.md`.")
    print("=" * 78)

    # --- Préalable : chargement, EDA, contrôle qualité, nettoyage ------------
    _section("Préalable — chargement, EDA & contrôle qualité")
    ships = load_ships()
    radio = load_radio()
    ais = load_ais()
    anom = load_anomalies()
    ships = annotate_mmsi_validity(ships)
    # Marqueur de cleaning : navire jamais entendu sur les ondes
    ships["has_radio_signature"] = ships["mmsi"].isin(set(radio["mmsi"]))

    qa = eda.run_quality_report(ships, radio, ais, anom)
    # Affichage synthétique des points clés
    print(f"  Volumétrie : {qa['volumetrie']}")
    print(f"  Manquants : {qa['manquants'] or 'aucun'}")
    print(f"  Doublons : {qa['doublons']}")
    print(f"  Hors-bornes : {qa['hors_bornes'] or 'aucun'}")
    print(f"  Catégories hors-liste : {qa['categories_hors_liste'] or 'aucune'}")
    print(f"  MMSI orphelins (radio/ais/anom) : "
          f"{qa['integrite_referentielle']['radio_mmsi_orphans']}/"
          f"{qa['integrite_referentielle']['ais_mmsi_orphans']}/"
          f"{qa['integrite_referentielle']['anom_mmsi_orphans']}")
    print(f"  Navires sans signature radio : {qa['integrite_referentielle']['ships_without_radio']} "
          f"({qa['integrite_referentielle']['ships_without_radio_mmsi']})")
    print(f"  Structure — V Cramér nav_status↔status : "
          f"{qa['structure'].get('cramers_v_navstatus_vs_status')} | "
          f"ais_active=True : {qa['structure'].get('ais_active_true_pct', 0):.1%}")
    if "implicit_speed_knots" in qa["structure"]:
        s = qa["structure"]["implicit_speed_knots"]
        print(f"  Vitesse implicite AIS : médiane {s['median']} kn, max {s['max']} kn "
              f"⇒ positions = points aléatoires (pas des trajectoires)")
    print(f"  ⚠️ Plafond de rappel atteignable : "
          f"{qa['ceiling']['achievable_recall_ceiling']:.0%} "
          f"({qa['ceiling']['n_unrecoverable_from_tables']}/{qa['ceiling']['n_anomalies_total']} "
          f"anomalies Speed/Course non récupérables des tables)")
    print(f"  Actions de cleaning : {len(qa['cleaning_items'])} →")
    for it in qa["cleaning_items"]:
        print(f"    • {it}")
    md_path = eda.write_markdown(qa)
    (DATA_PROCESSED / "data_quality.json").write_text(
        json.dumps(qa, indent=2, default=str))
    print(f"  → rapport détaillé : {md_path}  +  data/processed/data_quality.json")
    RESULTS["data_quality"] = qa

    # Détecteur « navire silencieux » (EDA)
    silent = anomalies.detect_silent_ships(ships, radio)
    n_silent_susp = int(silent["is_suspicious"].sum()) if "is_suspicious" in silent else 0
    print(f"  Navires silencieux (0 signature) : {len(silent)}, dont {n_silent_susp} `is_suspicious`")
    RESULTS["silent_ships"] = {"n": int(len(silent)), "n_suspicious": n_silent_susp,
                               "mmsi": silent["mmsi"].tolist()}

    # =========================================================================
    # PARTIE 1 — Base de profils radio
    # =========================================================================
    _section("Q1 — Agrégation des signatures radio + ship_radio_profiles.csv")
    prof = profiles.build_profiles(radio, ships=ships)
    top5 = profiles.top_n_frequency(prof, n=5)
    print(f"→ {len(prof)} profils écrits dans data/processed/ship_radio_profiles.csv")
    print("Top 5 freq_mean :")
    print(top5.to_string(index=False))
    RESULTS["q1_top5_freq"] = top5.to_dict(orient="records")
    # NOTE : on aurait pu compléter par la médiane (plus robuste) et le mode
    # catégoriel — c'est ce que fait `profiles.build_profiles`.

    _section("Q2 — `pulse_pattern` uniques")
    pp = profiles.unique_pulse_patterns(radio)
    print(f"Nombre de patterns distincts : {pp['n_distinct_patterns']}")
    print(f"Patterns avec count == 1     : {pp['patterns_with_count_1']}")
    print(f"Combinaisons (pulse × mod × bande) uniques : {pp['n_unique_combinations']}")
    # NOTE : `pulse_pattern` seul = 6 valeurs → peu discriminant. La combinaison
    # `pulse × modulation × bande de frequency` est bien plus informative.
    RESULTS["q2"] = pp

    _section("Q3 — K-Means K=5 + Silhouette + PCA 2D")
    res = cluster.fit_kmeans(prof)
    print(f"Nb clusters : {len(set(res['labels']))}")
    print(f"WCSS = {res['wcss']:.1f} | Silhouette = {res['silhouette']:.3f}")
    elbow = cluster.elbow_curve(prof, k_range=range(2, 11))
    print(elbow.round(3).to_string(index=False))
    elbow.to_csv(OUTPUTS / "q3_elbow_silhouette.csv", index=False)
    cluster.plot_clusters_2d(res, prof, out=OUTPUTS / "clusters_kmeans.png")
    # NOTE : cours 1 — K-Means++ + n_init=10 ; biais sphérique acceptable ici
    # (3 features quantitatives à peu près convexes ; à valider via la PCA).
    RESULTS["q3"] = {"silhouette": res["silhouette"], "wcss": res["wcss"],
                     "elbow": elbow.to_dict(orient="records")}

    # =========================================================================
    # PARTIE 2 — Détection d'anomalies
    # =========================================================================
    _section("Q4 — Anomalies de pavillon (EllipticEnvelope / MCD par flag)")
    ff_report = anomalies.detect_fake_flag(prof)
    top10 = anomalies.top_n_fake_flag(ff_report, n=10)
    print(top10.to_string(index=False))
    ff_report.to_csv(OUTPUTS / "q4_fake_flag_report.csv", index=False)
    # Évaluation vs vérité terrain
    flagged = set(ff_report[ff_report["predicted"] == -1]["mmsi"])
    truth_ff = set(anom[anom["type"] == "Fake Flag"]["mmsi"])
    if truth_ff:
        inter = flagged & truth_ff
        prec = len(inter) / max(len(flagged), 1)
        rec = len(inter) / max(len(truth_ff), 1)
        print(f"Évaluation vs anomalies_large[type='Fake Flag'] : "
              f"precision={prec:.2%} | recall={rec:.2%} | flaggés={len(flagged)} | vérité={len(truth_ff)}")
        RESULTS["q4_eval"] = {"precision": prec, "recall": rec,
                              "n_flagged": len(flagged), "n_truth": len(truth_ff)}
    RESULTS["q4_top10"] = top10.to_dict(orient="records")

    _section("Q5 — Changements de nom")
    nc = anomalies.detect_name_change(ships, threshold=2)
    print(f"Navires avec > 2 noms historiques : {len(nc)}")
    # croiser avec anomalies_large pour évaluer
    truth = set(anom[anom["type"] == "Name Change"]["mmsi"])
    found = set(nc["mmsi"])
    inter = found & truth
    print(f"Match avec `anomalies_large.type='Name Change'` : "
          f"{len(inter)}/{len(truth)} (recall) ; {len(inter)}/{len(found) or 1} (precision)")
    RESULTS["q5"] = {"n_flagged": len(nc),
                     "recall": len(inter) / max(len(truth), 1),
                     "precision": len(inter) / max(len(found), 1)}

    _section("Q6 — Signatures orphelines")
    orph = anomalies.detect_orphan_signatures(radio, ships)
    attrib = anomalies.attribute_orphans(orph, prof, k=5)
    print(f"Signatures orphelines : {len(orph)} "
          f"(MMSI distincts : {orph['mmsi'].nunique() if len(orph) else 0})")
    if len(attrib):
        print(attrib[["signature_id", "mmsi", "candidate_mmsi",
                      "candidate_distance", "candidate_confidence"]].head(10).to_string(index=False))
    attrib.to_csv(OUTPUTS / "q6_orphan_attribution.csv", index=False)
    RESULTS["q6"] = {"n_orphans": int(len(orph))}

    _section(f"Q7 — AIS off > {AIS_OFF_HOURS} h consécutives (exclut Moored/At Anchor)")
    blocks = anomalies.detect_ais_disabled(ais)
    suspect_blocks = blocks[blocks["suspect"]]
    print(f"Plages suspectes : {len(suspect_blocks)} ; "
          f"navires concernés : {suspect_blocks['mmsi'].nunique()}")
    suspect_blocks.to_csv(OUTPUTS / "q7_ais_off_blocks.csv", index=False)
    RESULTS["q7"] = {"n_blocks": int(len(suspect_blocks)),
                     "n_mmsi": int(suspect_blocks["mmsi"].nunique())}

    _section(f"Q8 — Écart position AIS↔radio > {POS_MISMATCH_KM} km + carte folium")
    mismatch = anomalies.detect_position_mismatch(ais, radio)
    print(f"Paires en écart : {len(mismatch)} ; "
          f"navires concernés : {mismatch['mmsi'].nunique()}")
    mismatch.to_csv(OUTPUTS / "q8_position_mismatch.csv", index=False)
    m = maps.base_map()
    maps.layer_position_mismatches(m, mismatch.head(300))   # cap pour rester lisible
    map_path = maps.save(m, "carte_anomalies.html")
    print(f"→ {map_path}")
    RESULTS["q8"] = {"n_pairs": int(len(mismatch)),
                     "n_mmsi": int(mismatch["mmsi"].nunique()),
                     "map": str(map_path)}

    # =========================================================================
    # PARTIE 3 — Temporel & statistique
    # =========================================================================
    _section("Q9 — Évolution frequency / signal_strength d'un MMSI + ruptures")
    # On choisit le MMSI ayant le plus de signatures.
    target = radio["mmsi"].value_counts().idxmax()
    g = radio[radio["mmsi"] == target].sort_values("timestamp")
    jumps_freq = (g["frequency"].diff().abs() > FREQ_JUMP_MHZ).sum()
    jumps_sig = (g["signal_strength"].diff().abs() > SIGNAL_JUMP_DBM).sum()
    print(f"MMSI {target} : {len(g)} signatures, sauts freq>{FREQ_JUMP_MHZ}MHz = {jumps_freq}, "
          f"sauts signal>{SIGNAL_JUMP_DBM}dBm = {jumps_sig}")
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        ax[0].plot(g["timestamp"], g["frequency"], "o-")
        ax[0].set_ylabel("frequency (MHz)")
        ax[0].set_title(f"Évolution MMSI {target}")
        ax[1].plot(g["timestamp"], g["signal_strength"], "o-", color="orange")
        ax[1].set_ylabel("signal_strength (dBm)")
        fig.autofmt_xdate()
        out = OUTPUTS / f"q9_temporal_{target}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"→ {out}")
    except Exception as exc:
        print(f"viz Q9 sautée ({exc})")
    RESULTS["q9"] = {"mmsi": target, "jumps_freq": int(jumps_freq),
                     "jumps_signal": int(jumps_sig)}

    _section("Q10 — Stats frequency par pavillon")
    flag_stats = (radio.merge(ships[["mmsi", "flag"]], on="mmsi")
                  .groupby("flag")["frequency"].agg(["mean", "std", "count"])
                  .sort_values("mean", ascending=False))
    print(flag_stats.round(3).to_string())
    flag_stats.to_csv(OUTPUTS / "q10_flag_stats.csv")
    RESULTS["q10"] = {"top_flag": flag_stats.index[0],
                      "top_mean": float(flag_stats.iloc[0]["mean"])}

    _section("Q11 — Corrélation speed (AIS) ↔ frequency (radio)")
    radio_s = radio.sort_values("timestamp").rename(columns={"timestamp": "ts_radio"})
    ais_s = (ais.sort_values("timestamp")
             .rename(columns={"timestamp": "ts_ais"})[["mmsi", "ts_ais", "speed"]])
    paired = pd.merge_asof(
        radio_s.sort_values("ts_radio"),
        ais_s.sort_values("ts_ais"),
        left_on="ts_radio", right_on="ts_ais",
        by="mmsi",
        tolerance=pd.Timedelta(minutes=ASOF_TOLERANCE_MIN),
        direction="nearest",
    ).dropna(subset=["speed"])
    if len(paired) >= 30:
        r, p = stats.pearsonr(paired["speed"], paired["frequency"])
        signif = (p < 0.05) and (abs(r) > 0.1)
        print(f"r = {r:.3f} | p = {p:.3g} | significatif = {signif} "
              f"(N={len(paired)})")
        # NOTE : hypothèse a priori — non significatif (la fréquence VHF ne
        # dépend pas physiquement de la vitesse). Une réponse négative
        # argumentée est valide.
        RESULTS["q11"] = {"r": float(r), "p": float(p),
                          "significant": bool(signif), "n": len(paired)}
    else:
        print(f"Trop peu de paires AIS↔radio appariées ({len(paired)}) — corrélation non calculée.")
        RESULTS["q11"] = {"r": None, "p": None, "n": len(paired)}

    # =========================================================================
    # PARTIE 4 — Automatisation, validation, mise à jour
    # =========================================================================
    _section("Score multi-facteurs (Levier 3 : Isolation Forest + LOF + zone + règles)")
    if_scores = anomaly_score.isolation_forest_scores(radio)
    lof_df = anomaly_score.lof_scores(radio)
    zone_df = anomaly_score.zone_anomaly_score(ais)
    spoof_df = spoofing_rules.run_all(ais)
    print(f"  iForest : {(if_scores['if_anomaly'] == -1).sum()} signatures flaguées")
    print(f"  LOF     : {(lof_df['lof_anomaly'] == -1).sum()} signatures flaguées")
    print(f"  Spoofing rules : {len(spoof_df)} occurrences "
          f"({spoof_df['rule'].value_counts().to_dict() if len(spoof_df) else {}})")
    global_score = anomaly_score.build_global_score(
        ships,
        fake_flag_report=ff_report,
        name_change_mmsi=nc["mmsi"].tolist(),
        orphan_attribution=attrib if len(orph) else None,
        ais_off_blocks=blocks,
        position_mismatch=mismatch,
        spoofing_rules=spoof_df,
        iforest_scores=if_scores,
        lof_scores_df=lof_df,
        zone_scores=zone_df,
    )
    global_score.to_csv(OUTPUTS / "global_score.csv", index=False)
    print(global_score[["mmsi", "score", "top_reasons"]].sort_values("score", ascending=False).head(10).to_string(index=False))

    metrics = anomaly_score.evaluate_score(global_score, ships)
    print("Évaluation vs `is_suspicious` (label dataset, ~50% des navires) :")
    print({k: round(v, 3) if isinstance(v, float) else v
           for k, v in metrics.items()
           if k in ("auc", "best_threshold", "precision_at_best",
                    "recall_at_best", "f1_at_best", "best_cost")})

    # Évaluation plus stricte : vs `anomalies_large.csv` (vérité terrain)
    truth_mmsi = set(anom["mmsi"])
    s = global_score.merge(ships[["mmsi"]], on="mmsi", how="right")
    y_true = ships["mmsi"].isin(truth_mmsi).astype(int).to_numpy()
    y_score = s["score"].fillna(0).to_numpy()
    if y_true.sum() > 0:
        from sklearn.metrics import roc_auc_score
        auc_anom = roc_auc_score(y_true, y_score)
        k = int(y_true.sum())
        top_k_idx = np.argsort(-y_score)[:k]
        prec_at_k = float(y_true[top_k_idx].sum() / k)
        print(f"Évaluation vs anomalies_large.csv ({k} navires anormaux) "
              f"[poids uniformes] : AUC={auc_anom:.3f} | precision@{k}={prec_at_k:.2%}")
        RESULTS["score_uniform_weights"] = {"auc": auc_anom,
                                            "precision_at_k": prec_at_k, "k": k}
    RESULTS["score_metrics"] = {k: v for k, v in metrics.items()
                                 if not isinstance(v, list)}

    # ----------------- Rappel par type d'anomalie ----------------------------
    _section("Évaluation par type d'anomalie (qu'attrape-t-on ? que rate-t-on ?)")
    detected_by_type = {
        "Fake Flag": set(ff_report[ff_report["predicted"] == -1]["mmsi"]),
        "Name Change": set(nc["mmsi"]),
        "Spoofing": (set(orph["mmsi"]) if len(orph) else set())
                    | set(spoof_df[spoof_df["type"] == "Spoofing"]["mmsi"]),
        "AIS Disabled": set(blocks[blocks["suspect"]]["mmsi"]),
        "Position Mismatch": set(mismatch["mmsi"]),
        # Récupération cinématique des Speed/Course Anomaly : on calcule la
        # vitesse implicite (Δdistance / Δt) et le Δcap entre positions
        # consécutives. Lève le plafond théorique de rappel (cf. EDA).
        "Speed Anomaly": set(spoof_df[spoof_df["type"] == "Speed Anomaly"]["mmsi"]),
        "Course Anomaly": set(spoof_df[spoof_df["type"] == "Course Anomaly"]["mmsi"]),
    }
    recall_df = evaluation.recall_by_anomaly_type(anom, detected_by_type)
    evaluation.pretty_print(recall_df)
    recall_df.to_csv(OUTPUTS / "recall_by_type.csv", index=False)
    evaluation.plot_recall_by_type(recall_df)
    print("  → outputs/recall_by_type.png  +  outputs/recall_by_type.csv")
    RESULTS["recall_by_type"] = recall_df.to_dict(orient="records")

    # ----------------- TUNING -------------------------------------------------
    _section("Tuning des poids — Régression L2 5-fold CV (vs anomalies_large)")
    tune_res = tuning.tune_weights(global_score, ships, anom,
                                   plot_path=OUTPUTS / "roc_score_tuned.png")
    tuning.pretty_print_tuning(tune_res)
    print("  → courbe ROC : outputs/roc_score_tuned.png")
    # Recalcule le score avec les poids optimaux
    tuned_score = anomaly_score.build_global_score(
        ships,
        fake_flag_report=ff_report,
        name_change_mmsi=nc["mmsi"].tolist(),
        orphan_attribution=attrib if len(orph) else None,
        ais_off_blocks=blocks,
        position_mismatch=mismatch,
        spoofing_rules=spoof_df,
        iforest_scores=if_scores,
        lof_scores_df=lof_df,
        zone_scores=zone_df,
        weights=tune_res["weights_optimal"],
    )
    tuned_score.to_csv(OUTPUTS / "global_score_tuned.csv", index=False)

    # Évaluation finale avec les nouveaux poids
    y_score_t = ships[["mmsi"]].merge(tuned_score, on="mmsi", how="left")["score"].fillna(0).to_numpy()
    from sklearn.metrics import roc_auc_score
    auc_tuned = float(roc_auc_score(y_true, y_score_t))
    k = int(y_true.sum())
    prec_at_k_tuned = float(y_true[np.argsort(-y_score_t)[:k]].sum() / k)
    print(f"\n  Score TUNÉ vs anomalies_large.csv : "
          f"AUC={auc_tuned:.3f} | precision@{k}={prec_at_k_tuned:.2%}")
    print(f"  Gain AUC : {auc_tuned - RESULTS['score_uniform_weights']['auc']:+.3f}  "
          f"| Gain precision@{k} : "
          f"{prec_at_k_tuned - RESULTS['score_uniform_weights']['precision_at_k']:+.2%}")
    RESULTS["score_tuned"] = {
        "auc": auc_tuned, "precision_at_k": prec_at_k_tuned,
        "auc_oof": tune_res["auc_oof"],
        "auc_train_mean": tune_res["auc_train_mean"],
        "precision_at_k_oof": tune_res["precision_at_k_oof"],
        "weights": asdict(tune_res["weights_optimal"]),
        "coefs_mean": tune_res["coefs_signed"],
        "coefs_std": tune_res["coefs_std"],
    }
    # On utilise désormais les scores tunés pour la carte & la fiche d'alerte
    global_score = tuned_score

    # ----------------- XGBoost (hors cours, valorisé par consigne) ----------
    _section("XGBoost supervisé (force de proposition — cible: anomalies_large)")
    gbm_res = gbm_classifier.train_eval(
        global_score, ships, anom, profiles=prof,
        plot_path=OUTPUTS / "roc_xgboost.png",
    )
    gbm_classifier.pretty_print(gbm_res)
    print("  → courbe ROC + importances : outputs/roc_xgboost.png")
    print("  Comparaison directe :")
    print(f"    LogReg L2 (cours)  : AUC OOF {tune_res['auc_oof']:.3f}  precision@k {tune_res['precision_at_k_oof']:.2%}")
    print(f"    XGBoost (hors cours): AUC OOF {gbm_res['auc_oof']:.3f}  precision@k {gbm_res['precision_at_k_oof']:.2%}")
    RESULTS["xgboost"] = {
        "auc_oof": gbm_res["auc_oof"],
        "auc_train_mean": gbm_res["auc_train_mean"],
        "precision_at_k_oof": gbm_res["precision_at_k_oof"],
        "feature_importance_top10": sorted(
            zip(gbm_res["feature_cols"], gbm_res["feature_importance"]),
            key=lambda kv: -kv[1])[:10],
    }

    # ----------------- PyOD benchmark (40+ détecteurs hors cours) -----------
    _section("Benchmark PyOD — 5 détecteurs au-delà du cours")
    bench_df = pyod_benchmark.benchmark(radio, ships, anom)
    pyod_benchmark.pretty_print(bench_df)
    bench_df.to_csv(OUTPUTS / "pyod_benchmark.csv", index=False)
    RESULTS["pyod_benchmark"] = bench_df.to_dict(orient="records")

    # ----------------- Graphe de connaissances (critère 1 explicite) --------
    _section("Graphe de connaissances NetworkX + pyvis (navires↔anomalies↔alertes)")
    G = graph.build_graph(ships, anom, global_score, top_n_suspect=30)
    gstats = graph.stats(G)
    print(f"  Nœuds : {gstats['n_nodes']} | Arêtes : {gstats['n_edges']} | "
          f"Composantes : {gstats['n_components']} | degré moyen : {gstats['avg_degree']:.2f}")
    print(f"  Répartition par type : {gstats['by_kind']}")
    g_html = graph.export_html(G, OUTPUTS / "knowledge_graph.html")
    print(f"  → {g_html}")
    RESULTS["graph"] = gstats

    # Carte enrichie : couche navires suspects
    m2 = maps.base_map()
    maps.layer_position_mismatches(m2, mismatch.head(300))
    maps.layer_suspect_ships(m2, global_score, ships, ais,
                             threshold=metrics.get("best_threshold", 0.5))
    maps.save(m2, "carte_anomalies.html")

    _section("Q12 — Pipeline d'identification passive (k-NN + One-Class SVM)")
    # Entraîné sur toutes les signatures individuelles (≠ profils moyens) →
    # vote majoritaire k=3, robuste au bruit synthétique.
    identifier = Identifier(radio, ships)
    # On prend une signature pour démo (toutes les colonnes pour le one-hot)
    demo_sig_row = radio.sample(1, random_state=42).iloc[0]
    res12 = identifier.identify(demo_sig_row.to_dict())
    print(f"Signature démo (vraie MMSI : {demo_sig_row['mmsi']})")
    print(f"  → mmsi prédit : {res12.mmsi} (confidence {res12.confidence:.3f})")
    print(f"  → suspect : {res12.suspect} | novelty_score : {res12.novelty_score:.3f}")
    print(f"  → raisons : {res12.reasons}")

    # Génère une fiche de renseignement + PDF pour le top-1 du score global
    top1 = global_score.sort_values("score", ascending=False).iloc[0]
    ship_row = ships[ships["mmsi"] == top1["mmsi"]].iloc[0].to_dict()
    dossier = build_osint_dossier(top1["mmsi"], ship_row)
    md = render_intel_note(
        mmsi=top1["mmsi"], name=ship_row.get("name"), imo=ship_row.get("imo"),
        flag_declared=ship_row.get("flag"), ship_type=ship_row.get("type"),
        country_attributed=dossier["uit"].get("country_attributed"),
        flag_mismatch_uit=bool(dossier.get("flag_mismatch_uit")),
        historical_names=ship_row.get("historical_names_list"),
        n_names_historical=int(ship_row.get("n_names_historical") or 0),
        sanctioned=dossier["sanctioned"], n_sanctions=len(dossier["sanctions"]),
        is_suspicious=bool(ship_row.get("is_suspicious")),
        score=float(top1["score"]), top_reasons=top1["top_reasons"],
    )
    (OUTPUTS / f"alerte_{top1['mmsi']}.md").write_text(md, encoding="utf-8")
    pdf = to_pdf(md, top1["mmsi"])
    print(f"→ fiche  : outputs/alerte_{top1['mmsi']}.md")
    print(f"→ alerte : {pdf}")

    _section("Q13 — Validation sur 10 signatures × 5 graines (leave-one-out)")
    val = identifier.validate(radio, ships, n=10, seeds=[1, 2, 3, 4, 5])
    print(f"Taux d'identification correcte : {val['mean']:.2%} ± {val['std']:.2%} "
          f"(par graine : {[f'{x:.0%}' for x in val['per_seed']]})")
    print(f"Matrice de confusion [[TP, FN], [FP, TN]] : {val['matrix']}")
    RESULTS["q13"] = val

    _section("Q14 — Mise à jour auto + ingestion OSINT (flux RSS)")
    import re as _re
    feeds = [
        "https://gcaptain.com/feed/",
        "https://www.maritime-executive.com/articles.rss",
        "https://ofac.treasury.gov/recent-actions/rss.xml",  # OFAC press releases
    ]
    all_items: list[dict] = []
    new_mmsi_from_rss: set[str] = set()
    for feed in feeds:
        try:
            items = fetch_rss(feed, max_items=30)
            new_mmsi_from_rss |= extract_suspicious_mmsi_from_rss(items)
            print(f"  {feed} → {len(items)} items")
            all_items.extend(items)
        except Exception as exc:
            print(f"  {feed} indisponible ({exc})")
    print(f"MMSI extraits des items RSS : {sorted(new_mmsi_from_rss)[:10]} ...")
    n_updated = int(ships["mmsi"].isin(new_mmsi_from_rss).sum())
    print(f"→ MMSI déjà connus dans le registre : {n_updated}")

    # Persistance veille : `data/processed/watchlist.csv` (alimente le dashboard).
    # On déduplique par GUID, on garde les 50 plus récents.
    if all_items:
        _imo_pat = _re.compile(r"\bIMO\s*0*(\d{7})\b", _re.I)
        _mmsi_pat = _re.compile(r"\b\d{9}\b")
        # Suspicion lexicale légère : mots-clés FR/EN
        _susp_pat = _re.compile(
            r"\b(sanction|sanctioned|sanctioned\svessel|seized|seizure|"
            r"shadow\sfleet|shadow|dark\sfleet|dark\svessel|"
            r"oil\stanker|spoof|spoofing|illicit|"
            r"saisi|saisie|sanctionn[eé]|flotte\sfant[oô]me|flotte\snoire)\b",
            _re.I,
        )
        rows = []
        seen = set()
        for it in all_items:
            g = it.get("guid")
            if g in seen:
                continue
            seen.add(g)
            text = f"{it.get('title','')} {it.get('summary','')}"
            mmsi_match = _mmsi_pat.search(text)
            imo_match = _imo_pat.search(text)
            susp_keywords = sorted({m.lower() for m in _susp_pat.findall(text)})
            rows.append({
                "guid": g,
                "source": it.get("feed", "").split("/")[2] if "://" in it.get("feed", "") else it.get("feed", ""),
                "title": it.get("title", "")[:240],
                "link": it.get("link", ""),
                "published": it.get("published", ""),
                "mmsi": mmsi_match.group(0) if mmsi_match else "",
                "imo": imo_match.group(1) if imo_match else "",
                "suspicion_keywords": ", ".join(susp_keywords),
            })
        wl = pd.DataFrame(rows)
        # On trie par date publiée (string) en plaçant les non-parseables à la fin
        try:
            wl["_d"] = pd.to_datetime(wl["published"], errors="coerce", utc=True)
            wl = wl.sort_values("_d", ascending=False, na_position="last").drop(columns="_d")
        except Exception:
            pass
        wl = wl.head(50).reset_index(drop=True)
        wl_path = (OUTPUTS.parent / "data" / "processed" / "watchlist.csv")
        wl_path.parent.mkdir(parents=True, exist_ok=True)
        wl.to_csv(wl_path, index=False)
        n_susp = int((wl["suspicion_keywords"] != "").sum())
        print(f"→ Veille persistée : {len(wl)} items dans {wl_path.relative_to(OUTPUTS.parent)} "
              f"({n_susp} avec mots-clés de suspicion)")

    # =========================================================================
    print(SECTION)
    elapsed = time.perf_counter() - t0
    print(f"  Pipeline terminé en {elapsed:.1f} s.")
    (OUTPUTS / "results.json").write_text(json.dumps(RESULTS, indent=2, default=str))
    print("  Synthèse → outputs/results.json")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
