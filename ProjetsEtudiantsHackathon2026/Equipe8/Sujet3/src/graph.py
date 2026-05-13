"""Graphe de connaissances navires ↔ anomalies ↔ zones ↔ alertes.

Le critère 1 de notation cite explicitement « **graphes de connaissance** ».
On le construit avec NetworkX (typed, simple) et on l'exporte en HTML
interactif avec pyvis pour le pitch.

Schéma :
  - Nœuds : Navire (mmsi), Anomalie (type), Pavillon (flag), Alerte (PDF)
  - Arêtes :
      Navire --[flagged_as]--> Anomalie
      Navire --[flies]--> Pavillon
      Navire --[triggers]--> Alerte
      Anomalie --[cited_in]--> Alerte
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd


def build_graph(ships: pd.DataFrame,
                anomalies: pd.DataFrame,
                scores: pd.DataFrame,
                top_n_suspect: int = 30) -> nx.Graph:
    """Construit un graphe **filtré** sur le top-N navires suspects (sinon
    illisible avec 1000 navires).

    Args:
        scores : DataFrame [mmsi, score, top_reasons, …] (sortie de build_global_score).
    """
    G = nx.Graph()

    # 1) Top N suspects
    top = scores.sort_values("score", ascending=False).head(top_n_suspect)
    susp_mmsi = set(top["mmsi"])

    # 2) Tous les MMSI dans anomalies_large
    anom_mmsi = set(anomalies["mmsi"]) & set(ships["mmsi"])

    selected_mmsi = susp_mmsi | anom_mmsi

    # 3) Ajout des nœuds
    for _, ship in ships[ships["mmsi"].isin(selected_mmsi)].iterrows():
        score = float(scores.set_index("mmsi").loc[ship["mmsi"], "score"]) \
            if ship["mmsi"] in scores["mmsi"].values else 0.0
        G.add_node(
            f"ship:{ship['mmsi']}", kind="Navire", mmsi=str(ship["mmsi"]),
            label=str(ship.get("name") or ship["mmsi"]),
            flag=str(ship.get("flag", "?")), score=score,
            color="#e74" if score > 0.4 else "#9be",
        )
        G.add_node(f"flag:{ship.get('flag', '?')}", kind="Pavillon",
                   label=str(ship.get("flag", "?")), color="#aaa")
        G.add_edge(f"ship:{ship['mmsi']}", f"flag:{ship.get('flag', '?')}",
                   relation="flies")

    # 4) Anomalies par type
    anom_filtered = anomalies[anomalies["mmsi"].isin(selected_mmsi)]
    for atype, grp in anom_filtered.groupby("type"):
        n_anom = f"anom:{atype}"
        G.add_node(n_anom, kind="Anomalie", label=atype, color="#fa3",
                   count=len(grp))
        for mmsi in grp["mmsi"]:
            G.add_edge(f"ship:{mmsi}", n_anom, relation="flagged_as")

    # 5) Alertes (1 par navire suspect au-dessus du seuil)
    for _, r in top.iterrows():
        n_alert = f"alert:{r['mmsi']}"
        G.add_node(n_alert, kind="Alerte", label=f"Alerte {r['mmsi']}",
                   color="#c33", score=float(r["score"]))
        G.add_edge(f"ship:{r['mmsi']}", n_alert, relation="triggers")

    return G


def export_html(G: nx.Graph, out: Path, height: str = "800px") -> Path:
    """Exporte le graphe en HTML interactif via pyvis."""
    from pyvis.network import Network

    net = Network(height=height, width="100%", bgcolor="#fff",
                  font_color="#222", directed=False,
                  notebook=False, cdn_resources="remote")
    net.from_nx(G)
    net.toggle_physics(True)
    net.set_options("""
    var options = {
      "physics": {"barnesHut": {"gravitationalConstant": -8000,
                                  "centralGravity": 0.3,
                                  "springLength": 120}},
      "nodes": {"font": {"size": 14}},
      "edges": {"smooth": {"enabled": true, "type": "dynamic"}}
    }
    """)
    net.write_html(str(out), notebook=False)
    return out


def stats(G: nx.Graph) -> dict:
    """Stats du graphe pour le rapport."""
    kinds = {}
    for _, d in G.nodes(data=True):
        kinds[d.get("kind", "?")] = kinds.get(d.get("kind", "?"), 0) + 1
    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "by_kind": kinds,
        "n_components": nx.number_connected_components(G),
        "avg_degree": (2 * G.number_of_edges() / max(G.number_of_nodes(), 1)),
    }
