'Convert clinical trajectories to networkx graphs.'

from __future__ import annotations

import networkx as nx

from .schema import ClinicalTrajectory


def trajectory_to_digraph(trajectory: ClinicalTrajectory) -> nx.DiGraph:
    'Convert a ClinicalTrajectory to a directed attribute graph.'
    graph = nx.DiGraph()
    for node in trajectory.nodes:
        graph.add_node(
            node.node_id,
            node_type=node.node_type,
            normalized_label=node.normalized_label,
            label=node.label,
            relative_day=node.relative_day,
            time=node.time,
            description=node.description or "",
        )
    for edge in trajectory.edges:
        graph.add_edge(
            edge.source,
            edge.target,
            relation=edge.relation,
            description=edge.description or "",
        )
    return graph

