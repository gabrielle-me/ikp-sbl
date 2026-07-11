"""SearchTee class for all planners"""

from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree



class SearchTree:
    """A lightweight tree wrapper that stores nodes, parents, and positions."""

    def __init__(self, root_position: List[float]):
        self.graph = nx.Graph()
        self.parent: Dict[int, Optional[int]] = {}
        self.node_ids: List[int] = []
        self._next_node_id = 0
        self.root: int = self.add_node(root_position, parent=None)

    def add_node(self, position: List[float], parent: Optional[int]) -> int:
        node_id = self._next_node_id
        self.graph.add_node(node_id, pos=position)
        self.node_ids.append(node_id)
        self.parent[node_id] = parent
        if parent is not None:
            # Set the edge status attribute to 'unknown' by default
            self.graph.add_edge(parent, node_id, status="unknown")
        self._next_node_id += 1
        return node_id

    def position(self, node_id: int) -> List[float]:
        return self.graph.nodes[node_id]["pos"]

    def nearest(self, position: List[float]) -> Tuple[int, float]:
        """Returns the nearest node ID and the distance to it."""
        if len(self.node_ids) == 1:
            dist = np.linalg.norm(np.array(self.position(self.node_ids[0])) - np.array(position))
            return self.node_ids[0], float(dist)
        positions = [self.position(node_id) for node_id in self.node_ids]
        kd_tree = cKDTree(positions)
        dist, index = kd_tree.query(position, k=1)
        return self.node_ids[int(index)], float(dist)

    def path_to_root(self, node_id: int) -> List[List[float]]:
        path: List[List[float]] = []
        current = node_id
        while current is not None:
            path.append(self.position(current))
            current = self.parent[current]
        return list(reversed(path))