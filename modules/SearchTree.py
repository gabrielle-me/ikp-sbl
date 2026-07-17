"""SearchTree class for all planners"""

from typing import Any, Dict, List, Optional, Tuple

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

    def _make_node_uid(self, node_id: int, position: List[float]) -> int:
        coord_parts = [str(coord).replace('.', '') for coord in position]
        return int(f"{node_id}{''.join(coord_parts)}")

    def add_node(self, position: List[float], parent: Optional[int]) -> int:
        node_id = self._next_node_id
        node_uid = self._make_node_uid(node_id, position)
        self.graph.add_node(node_uid, pos=position)
        self.node_ids.append(node_uid)
        self.parent[node_uid] = parent
        if parent is not None:
            # Set the edge status attribute to 'unknown' by default
            self.graph.add_edge(parent, node_uid, status="unknown")
        self._next_node_id += 1
        return node_uid

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
    
    def mark_unreachable_subtree(self, root_node_id: int) -> None:
        """Soft-prune helper: mark all edges in the subtree rooted at ``root_node_id`` as unreachable.

        This does not remove nodes or edges from the tree, but ensures that all
        edges in the subtree are never used again for candidate paths. The
        traversal is restricted to *descendants* of ``root_node_id`` by using
        the ``parent`` mapping to orient the undirected NetworkX graph.
        """

        # Depth-first traversal over the subtree starting at ``root_node_id``.
        # We avoid building a global children map and instead derive children
        # on-the-fly from graph neighbors whose parent is the current node.
        stack: List[int] = [root_node_id]
        visited = set()

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)

            # Mark the edge between the current node and its parent as unreachable
            parent_id = self.parent.get(current)
            if parent_id is not None and self.graph.has_edge(parent_id, current):
                self.graph[parent_id][current]["status"] = "unreachable"

            # Children are exactly those neighbors whose recorded parent is ``current``
            for neighbor in self.graph.neighbors(current):
                if self.parent.get(neighbor) == current and neighbor not in visited:
                    stack.append(neighbor)

    def path_to_root(self, node_id: int) -> List[List[float]]:
        path: List[List[float]] = []
        current = node_id
        while current is not None:
            path.append(self.position(current))
            current = self.parent[current]
        return list(reversed(path))

    def path_to_root_uids(self, node_id: int) -> List[int]:
        """Returns the path from node to root as a list of node UIDs."""
        path: List[int] = []
        current = node_id
        while current is not None:
            path.append(current)
            current = self.parent[current]
        return list(reversed(path))

    def to_checkpoint(self) -> Dict[str, Any]:
        return {
            "root": self.root,
            "next_node_id": self._next_node_id,
            "parent": {str(node_id): parent_id for node_id, parent_id in self.parent.items()},
            "node_ids": self.node_ids,
            "graph": nx.node_link_data(self.graph),
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: Dict[str, Any]) -> "SearchTree":
        tree = cls.__new__(cls)
        tree.graph = nx.node_link_graph(checkpoint["graph"])
        tree.parent = {
            int(node_id): parent_id if parent_id is None else int(parent_id)
            for node_id, parent_id in checkpoint["parent"].items()
        }
        tree.node_ids = [int(node_id) for node_id in checkpoint["node_ids"]]
        tree._next_node_id = int(checkpoint["next_node_id"])
        tree.root = int(checkpoint["root"])
        return tree