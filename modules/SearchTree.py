"""SearchTree class for all planners"""

from typing import Any, Dict, List, Optional, Tuple, Set
from collections import defaultdict

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from modules import node



class SearchTree:
    """A lightweight tree wrapper that stores nodes, parents, and positions."""

    def __init__(self, root_position: List[float], name: str):
        self.graph = nx.Graph()
        self.parent: Dict[int, Optional[int]] = {}
        self.children: Dict[int, List[int]] = defaultdict(list)
        
        self.node_ids: List[int] = []          # All nodes (keeps graph intact for visualization)
        self.active_nodes: Set[int] = set()    # Valid nodes only (used for KDTree nearest search)
        
        self._next_node_id = 0
        self.root: int = self.add_node(root_position, parent=None)
        self.name = name

    def _make_node_uid(self, node_id: int, position: List[float]) -> int:
        coord_parts = [str(coord).replace('.', '') for coord in position]
        return int(f"{node_id}{''.join(coord_parts)}")

    def add_node(self, position: List[float], parent: Optional[int]) -> int:
        node_id = self._next_node_id
        node_uid = self._make_node_uid(node_id, position)
        self.graph.add_node(node_uid, pos=position)
        
        self.node_ids.append(node_uid)
        self.active_nodes.add(node_uid) # Node is active by default
        self.parent[node_uid] = parent
        
        if parent is not None:
            self.children[parent].append(node_uid)
            self.graph.add_edge(parent, node_uid, status="unknown")
            
        self._next_node_id += 1
        return node_uid

    def position(self, node_id: int) -> List[float]:
        return self.graph.nodes[node_id]["pos"]

    def nearest(self, position: List[float]) -> Tuple[int, float]:
        """Returns the nearest ACTIVE node ID and the distance to it."""
        if not self.active_nodes:
            raise ValueError("No active nodes available in the tree.")
            
        active_list = list(self.active_nodes)
        
        if len(active_list) == 1:
            dist = np.linalg.norm(np.array(self.position(active_list[0])) - np.array(position))
            return active_list[0], float(dist)
            
        positions = [self.position(node_id) for node_id in active_list]
        kd_tree = cKDTree(positions)
        dist, index = kd_tree.query(position, k=1)
        
        return active_list[int(index)], float(dist)
    
    def invalidate_edge(self, u: int, v: int):
        """Marks an edge as invalid and deactivates the disconnected subtree."""
        self.graph[u][v]["status"] = "invalid"
        
        # Determine which node is the child in the directed tree
        child = v if self.parent.get(v) == u else u
        
        # Remove the child and all its descendants from active_nodes
        self._deactivate_subtree(child)

    def _deactivate_subtree(self, node_id: int):
        """Recursively removes nodes from the active set."""
        self.active_nodes.discard(node_id)
        for child in self.children.get(node_id, []):
            self._deactivate_subtree(child)

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
    
    def nodes_to_root(self, node_uid) -> List[node.Node]:
        path_uids = self.path_to_root_uids(node_uid)
        return [node.Node(uid,self.name,np.array(self.position(uid), dtype=float)) for uid in path_uids]
        

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
        
        # 1. Restore standard variables
        tree.parent = {
            int(node_id): parent_id if parent_id is None else int(parent_id)
            for node_id, parent_id in checkpoint["parent"].items()
        }
        tree.node_ids = [int(node_id) for node_id in checkpoint["node_ids"]]
        tree._next_node_id = int(checkpoint["next_node_id"])
        tree.root = int(checkpoint["root"])
        
        # 2. Reconstruct `children` and `active_nodes`
        tree.children = defaultdict(list)
        for child_id, parent_id in tree.parent.items():
            if parent_id is not None:
                tree.children[parent_id].append(child_id)
                
        # To determine active nodes, we look at the edges. 
        # Any node belonging to an 'invalid' edge (and its descendants) is inactive.
        tree.active_nodes = set(tree.node_ids)
        for u, v, data in tree.graph.edges(data=True):
            if data.get("status") == "invalid":
                child = v if tree.parent.get(v) == u else u
                tree._deactivate_subtree(child)
                
        return tree