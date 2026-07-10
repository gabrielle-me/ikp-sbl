"""Bidirectional search tree utilities for sampling-based motion planning."""

import random
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

try:
    from lecture_examples.IPPRMBase import PRMBase
except ImportError:
    from IPPRMBase import PRMBase


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
            self.graph.add_edge(parent, node_id)
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


class BidirectionalSBL(PRMBase):
    """SBL planner that grows two trees lazily without local line collision checking."""

    DEFAULT_CONFIG = {
        "max_nodes": 500,
        "rho": 6.0,            # Neighborhood radius threshold
    }

    def __init__(self, coll_checker):
        super(BidirectionalSBL, self).__init__(coll_checker)

    @staticmethod
    def _merge_config(config: Optional[Dict[str, float]]) -> Dict[str, float]:
        merged = BidirectionalSBL.DEFAULT_CONFIG.copy()
        if config:
            merged.update(config)
        return merged

    def _connection_path(
        self,
        tree_a: SearchTree,
        node_a: int,
        tree_b: SearchTree,
        node_b: int,
    ) -> List[List[float]]:
        path_a = tree_a.path_to_root(node_a)
        path_b = tree_b.path_to_root(node_b)
        return path_a + list(reversed(path_b))

    def _try_connect(
        self,
        active_tree: SearchTree,
        passive_tree: SearchTree,
        new_node_id: int,
        config: Dict[str, float],
    ) -> Optional[List[List[float]]]:
        """SBL: Connect trees if within radius rho (WITHOUT checking edge collision)."""
        new_position = active_tree.position(new_node_id)
        nearest_passive_id, distance = passive_tree.nearest(new_position)
        
        # Check if the passive tree has a node close enough to connect
        if distance < float(config["rho"]):
            return self._connection_path(active_tree, new_node_id, passive_tree, nearest_passive_id)
        return None

    def _expand_tree(
            self,
            tree: SearchTree,
            config: Dict[str, float],
        ) -> int:
            """SBL: Pick a node weighted inversely by its local density, 
            then sample a new free point within its local neighborhood radius rho.
            """
            rho = float(config["rho"])
            
            # --- 1. COMPUTE CELL-BASED DENSITY FOR EVERY NODE ---
            # We define a cell size matching our neighborhood radius rho
            cell_size = rho 
            
            # Track which cell each node belongs to, and count cell populations
            cell_counts = {}
            node_cells = {}
            
            for v_id in tree.node_ids:
                pos = np.array(tree.position(v_id), dtype=float)
                # Find the discrete grid coordinates for this node's position
                cell_coord = tuple(np.floor(pos / cell_size).astype(int))
                
                node_cells[v_id] = cell_coord
                cell_counts[cell_coord] = cell_counts.get(cell_coord, 0) + 1
                
            # Compute weights inversely proportional to cell density: weight = 1 / eta(v)
            # (Where eta(v) is the number of nodes sharing that node's grid cell)
            weights = [1.0 / cell_counts[node_cells[v_id]] for v_id in tree.node_ids]
            
            # --- 2. SAMPLE A NODE BASED ON WEIGHTS ---
            limits = self._collisionChecker.getEnvironmentLimits()
            for _ in range(1000):
                # Pick a node v using the calculated density-based weights
                v_id = random.choices(tree.node_ids, weights=weights, k=1)[0]
                v_pos = np.array(tree.position(v_id), dtype=float)
                
                # 3. Sample a random configuration q uniformly from B(v, rho)
                dim = len(v_pos)
                offset = np.random.uniform(-rho, rho, size=dim)
                
                # Ensure the sample falls strictly within the hypersphere of radius rho
                if np.linalg.norm(offset) <= rho:
                    q_pos = np.array(v_pos + offset, dtype=float)
                    lower = np.array([limits[0][0], limits[1][0]], dtype=float)
                    upper = np.array([limits[0][1], limits[1][1]], dtype=float)

                    # Keep points strictly inside the workspace bounds.
                    # We shrink the interval slightly so the exact boundary values are avoided.
                    margin = 0.1
                    lower = lower + margin
                    upper = upper - margin

                    q_pos = np.clip(q_pos, lower, upper)
                    q_pos_list = q_pos.tolist()

                    # Check only if the newly sampled node itself is collision-free
                    if not self._collisionChecker.pointInCollision(q_pos_list):
                        return tree.add_node(q_pos_list, parent=v_id)

            raise RuntimeError("Unable to sample a valid point within the configured limits")

    def grow_trees(
        self,
        start: List[float],
        goal: List[float],
        config: Optional[Dict[str, float]] = None,
    ) -> Tuple[nx.Graph, nx.Graph, Optional[List[List[float]]]]:
        """Grow two search trees using SBL.

        Returns:
            T_start: search tree rooted at start
            T_goal: search tree rooted at goal
            path: unverified candidate path connecting the two trees
        """
        config = self._merge_config(config)
        checked_start, checked_goal = self._checkStartGoal([start], [goal])
        tree_start = SearchTree(checked_start[0])
        tree_goal = SearchTree(checked_goal[0])

        for iteration in range(int(config["max_nodes"])):
            # 1. Pick a tree to expand at random with probability P=0.5
            if random.random() < 0.5:
                active, passive = tree_start, tree_goal
            else:
                active, passive = tree_goal, tree_start
                
            # 2. Expand the active tree by creating a single node
            new_node = self._expand_tree(active, config)
            
            # 3. Attempt to connect the trees based on proximity
            connection = self._try_connect(active, passive, new_node, config)
            if connection is not None:
                # Returns the candidate path layout; the next module will evaluate edge validity.
                return tree_start.graph, tree_goal.graph, connection

        return tree_start.graph, tree_goal.graph, None

    def build_trees(
        self,
        start: List[float],
        goal: List[float],
        config: Optional[Dict[str, float]] = None,
    ) -> Tuple[nx.Graph, nx.Graph]:
        """Generate the two search trees and return them without requiring a path connection."""
        tree_start, tree_goal, _ = self.grow_trees(start, goal, config)
        return tree_start, tree_goal