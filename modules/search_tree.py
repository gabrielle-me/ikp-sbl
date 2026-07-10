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


class BidirectionalSBL(PRMBase):
    """SBL planner that grows two trees lazily without local collision checking."""
    #TODO: check if distance should change based on narrow passage or free space (kappa at the end?)

    DEFAULT_CONFIG = {
        "max_nodes": 500,
        "eta": 2.0,            # Step size for tree expansion (maximum distance between nodes)
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
        new_node_id: Optional[int],
        config: Dict[str, float],
    ) -> Optional[List[List[float]]]:
        """SBL: Connect v (most recent node in active tree) to closest v' in passive tree."""
        if new_node_id is None:
            return None

        v_pos = active_tree.position(new_node_id)
        eta = float(config["eta"])
        
        # Find closest node in passive tree
        v_prime_id, distance = passive_tree.nearest(v_pos)
        
        # Only connect if within threshold
        if distance >= eta:
            return None

        # Return candidate path for lazy collision checking.
        return self._connection_path(active_tree, new_node_id, passive_tree, v_prime_id)

    def _expand_tree(
            self,
            tree: SearchTree,
            config: Dict[str, float],
        ) -> Optional[int]:
            """SBL Tree expansion using elements from the RRT lecture code (IPRRT.py)."""
            eta = float(config["eta"])
            
            # Random sample within the environment limits
            # Dynamically handles n-DoF limits
            q_rand = np.array(self._getRandomFreePosition(), dtype=float)
            
            # Choose a near node v from the tree based on distance to q_rand
            v_id, distance = tree.nearest(q_rand.tolist())
            v_pos = np.array(tree.position(v_id), dtype=float)
            
            # Expand with step size eta
            if distance <= eta:
                q_new = q_rand
            else:
                # Step exactly distance eta along the direction vector from v_pos to q_rand
                direction = (q_rand - v_pos) / distance
                q_new = v_pos + direction * eta
        
            # Add the new node to the tree if it is not in collision
            if not self._collisionChecker.pointInCollision(q_new.tolist()):
                return tree.add_node(q_new.tolist(), parent=v_id)

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