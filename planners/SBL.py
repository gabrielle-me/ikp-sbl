"""Bidirectional search tree utilities for sampling-based motion planning."""

import random
from typing import Dict, List, Optional, Tuple
from numbers import Number

import networkx as nx
import numpy as np

try:
    from lecture_examples.IPPRMBase import PRMBase
except ImportError:
    from IPPRMBase import PRMBase

from modules.SearchTree import SearchTree
from modules.adaptiveLocalCollisionCheck import adaptive_local_collision_check


class BidirectionalSBL(PRMBase):
    """SBL planner that grows two trees lazily without local collision checking."""

    DEFAULT_CONFIG = {
        "max_nodes": 500,
        "eta": 1.0,
        "standard_eta": 1.5,
        "eta_min": 0.05,
        "eta_max": 5.0,
        "eta_shrink": 0.5,
        "eta_grow": 1.2,
        "expand_attempts": 5,
        "iterations": 10,
        "epsilon": 0.05,
        "kappa_max": 10,
        "goal_bias": 0.5
    }

    def __init__(self, coll_checker):
        super(BidirectionalSBL, self).__init__(coll_checker)

    @staticmethod
    def _merge_config(config: Optional[Dict[str, Number]]) -> Dict[str, Number]:
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
    
    def _is_edge_valid(self, tree: SearchTree, config1: List[float], config2: List[float]) -> bool:
        """Check if an edge between two configurations is valid (not marked invalid).
        Returns True if the edge is valid or unknown, False if it is invalid."""
        config1_tuple = tuple(config1)
        config2_tuple = tuple(config2)
        
        for u, v in tree.graph.edges():
            u_pos = tuple(tree.position(u))
            v_pos = tuple(tree.position(v))
            
            if (u_pos == config1_tuple and v_pos == config2_tuple) or \
               (u_pos == config2_tuple and v_pos == config1_tuple):
                status = tree.graph[u][v].get('status', 'unknown')
                return status != 'invalid'
        
        # Edge not found in tree (shouldn't happen for valid path traversal)
        return True

    def _try_connect(
        self,
        active_tree: SearchTree,
        passive_tree: SearchTree,
        new_node_id: int,
        config: Dict[str, float],
    ) -> Optional[List[List[float]]]:
        """SBL: Connect v (most recent node in active tree) to closest v' in passive tree.
        Only returns paths that use non-invalid edges.
        """
        v_pos = active_tree.position(new_node_id)
        eta = float(config["eta"])
        
        # Find closest node in passive tree
        v_prime_id, distance = passive_tree.nearest(v_pos)
        
        # Only connect if within step size eta
        if distance >= eta:
            return None
        
        # Get candidate path
        candidate_path = self._connection_path(active_tree, new_node_id, passive_tree, v_prime_id)
        
        # Verify all edges in the path are valid
        for idx in range(len(candidate_path) - 1):
            config1 = candidate_path[idx]
            config2 = candidate_path[idx + 1]
            
            # Check validity in both trees
            if idx < len(active_tree.path_to_root(new_node_id)) - 1:
                # This segment is in active tree
                if not self._is_edge_valid(active_tree, config1, config2):
                    return None
            else:
                # This segment is in passive tree
                if not self._is_edge_valid(passive_tree, config1, config2):
                    return None
        
        return candidate_path

    def _expand_tree(
            self,
            active_tree: SearchTree,
            passive_tree: SearchTree,
            config: Dict[str, Number]
        ) -> Optional[int]:
            """SBL Tree expansion with adaptive step-size (eta).

            On collision, shrink eta and retry. On success, grow eta and return the new node.
            """

            eta_standard = float(config.get("standard_eta", 2.0))

            # Start with whatever value eta currently is
            eta = float(config["eta"])

            eta_min = float(config.get("eta_min", 0.05))
            eta_max = float(config.get("eta_max", 5.0))
            eta_shrink = float(config.get("eta_shrink", 0.5))
            eta_grow = float(config.get("eta_grow", 1.2))
            attempts = int(config.get("expand_attempts", 5))

            for _ in range(attempts):
                # Random sample within the environment limits with bias toward the passive tree root.
                if random.random() < float(config["goal_bias"]):
                    q_rand = np.array(passive_tree.position(passive_tree.root), dtype=float)
                else:
                    q_rand = np.array(self._getRandomFreePosition(), dtype=float)

                # Choose a near node v from the tree based on distance to q_rand
                v_id, distance = active_tree.nearest(q_rand.tolist())
                v_pos = np.array(active_tree.position(v_id), dtype=float)

                # Expand with step size eta
                if distance <= eta:
                    q_new = q_rand
                else:
                    direction = (q_rand - v_pos) / distance
                    q_new = v_pos + direction * eta

                q_new_list = q_new.tolist()
                if not self._collisionChecker.pointInCollision(q_new_list):
                    # Success: increase eta for future expansions and return the new node.
                    # 1. Calculate the grown eta locally
                    eta_increased = eta * eta_grow
                    
                    # 2. Ensure it is AT LEAST the standard value, but AT MOST the max value
                    eta_final = min(max(eta_increased, eta_standard), eta_max)

                    config["eta"] = eta_final
                    return active_tree.add_node(q_new_list, parent=v_id)

                # Collision: shrink eta locally for the next attempt in this loop
                eta = max(eta * eta_shrink, eta_min)

            # --- TOTAL FAILURE (All attempts collided) ---
            # Reset back to the standard baseline so we don't start the next call at eta_min
            config["eta"] = eta_standard
            return None

    def _get_node_uid(self, node) -> Tuple[int, str]:
        if isinstance(node, np.ndarray):
            node_tuple = tuple(node.tolist())
        else:
            node_tuple = tuple(node)

        for tree_name, tree in [("start", self._tree_start), ("goal", self._tree_goal)]:
            for node_uid, data in tree.graph.nodes(data=True):
                if tuple(data["pos"]) == node_tuple:
                    return node_uid, tree_name

        raise ValueError(f"Node {node_tuple} not found in either tree")

    def grow_trees(
        self,
        start: List[float],
        goal: List[float],
        config: Optional[Dict[str, Number]] = None,
    ) -> Tuple[nx.Graph, nx.Graph, Optional[List[List[float]]]]:
        """Grow two search trees using SBL.

        Returns:
            T_start: search tree rooted at start
            T_goal: search tree rooted at goal
            path: unverified candidate path connecting the two trees
        """
        tree_start, tree_goal = self.init_trees(start,goal,config)
        start, goal, path = self.iterate_trees(tree_start, tree_goal, config)
        return start.graph, goal.graph, path
    
    def plan_path(
        self,
        start: List[float],
        goal: List[float],
        config: Optional[Dict[str, Number]] = None,
        ) -> Tuple[nx.Graph, nx.Graph, Optional[List[List[float]]]]:
        """Grow two search trees using SBL.

        Returns:
            T_start: search tree rooted at start
            T_goal: search tree rooted at goal
            path: unverified candidate path connecting the two trees
        """
        tree_start, tree_goal = self.init_trees(start,goal,config)

        for n_iter in range(config["iterations"]):
            start, goal, path = self.iterate_trees(tree_start, tree_goal, config)
            collision, start, goal = self.collision_check_solution(start, goal, path, config)
            if not collision:
                return start.graph, goal.graph, path
            
        return start.graph, goal.graph, None

    def init_trees(
        self,
        start: List[float],
        goal: List[float],
        config: Optional[Dict[str, Number]] = None,
        ):
        config = self._merge_config(config)
        checked_start, checked_goal = self._checkStartGoal([start], [goal])
        tree_start = SearchTree(checked_start[0])
        tree_goal = SearchTree(checked_goal[0])
        self._tree_start = tree_start
        self._tree_goal = tree_goal
        return tree_start, tree_goal

    def iterate_trees(self, tree_start: SearchTree, tree_goal: SearchTree, config: Dict[str, float]) -> Tuple[SearchTree, SearchTree, Optional[List[List[float]]]]:
        """
        Perform 1 iteration of expanding tree and checking connectivity
        """
        for iteration in range(int(config["max_nodes"])):
            # 1. Pick a tree to expand at random with probability P=0.5
            if random.random() < 0.5:
                active, passive = tree_start, tree_goal
            else:
                active, passive = tree_goal, tree_start

            # 2. Expand the active tree by creating a single node
            new_node = self._expand_tree(active, passive, config)

            # If no new node was added (e.g., sample in collision), skip this iteration
            if new_node is None:
                continue

            # 3. Attempt to connect the trees based on proximity
            connection = self._try_connect(active, passive, new_node, config)
            if connection is not None:
                # Returns the candidate path layout; the next module will evaluate edge validity.
                return tree_start, tree_goal, connection

        return tree_start, tree_goal, None

    def collision_check_solution(
            self,
            tree_start: SearchTree, tree_goal: SearchTree,
            connection: List[List[float]],
            config: Dict[str, Number]
        ) -> Tuple[bool, SearchTree, SearchTree]:
        """
        Adaptive collision check for a candidate path.
        Returns whether collision happens (True) & returns start/goal tree with marked edges
        """
        for node_id in range(0, len(connection) - 1):
            node1 = np.array(connection[node_id])
            node2 = np.array(connection[node_id + 1])
            node1_uid, node1_tree = self._get_node_uid(node1)
            node2_uid, node2_tree = self._get_node_uid(node2)

            # Doublecheck if both nodes belong to same tree
            if node1_tree != node2_tree:
                raise ValueError(f"Nodes belong to different trees: {node1_uid}: {node1_tree}, {node2_uid}: {node2_tree}")
            
            # Check if edge was already validated
            if node1_tree == "start":
                # start tree
                edge_status = tree_start.graph[node1_uid][node2_uid]["status"]
                if edge_status == "valid":
                    # Jump to next iteration (next edge)
                    continue
                elif edge_status == "invalid":
                    return True, tree_start, tree_goal
                
            else:
                # goal tree
                edge_status = tree_goal.graph[node1_uid][node2_uid]["status"]
                if edge_status == "valid":
                    # Jump to next iteration (next edge)
                    continue
                elif edge_status == "invalid":
                    return True, tree_start, tree_goal

            # Check collision otherwise
            collision, checkedPoints = adaptive_local_collision_check(node1, node2, self._collisionChecker, config["kappa_max"], config["epsilon"])

            #TODO: add option to visualize checked points

            # mark edge as invalid
            if node1_tree == "start":
                # start tree
                tree_start.graph[node1_uid][node2_uid]["status"] = ["valid","invalid"][collision]
            else:
                # goal tree
                tree_goal.graph[node1_uid][node2_uid]["status"] = ["valid","invalid"][collision]
                
            # return if collision found
            if collision:
                return True, tree_start, tree_goal
            
        # no invalid edges and no collisions found
        return False, tree_start, tree_goal
    
    