"""Bidirectional search tree utilities for sampling-based motion planning."""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from numbers import Number

import networkx as nx
import numpy as np
from matplotlib.axes import Axes

try:
    from lecture_examples.IPPRMBase import PRMBase
except ImportError:
    from IPPRMBase import PRMBase

from modules.SearchTree import SearchTree
from modules import draw
from modules.node import Node
from lecture_examples import IPEnvironment
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
        "goal_bias": 0.5,
        "repair_bias": 0.6    # Probability to sample near a broken path    
    }

    def __init__(self, coll_checker: IPEnvironment.CollisionChecker, config: Optional[Dict[str,Number]] = {}):
        self.config =  self._merge_config(config)
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
    ) -> Optional[List[List[float]]]:
        """SBL: Connect v (most recent node in active tree) to closest v' in passive tree.
        Only returns paths that use non-invalid edges.
        """
        v_pos = active_tree.position(new_node_id)
        eta = float(self.config["eta"])
        
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
            repair_focus: Optional[List[float]] = None
        ) -> Optional[int]:
        """SBL Tree expansion with adaptive step-size (eta) and local repair sampling.

        On collision, shrink eta and retry. On success, grow eta and return the new node.
        """

        eta_standard = float(self.config.get("standard_eta", 2.0))

        # Start with whatever value eta currently is
        eta = float(self.config["eta"])

        eta_min = float(self.config.get("eta_min", 0.05))
        eta_max = float(self.config.get("eta_max", 5.0))
        eta_shrink = float(self.config.get("eta_shrink", 0.5))
        eta_grow = float(self.config.get("eta_grow", 1.2))
        attempts = int(self.config.get("expand_attempts", 5))

        for _ in range(attempts):
            # Random sample within the environment limits with bias toward the passive tree root.
            p = random.random()
            repair_prob = float(self.config.get("repair_bias", 0.6)) if repair_focus is not None else 0.0

            if p < repair_prob:
                # 1. Local repair using the CURRENT adaptive step size (eta)
                # If attempts fail and eta shrinks, the repair sampling tightly focuses near the valid node
                q_rand = np.array(repair_focus) + np.random.uniform(-eta, eta, size=len(repair_focus))
            else:
                # 2. Standard exploration (Goal bias vs completely Random)
                p_standard = random.random()
                if p_standard < float(self.config["goal_bias"]):
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
                eta_final = min(max(eta * eta_grow, eta_standard), eta_max)
                self.config["eta"] = eta_final
                return active_tree.add_node(q_new_list, parent=v_id)

            # Collision: shrink eta locally for the next attempt in this loop
            eta = max(eta * eta_shrink, eta_min)

        # --- TOTAL FAILURE (All attempts collided) ---
        # Reset back to the standard baseline so we don't start the next call at eta_min
        self.config["eta"] = eta_standard
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
    ) -> Tuple[nx.Graph, nx.Graph, Optional[List[List[float]]]]:
        """Grow two search trees using SBL."""
        tree_start, tree_goal = self.init_trees(start, goal)
        start_tree, goal_tree, path = self.iterate_trees(tree_start, tree_goal)
        return start_tree.graph, goal_tree.graph, path

    def _build_checkpoint_frame(
        self,
        iteration: int,
        collision: bool,
        collision_index: Optional[int],
        path: Optional[List[List[float]]],
        start_tree: SearchTree,
        goal_tree: SearchTree,
    ) -> Dict:
        return {
            "iteration": iteration,
            "collision": collision,
            "collision_index": collision_index,
            "path": path,
            "trees": {
                "start": start_tree.to_checkpoint(),
                "goal": goal_tree.to_checkpoint(),
            },
        }

    def _write_checkpoint_file(self, checkpoint_path: str, payload: Dict) -> None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2)
    
    def plan_path(
        self,
        start: List[float],
        goal: List[float],
        ax: Optional[Axes],
        checkpoint_path: Optional[str] = None,
        ) -> Tuple[SearchTree, SearchTree, Optional[List[List[float]]]]:
        """Grow two search trees using SBL.

        Returns:
            T_start: search tree rooted at start
            T_goal: search tree rooted at goal
            path: unverified candidate path connecting the two trees
        """
        tree_start, tree_goal = self.init_trees(start, goal)
        checkpoint_frames: List[Dict] = []
        
        repair_focus = None

        for n_iter in range(self.config["iterations"]):
            print(f"Iteration {n_iter}")
            
            # Pass repair_focus to iteration
            start_tree, goal_tree, path = self.iterate_trees(tree_start, tree_goal, repair_focus)
            
            collision = False
            collision_index = None
            
            if path:
                # Unpack 5 values to support both the graph tools and the repair focus
                collision, collision_index, start_tree, goal_tree, new_focus = self.collision_check_solution(start_tree, goal_tree, path)
                
                # Update focus point for the next iteration if the path broke
                repair_focus = new_focus if collision else None
            else:
                print("no path found")
                
            if ax:
                # draw iteration
                draw.plot_iteration(ax, start_tree, goal_tree, path, collision, collision_index)

            if checkpoint_path:
                checkpoint_frames.append(
                    self._build_checkpoint_frame(
                        n_iter,
                        collision,
                        collision_index,
                        path,
                        start_tree,
                        goal_tree,
                    )
                )
                self._write_checkpoint_file(
                    checkpoint_path,
                    {
                        "metadata": {
                            "start": start,
                            "goal": goal,
                            "config": self.config,
                        },
                        "frames": checkpoint_frames,
                    },
                )
                
            # Fixed early termination bug: only return if a valid path actually exists
            if path and not collision:
                return start_tree, goal_tree, path
            
        return start_tree, goal_tree, None

    def init_trees(
        self,
        start: List[float],
        goal: List[float],
        ):
        checked_start, checked_goal = self._checkStartGoal([start], [goal])
        tree_start = SearchTree(checked_start[0])
        tree_goal = SearchTree(checked_goal[0])
        self._tree_start = tree_start
        self._tree_goal = tree_goal
        return tree_start, tree_goal

    def iterate_trees(self, 
        tree_start: SearchTree, 
        tree_goal: SearchTree, 
        repair_focus: Optional[List[float]] = None
    ) -> Tuple[SearchTree, SearchTree, Optional[List[List[float]]]]:
        """
        Perform 1 iteration of expanding tree and checking connectivity
        """
        for iteration in range(int(self.config["max_nodes"])):
            # 1. Pick a tree to expand at random with probability P=0.5
            if random.random() < 0.5:
                active, passive = tree_start, tree_goal
            else:
                active, passive = tree_goal, tree_start

            # 2. Expand the active tree by creating a single node
            new_node = self._expand_tree(active, passive, self.config, repair_focus)

            # If no new node was added (e.g., sample in collision), skip this iteration
            if new_node is None:
                continue

            # 3. Attempt to connect the trees based on proximity
            connection = self._try_connect(active, passive, new_node, self.config)
            if connection is not None:
                # Returns the candidate path layout; the next module will evaluate edge validity.
                return tree_start, tree_goal, connection

        return tree_start, tree_goal, None

    def collision_check_solution(
            self,
            tree_start: SearchTree, tree_goal: SearchTree,
            connection: List[List[float]],
        ) -> Tuple[bool, Optional[int], SearchTree, SearchTree, Optional[List[float]]]:
        """
        Adaptive collision check for a candidate path.
        Returns (collision_bool, collision_index, tree_start, tree_goal, repair_focus).
        """
        unchecked_nodes = []
        
        # Determine the tree from which the candidate path originates (first node)
        first_node = np.array(connection[0])
        _, first_node_tree = self._get_node_uid(first_node)
        
        for node_id in range(0, len(connection) - 1):
            node1 = np.array(connection[node_id])
            node2 = np.array(connection[node_id + 1])
            node1_uid, node1_tree = self._get_node_uid(node1)
            node2_uid, node2_tree = self._get_node_uid(node2)

            # 1. Connection between both trees (Bridge edge)
            if node1_tree != node2_tree:
                unchecked_nodes.append((node_id,
                                        Node(node1_uid, node1_tree, node1),
                                        Node(node2_uid, node2_tree, node2)))
            
            # 2. Check if edge was already validated/invalidated in start tree
            elif node1_tree == "start":
                edge_status = tree_start.graph[node1_uid][node2_uid]["status"]
                if edge_status == "valid":
                    continue
                elif edge_status == "invalid":
                    print(f"Invalid edge in start tree between {node1} - {node2}")
                    # node1 is the valid parent traversing away from start root
                    return True, node_id, tree_start, tree_goal, node1.tolist()
                else:
                    unchecked_nodes.append((node_id,
                                            Node(node1_uid, node1_tree, node1),
                                            Node(node2_uid, node2_tree, node2)))
                
            # 3. Check if edge was already validated/invalidated in goal tree
            elif node1_tree == "goal":
                edge_status = tree_goal.graph[node1_uid][node2_uid]["status"]
                if edge_status == "valid":
                    continue
                elif edge_status == "invalid":
                    print(f"Invalid edge in goal tree between {node1} - {node2}")
                    # node2 is the valid parent traversing towards goal root
                    return True, node_id, tree_start, tree_goal, node2.tolist()
                else:
                    unchecked_nodes.append((node_id,
                                            Node(node1_uid, node1_tree, node1),
                                            Node(node2_uid, node2_tree, node2)))
            else:
                raise ValueError(f"Unknown tree: {node1_tree}")

        # Check collisions of unknown edges lazily
        checks_performed = 0
        for idx, (segment_index, node1, node2) in enumerate(unchecked_nodes):
            checks_performed += 1
            collision, checkedPoints = adaptive_local_collision_check(
                node1.coordinates, node2.coordinates, 
                self._collisionChecker, self.config["kappa_max"], self.config["epsilon"]
            )

            # TODO: add option to visualize checked points

            repair_focus = None

            # Mark edge as valid/invalid and prune active nodes
            if node1.tree == node2.tree == "start":
                if collision:
                    tree_start.invalidate_edge(node1.id, node2.id)
                    repair_focus = node1.coordinates.tolist() # Parent is node1
                else:
                    tree_start.graph[node1.id][node2.id]["status"] = "valid"
                    
            elif node1.tree == node2.tree == "goal":
                if collision:
                    tree_goal.invalidate_edge(node1.id, node2.id)
                    repair_focus = node2.coordinates.tolist() # Parent is node2
                else:
                    tree_goal.graph[node1.id][node2.id]["status"] = "valid"
                    
            else:
                # Collision on the bridge connecting the two trees
                if collision:
                    # The bridge isn't natively in either tree's edges, so no tree pruning.
                    # We just focus repair on the start tree's side of the gap.
                    repair_focus = node1.coordinates.tolist()

            # Return immediately if collision found
            if collision:
                print(f"Collision found between {node1.coordinates} - {node2.coordinates}")
                print(f"#Collision checks: {checks_performed}")
                return True, segment_index, tree_start, tree_goal, repair_focus
            
        # No invalid edges and no collisions found
        print(f"#Collision checks: {checks_performed}")
        return False, None, tree_start, tree_goal, None