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
        "goal_bias": 0.5
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
        tree_a_name: str,
        tree_b: SearchTree,
        node_b: int,
        tree_b_name: str,
    ) -> List[Node]:
        """Build a connection path as a list of :class:`Node` objects.

        The resulting list starts at the root of ``tree_a`` and ends at the
        root of ``tree_b`` and encodes, for each configuration along the
        candidate path, from which tree it originates.
        """

        path_a_uids = tree_a.path_to_root_uids(node_a)
        path_b_uids = tree_b.path_to_root_uids(node_b)

        path: List[Node] = []

        # path from tree A (root -> node_a)
        for uid in path_a_uids:
            pos = np.array(tree_a.position(uid), dtype=float)
            path.append(Node(uid, tree_a_name, pos))

        # path from tree B (node_b -> root) in reverse order so that the
        # concatenated list goes from node_a over to tree_b's root
        for uid in reversed(path_b_uids):
            pos = np.array(tree_b.position(uid), dtype=float)
            path.append(Node(uid, tree_b_name, pos))

        return path
    
    def _is_edge_valid(self, tree: SearchTree, node_a: int, node_b: int) -> bool:
        """Check if an edge between two nodes is valid (not marked invalid).
        Returns True if the edge is valid or unknown, False if it is invalid."""
        if not tree.graph.has_edge(node_a, node_b):
            return True
        status = tree.graph[node_a][node_b].get('status', 'unknown')
        return status != 'invalid'

    def _try_connect(
        self,
        active_tree: SearchTree,
        passive_tree: SearchTree,
        new_node_id: int,
        active_tree_name: str,
        passive_tree_name: str,
    ) -> Optional[List[Node]]:
        """SBL: Connect v (most recent node in active tree) to closest v' in passive tree.
        Only returns paths that use non-invalid edges.

        Returns a list of :class:`Node` objects describing the candidate
        connection path.
        """
        v_pos = active_tree.position(new_node_id)
        eta = float(self.config["eta"])
        
        # Find closest node in passive tree
        v_prime_id, distance = passive_tree.nearest(v_pos)
        
        # Only connect if within step size eta
        if distance >= eta:
            return None
        
        # Get candidate path as list of Node objects
        candidate_path = self._connection_path(
            active_tree,
            new_node_id,
            active_tree_name,
            passive_tree,
            v_prime_id,
            passive_tree_name,
        )

        # Verify all edges in the path are valid
        for idx in range(len(candidate_path) - 1):
            node1 = candidate_path[idx]
            node2 = candidate_path[idx + 1]

            # Determine which tree to check based on the *source* node
            if node1.tree == active_tree_name:
                tree = active_tree
            else:
                tree = passive_tree

            # Check if edge is valid
            if not self._is_edge_valid(tree, node1.id, node2.id):
                return None

        return candidate_path

    def _expand_tree(
            self,
            active_tree: SearchTree,
            passive_tree: SearchTree
        ) -> Optional[int]:
            """SBL Tree expansion with adaptive step-size (eta).

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
                if random.random() < float(self.config["goal_bias"]):
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

                    self.config["eta"] = eta_final
                    return active_tree.add_node(q_new_list, parent=v_id)

                # Collision: shrink eta locally for the next attempt in this loop
                eta = max(eta * eta_shrink, eta_min)

            # --- TOTAL FAILURE (All attempts collided) ---
            # Reset back to the standard baseline so we don't start the next call at eta_min
            self.config["eta"] = eta_standard
            return None



    def grow_trees(
        self,
        start: List[float],
        goal: List[float],
    ) -> Tuple[SearchTree, SearchTree, Optional[List[Node]]]:
        """Grow two search trees using SBL.

        Returns:
            T_start: search tree rooted at start
            T_goal: search tree rooted at goal
            path: unverified candidate path connecting the two trees
        """
        tree_start, tree_goal = self.init_trees(start,goal)
        start_tree, goal_tree, path = self.iterate_trees(tree_start, tree_goal)
        return start_tree, goal_tree, path

    def _build_checkpoint_frame(
        self,
        iteration: int,
        collision: bool,
        collision_index: Optional[int],
        path: Optional[List[Node]],
        start_tree: SearchTree,
        goal_tree: SearchTree,
    ) -> Dict:
        return {
            "iteration": iteration,
            "collision": collision,
            "collision_index": collision_index,
            # Store path as plain coordinates for checkpoint visualization.
            # The planner internally works with ``List[Node]`` objects, but
            # checkpoints only need the geometric layout, so we serialize each
            # node as its coordinate vector.
            "path": [node.coordinates.tolist() for node in path] if path is not None else None,
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
        ) -> Tuple[SearchTree, SearchTree, Optional[List[Node]]]:
        """Grow two search trees using SBL and optionally visualize / record.

        Returns
        -------
        T_start, T_goal:
            The two search trees rooted at ``start`` and ``goal``.
        path:
            Unverified candidate path connecting the two trees, represented as
            a ``List[Node]`` (or ``None`` if no connection was found).
        """
        tree_start, tree_goal = self.init_trees(start,goal)
        checkpoint_frames: List[Dict] = []

        for n_iter in range(self.config["iterations"]):
            print(f"Iteration {n_iter}")
            start_tree, goal_tree, path = self.iterate_trees(tree_start, tree_goal)
            collision = False
            collision_index = None
            if path:
                collision, collision_index, start_tree, goal_tree = self.collision_check_solution(start_tree, goal_tree, path)
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
                
            if not collision:
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

    def iterate_trees(self, tree_start: SearchTree, tree_goal: SearchTree) -> Tuple[SearchTree, SearchTree, Optional[List[Node]]]:
        """
        Perform 1 iteration of expanding tree and checking connectivity
        """
        for iteration in range(int(self.config["max_nodes"])):
            # 1. Pick a tree to expand at random with probability P=0.5
            if random.random() < 0.5:
                active, passive = tree_start, tree_goal
                active_name, passive_name = "start", "goal"
            else:
                active, passive = tree_goal, tree_start
                active_name, passive_name = "goal", "start"

            # 2. Expand the active tree by creating a single node
            new_node = self._expand_tree(active, passive)

            # If no new node was added (e.g., sample in collision), skip this iteration
            if new_node is None:
                continue

            # 3. Attempt to connect the trees based on proximity
            connection = self._try_connect(active, passive, new_node, active_name, passive_name)
            if connection is not None:
                # Returns the candidate path as a list of Node objects
                return tree_start, tree_goal, connection

        return tree_start, tree_goal, None

    def collision_check_solution(
            self,
            tree_start: SearchTree, tree_goal: SearchTree,
            connection: List[Node],
        ) -> Tuple[bool, Optional[int], SearchTree, SearchTree]:
        """
        Adaptive collision check for a candidate path.

        Parameters
        ----------
        tree_start, tree_goal:
            The two search trees grown by the planner.
        connection:
            Candidate path represented as ``List[Node]`` objects.

        Returns
        -------
        collision:
            ``True`` if a collision was detected along the path.
        collision_index:
            Index of the colliding segment in the path (or ``None``).
        tree_start, tree_goal:
            Potentially updated trees with edge status / pruning applied.
        """
        unchecked_nodes: List[Tuple[int, Node, Node]] = []
        # Determine the tree from which the candidate path originates (first node)
        first_node_tree = connection[0].tree

        for node_idx in range(0, len(connection) - 1):
            node1 = connection[node_idx]
            node2 = connection[node_idx + 1]

            if node1.tree != node2.tree:
                # Nodes belong to different trees: connection between both trees
                unchecked_nodes.append((node_idx, node1, node2))
            else:
                # Check if edge was already validated (same tree)
                tree = tree_start if node1.tree == "start" else tree_goal

                if not tree.graph.has_edge(node1.id, node2.id):
                    raise Warning(f"Edge not found in {node1.tree} tree between nodes {node1.id} - {node2.id}")
                    unchecked_nodes.append((node_idx, node1, node2))
                else:
                    edge_status = tree.graph[node1.id][node2.id]["status"]
                    if edge_status == "valid":
                        # Jump to next iteration (next edge)
                        continue
                    elif edge_status == "invalid":
                        print(f"Invalid edge in {node1.tree} tree between {node1.id} - {node2.id}")
                        return True, node_idx, tree_start, tree_goal
                    else:
                        unchecked_nodes.append((node_idx, node1, node2))

        for idx, (segment_index, node1, node2) in enumerate(unchecked_nodes):
            # Check collisions of unknown edges
            collision, checkedPoints = adaptive_local_collision_check(
                node1.coordinates,
                node2.coordinates,
                self._collisionChecker,
                self.config["kappa_max"],
                self.config["epsilon"],
            )

            #TODO: add option to visualize checked points

            # mark edge as invalid
            if node1.tree == node2.tree == "start":
                # start tree
                tree_start.graph[node1.id][node2.id]["status"] = ["valid", "invalid"][collision]
            elif node1.tree == node2.tree == "goal":
                # goal tree
                tree_goal.graph[node1.id][node2.id]["status"] = ["valid", "invalid"][collision]
                
            # return if collision found
            if collision:
                print(f"Collision found between {node1} - {node2}")
                # Soft pruning: if the collision occurs in the same tree from which
                # the candidate path originates, mark the subtree behind node2 as
                # unreachable (all its edges set to invalid).
                if first_node_tree == node2.tree:
                    target_tree = tree_start if node2.tree == "start" else tree_goal
                    target_tree.mark_unreachable_subtree(node2.id)
                print(f"#Collision checks: {idx}")
                return True, segment_index, tree_start, tree_goal
            
        # no invalid edges and no collisions found
        print(f"#Collision checks: {len(unchecked_nodes)}")
        return False, None, tree_start, tree_goal