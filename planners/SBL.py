"""Bidirectional search tree utilities for sampling-based motion planning."""

# General imports
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from numbers import Number
import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from tqdm import tqdm

# Lecture examples imports
from lecture_examples.IPPRMBase import PRMBase
from lecture_examples.IPPerfMonitor import IPPerfMonitor
from lecture_examples import IPEnvironment

# Module imports
from modules.SearchTree import SearchTree
from modules.node import Node
from modules.adaptiveLocalCollisionCheck import LineChecker, AdaptiveLineChecker
from modules.PlannerStats import PlannerStats


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
        "collision_check": {
            "adaptive": True,
            "epsilon": 0.05,
            "steps": 50},
        "goal_bias": 0.5,
        "repair_bias": 0.6,    # Probability to sample near a broken path
        "count_edge_checks": False,
        "checkpoint_path": None,
    }

    def __init__(self, coll_checker: IPEnvironment.CollisionChecker, config: Optional[Dict[str,Number]] = {}):
        self.config =  self._merge_config(config)
        super(BidirectionalSBL, self).__init__(coll_checker)
        
        # Initialize attributes so they always exist for the visualizer
        self.startTree = None
        self.goalTree = None
        self.collision_check_counter = {}
        self._collisionCheckFun = [LineChecker(coll_checker, self.config["collision_check"]), AdaptiveLineChecker(coll_checker, self.config["collision_check"])][self.config["collision_check"]["adaptive"]]
                                       
    
    @staticmethod
    def _merge_config(config: Optional[Dict[str, Number]]) -> Dict[str, Number]:
        merged = BidirectionalSBL.DEFAULT_CONFIG.copy()
        if config:
            merged.update(config)
        return merged

    def _try_connect(
        self,
        active_tree: SearchTree,
        passive_tree: SearchTree,
        new_node_id: int,
    ) -> Tuple[Optional[List[Node]],Optional[List[Node]]]:
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
            return None, None
        
        
        # Get candidate path as list of Node objects
        path_a = active_tree.nodes_to_root(new_node_id)
        path_b = passive_tree.nodes_to_root(v_prime_id)

        path_b_rev = path_b[::-1]
        return path_a, path_b_rev

    @IPPerfMonitor
    def _expand_tree(
            self,
            active_tree: SearchTree,
            passive_tree: SearchTree,
            repair_focus: Optional[List[float]] = None
        ) -> Optional[int]:
        """SBL Tree expansion with adaptive step-size (eta) and local repair sampling.

        On collision, shrink eta and retry. On success, grow eta and return the new node.
        """
        if not hasattr(self, 'stats'):
            self.stats = PlannerStats()

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
            # Quantity of point collision tests
            self.stats.point_collision_tests += 1
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

    def grow_trees(
        self,
        start: List[float],
        goal: List[float],
    ) -> Tuple[SearchTree, SearchTree, Optional[List[Node]], Optional[List[Node]]]:
        """Grow two search trees using SBL.

        Returns:
            T_start: search tree rooted at start
            T_goal: search tree rooted at goal
            path: unverified candidate path connecting the two trees
        """
        tree_start, tree_goal = self.init_trees(start,goal)
        return self.iterate_trees(tree_start, tree_goal)

    def _build_checkpoint_frame(
        self,
        iteration: int,
        collision: bool,
        collision_index: Optional[int],
        path: Optional[List[Node]],
        start_tree: SearchTree,
        goal_tree: SearchTree,
    ) -> Dict:
        # Determine the bridge segment index within ``path`` (if any). The
        # bridge connects a node from the start tree to a node from the goal
        # tree, so it is the first segment where the node's ``tree`` attribute
        # changes.
        bridge_index: Optional[int] = None
        if path:
            for i in range(len(path) - 1):
                if path[i].tree != path[i + 1].tree:
                    bridge_index = i
                    break

        # Derive a coarse status for the bridge segment based on the global
        # collision outcome and which segment index collided first:
        #
        #   - If no collision at all, the bridge was checked and is valid.
        #   - If the collision happened exactly on the bridge, mark it as
        #     a colliding segment.
        #   - If the collision happened on some other segment first, the
        #     bridge might not have been checked yet and remains unknown.
        bridge_status: Optional[str] = None
        if bridge_index is not None:
            if not collision:
                bridge_status = "valid"
            elif collision_index == bridge_index:
                bridge_status = "collision"
            else:
                bridge_status = "unknown"

        return {
            "iteration": iteration,
            "collision": collision,
            "collision_index": collision_index,
            "bridge_index": bridge_index,
            "bridge_status": bridge_status,
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


    def order_path(self,path):
        if path:
            if self._active.name == "start":
                return path
            else:
                # reverse path node order
                return path[::-1]
        else:
            return []
    
    @IPPerfMonitor
    def planPath(self,
        start: List[float],
        goal: List[float],
        config: Optional[Dict] = None) -> List[Optional[Node]]:
        """Grow two search trees using SBL and optionally visualize / record."""
        if config:
            self.config = self._merge_config(config)
        # Initialize the clean stats object
        self.stats = PlannerStats()
        start_time = time.perf_counter()
        
        tree_start, tree_goal = self.init_trees(start, goal)
        checkpoint_frames: List[Dict] = []
        
        repair_focus = None

        for n_iter in tqdm(range(self.config["iterations"])):
            #print(f"Iteration {n_iter}")
            
            # Pass repair_focus to iteration
            start_tree, goal_tree, path_a, path_b = self.iterate_trees(tree_start, tree_goal, repair_focus)
            
            collision = False
            collision_index = None
            
            if path_a and path_b:
                self.stats.candidate_paths_checked += 1
                if self.stats.time_to_first_candidate is None:
                    self.stats.time_to_first_candidate = time.perf_counter() - start_time
                    
                # Unpack 5 values to support both the graph tools and the repair focus
                collision, collision_index, start_tree, goal_tree, new_focus = self.collision_check_solution(start_tree, goal_tree, path_a,path_b)
                path = path_a+path_b
                
                # Update focus point for the next iteration if the path broke
                repair_focus = new_focus if collision else None
            else:
                path = None
                print("no path found")
                

            if self.config["checkpoint_path"]:
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
                    self.config["checkpoint_path"],
                    {
                        "metadata": {
                            "start": start,
                            "goal": goal,
                            "config": self.config,
                        },
                        "frames": checkpoint_frames,
                    },
                )
                
            # CHANGED: Instead of returning, we log the success metrics and 'break' out of the loop
            if path and not collision:
                self.stats.time_to_first_valid_path = time.perf_counter() - start_time
                self.stats.success = True
                
                # Calculate final path length
                path_coords = [n.coordinates for n in path]
                self.stats.path_length = sum(np.linalg.norm(path_coords[i+1] - path_coords[i]) for i in range(len(path_coords)-1))
                break 
            if not path:
                # do not do another iteration if current iteration did not find a path
                break
            
        # --- FINAL STATS CALCULATION ---
        # This runs after the loop finishes, regardless of whether it succeeded or failed.
        
        self.stats.planning_time = time.perf_counter() - start_time
        self.stats.total_nodes_start_tree = len(start_tree.node_ids)
        self.stats.total_nodes_goal_tree = len(goal_tree.node_ids)
        
        # Tally up all the edge statuses
        for tree in [start_tree, goal_tree]:
            for u, v, data in tree.graph.edges(data=True):
                status = data.get("status", "unknown")
                if status == "unknown": self.stats.edges_unchecked += 1
                elif status == "valid": self.stats.edges_valid += 1
                elif status == "invalid": self.stats.edges_invalid += 1
                
        # Finally return the result
        self.startTree = start_tree
        self.goalTree = goal_tree
        if self.stats.success:
            return self.order_path(path)
        return None

    def init_trees(
        self,
        start: List[float],
        goal: List[float],
        ):
        checked_start, checked_goal = self._checkStartGoal([start], [goal])
        tree_start = SearchTree(checked_start[0], "start")
        tree_goal = SearchTree(checked_goal[0], "goal")
        
        # Assign directly to the attributes the visualizer expects
        self.startTree = tree_start
        self.goalTree = tree_goal
        
        self._tree_start = tree_start
        self._tree_goal = tree_goal
        return tree_start, tree_goal

    def iterate_trees(self,
        tree_start: SearchTree, 
        tree_goal: SearchTree, 
        repair_focus: Optional[List[float]] = None
        ) -> Tuple[SearchTree, SearchTree, Optional[List[Node]], Optional[List[Node]]]:
        """
        Perform 1 iteration of expanding tree and checking connectivity
        """
        for iteration in range(int(self.config["max_nodes"])):
            # 1. Pick a tree to expand at random with probability P=0.5
            if random.random() < 0.5:
                self._active, self._passive = tree_start, tree_goal
            else:
                self._active, self._passive = tree_goal, tree_start

            # 2. Expand the active tree by creating a single node
            new_node = self._expand_tree(self._active, self._passive, repair_focus)

            # If no new node was added (e.g., sample in collision), skip this iteration
            if new_node is None:
                continue

            # 3. Attempt to connect the trees based on proximity
            connection_start, connection_goal = self._try_connect(self._active, self._passive, new_node)
            if connection_start and connection_goal:
                # Returns the candidate path as a list of Node objects
                return tree_start, tree_goal, connection_start, connection_goal

        return tree_start, tree_goal, None, None

    def collision_check_solution(
        self,
        tree_start: SearchTree, tree_goal: SearchTree,
        connection_a: List[Node],
        connection_b: List[Node],
    ) -> Tuple[bool, Optional[int], SearchTree, SearchTree, Optional[List[float]]]:
        """
        Adaptive collision check for a candidate path.
        Returns (collision_bool, collision_index, tree_start, tree_goal, repair_focus).
        """
        unchecked_nodes: List[Tuple[int, Node, Node]] = []

        connection_b_rev = connection_b[::-1]
        counter_a = 0
        counter_b = 0
        a_done = len(connection_a) < 2
        b_done = len(connection_b) < 2
        i = 0
        
        # Alternating edge selection logic
        while (not a_done) or (not b_done):
            if (not a_done) and (not b_done):
                # alternating
                take_path_b = bool(i%2)
                i+=1
            elif a_done and (not b_done):
                # only B
                take_path_b = True
            elif (not a_done) and b_done:
                # only A
                take_path_b = False
            else:
                # A and B done
                raise IndexError("loop too long")

            if take_path_b:
                # only sample from B
                node1 = connection_b_rev[counter_b]
                # Segment index within the full path ``connection_a + connection_b``.
                #
                # Path segments are indexed as follows:
                #   - edges inside ``connection_a``: 0 .. len(connection_a) - 2
                #   - bridge between trees:        len(connection_a) - 1
                #   - edges inside ``connection_b``: len(connection_a) .. len(path) - 2
                #
                # We iterate edges of ``connection_b`` using ``connection_b_rev`` from
                # the end of the path back towards the bridge, so the first B edge
                # we check corresponds to the last segment index in the path.
                node_idx = len(connection_a) + len(connection_b) - 2 - counter_b
                counter_b += 1
                node2 = connection_b_rev[counter_b]
                if counter_b + 1 == len(connection_b):
                    b_done = True
            else:
                # only sample from A
                node1 = connection_a[counter_a]
                # Edges inside ``connection_a`` are between consecutive nodes
                # ``connection_a[k]`` and ``connection_a[k+1]`` and correspond
                # to path segment index ``k`` in ``connection_a + connection_b``.
                node_idx = counter_a
                counter_a += 1
                node2 = connection_a[counter_a]
                if counter_a + 1 == len(connection_a):
                    a_done = True

            # Check if edge was already validated (same tree)
            tree = tree_start if node1.tree == "start" else tree_goal
            edge_status = tree.graph[node1.id][node2.id]["status"]
            
            if edge_status == "valid":
                # Jump to next iteration (next edge)
                continue
            elif edge_status == "invalid":
                # Tree-specific repair focus for known invalid edges
                if node1.tree == "start":
                    repair_focus = node1.coordinates.tolist()
                else:
                    repair_focus = node2.coordinates.tolist()
                return True, node_idx, tree_start, tree_goal, repair_focus
            else:
                unchecked_nodes.append((node_idx, node1, node2))
        
        # Check connection between trees for collision
        # Bridge segment between the two trees connects the last node of A to
        # the first node of B. Its segment index in ``connection_a + connection_b``
        # is ``len(connection_a) - 1``.
        unchecked_nodes.append((len(connection_a) - 1, connection_a[-1], connection_b[0]))

        # Check collisions of unknown edges lazily
        checks_performed = 0
        for idx, (segment_index, node1, node2) in enumerate(unchecked_nodes):
            self.stats.line_tests += 1
            checks_performed += 1
            
            # Check collisions of unknown edges
            collision = self._collisionCheckFun(
                node1.coordinates,
                node2.coordinates,
            )


            # count how often edge was checked:
            if self.config["count_edge_checks"]:
                edge_points = sorted([node1.id, node2.id])
                edge_id = str(edge_points[0])+str(edge_points[1])
                if edge_id in self.collision_check_counter:
                    current_count=self.collision_check_counter[edge_id]
                    self.collision_check_counter[edge_id] = current_count+1
                else:
                    self.collision_check_counter[edge_id] = 1

            # TODO: add option to visualize checked points

            repair_focus = None

            # Edge invalidation and repair focus logic
            if node1.tree == node2.tree == "start":                
                if collision:
                    tree_start.invalidate_edge(node1.id, node2.id)
                    repair_focus = node1.coordinates.tolist()  # Parent is node1
                else:
                    tree_start.graph[node1.id][node2.id]["status"] = "valid"
                    
            elif node1.tree == node2.tree == "goal":
                if collision:
                    tree_goal.invalidate_edge(node1.id, node2.id)
                    repair_focus = node2.coordinates.tolist()  # Parent is node2
                else:
                    tree_goal.graph[node1.id][node2.id]["status"] = "valid"
                    
            else:
                # Collision on the bridge connecting the two trees
                if collision:
                    # The bridge isn't natively in either tree's edges, so no tree pruning.
                    # We just focus repair on the start tree's side of the gap.
                    repair_focus = node1.coordinates.tolist()

            # return if collision found
            if collision:
                self.stats.aborted_adaptive_tests += 1
                return True, segment_index, tree_start, tree_goal, repair_focus
            
        # No invalid edges and no collisions found
        return False, None, tree_start, tree_goal, None
