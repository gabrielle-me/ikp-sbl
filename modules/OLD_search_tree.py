"""

DEPECATED

Bidirectional search tree utilities for sampling-based motion planning.
"""

import random
from typing import Dict, List, Optional, Tuple, Set

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
    
    def _is_edge_valid(self, tree: SearchTree, config1: List[float], config2: List[float]) -> bool:
        """Check if an edge between two configurations is valid (not marked invalid)."""
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
            
            return None

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
        tree_start, tree_goal = self.init_trees(start,goal,config)
        start, goal, path = self.iterate_trees(tree_start, tree_goal, config)
        return start.graph, goal.graph, path

    def init_trees(
        self,
        start: List[float],
        goal: List[float],
        config: Optional[Dict[str, float]] = None,
        ):
        config = self._merge_config(config)
        checked_start, checked_goal = self._checkStartGoal([start], [goal])
        tree_start = SearchTree(checked_start[0])
        tree_goal = SearchTree(checked_goal[0])
        return tree_start, tree_goal

    def iterate_trees(self, tree_start: SearchTree, tree_goal: SearchTree, config: dict) -> Tuple[SearchTree, SearchTree, Optional[List[List[float]]]]:

        for iteration in range(int(config["max_nodes"])):
            # 1. Pick a tree to expand at random with probability P=0.5
            if random.random() < 0.5:
                active, passive = tree_start, tree_goal
            else:
                active, passive = tree_goal, tree_start

            # 2. Expand the active tree by creating a single node
            new_node = self._expand_tree(active, config)

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
        connection: List[List[float]],
        kappa_max: int = 0,
        epsilon: float = 0.05,
    ) -> Optional[int]:
        """Adaptive collision check for a candidate path.

        Returns the index of the first configuration of a colliding edge,
        or ``None`` if no collision is found.
        """
        # Conceptual variant for kappa = 0 (node-only checking), kept for reference:
        # for node in connection:
        #     collision = self._collisionChecker.pointInCollision(node)
        #     if collision:
        #         return 0

        for kappa in range(1, kappa_max + 1):
            # get points between nodes
            for node_id in range(0, len(connection) - 1):
                node1 = np.array(connection[node_id])
                node2 = np.array(connection[node_id + 1])
                node_vec = node2 - node1
                distance = np.linalg.norm(node_vec)
                n_points = 2 ** kappa

                # stop subdividing once segments are fine enough
                if distance / n_points < epsilon:
                    continue

                for i in range(1, n_points, 2):
                    edge_ratio = i / n_points
                    middle_point = node1 + edge_ratio * node_vec
                    # collision check on middle point
                    collision = self._collisionChecker.pointInCollision(middle_point)
                    if collision:
                        print(
                            f"collision found for kappa={kappa}, "
                            f"ratio={edge_ratio}, point={middle_point}"
                        )
                        # Return the index of the first configuration of the colliding edge
                        return node_id

            print(f"no collision for kappa={kappa}")

        # no collision found for all kappas
        return None
    
    def _mark_edge_invalid(self, tree: SearchTree, config1: List[float], config2: List[float]) -> bool:
        """Mark an edge as invalid in the tree if it exists.
        
        Returns True if edge was found and marked, False otherwise.
        """
        config1_tuple = tuple(config1)
        config2_tuple = tuple(config2)
        
        # Search for the edge in the tree
        for u, v in tree.graph.edges():
            u_pos = tuple(tree.position(u))
            v_pos = tuple(tree.position(v))
            
            # Check both directions of the edge
            if (u_pos == config1_tuple and v_pos == config2_tuple) or \
               (u_pos == config2_tuple and v_pos == config1_tuple):
                tree.graph[u][v]['status'] = 'invalid'
                return True
        return False

    def _draw_trees_on_axis(
        self,
        ax,
        tree_start: SearchTree,
        tree_goal: SearchTree,
        iteration: int = 0,
        connection: Optional[List[List[float]]] = None,
    ) -> None:
        """Internal method to draw trees on a given axis."""
        from matplotlib.collections import LineCollection
        
        # Draw edges with different colors for valid/invalid
        for tree, color_valid in [(tree_start, "blue"), (tree_goal, "green")]:
            positions = {node_id: tree.position(node_id) for node_id in tree.node_ids}
            
            # Valid edges
            valid_edges = []
            invalid_edges = []
            for u, v in tree.graph.edges():
                status = tree.graph[u][v].get('status', 'unknown')
                if status == 'invalid':
                    invalid_edges.append([positions[u], positions[v]])
                else:
                    valid_edges.append([positions[u], positions[v]])
            
            # Draw valid edges
            if valid_edges:
                collection = LineCollection(valid_edges, colors=color_valid, linewidths=1.2, alpha=0.6)
                ax.add_collection(collection)
            
            # Draw invalid edges in red
            if invalid_edges:
                collection = LineCollection(invalid_edges, colors="red", linewidths=2.0, alpha=0.8, linestyle="--")
                ax.add_collection(collection)
            
            # Draw nodes
            xs = [pos[0] for pos in positions.values()]
            ys = [pos[1] for pos in positions.values()]
            ax.scatter(xs, ys, c=color_valid, s=30, alpha=0.7, edgecolors="black", linewidths=0.5)
        
        # Draw candidate path if provided
        if connection:
            path_array = np.array(connection)
            ax.plot(path_array[:, 0], path_array[:, 1], marker="o", color="purple", 
                   linewidth=2.0, markersize=4, label="Candidate path", alpha=0.8)
        
        # Annotate start and goal
        ax.scatter([tree_start.position(tree_start.root)[0]], 
                  [tree_start.position(tree_start.root)[1]], 
                  c="cyan", s=120, edgecolors="black", zorder=5, label="Start", marker="s")
        ax.scatter([tree_goal.position(tree_goal.root)[0]], 
                  [tree_goal.position(tree_goal.root)[1]], 
                  c="magenta", s=120, edgecolors="black", zorder=5, label="Goal", marker="s")
        
        ax.set_title(f"SBL Trees - Iteration {iteration}\n(Blue=T_start, Green=T_goal, Red=Invalid edges)")
        ax.set_xlim(-1, 22)
        ax.set_ylim(-1, 22)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

    def visualize_trees(
        self,
        tree_start: SearchTree,
        tree_goal: SearchTree,
        iteration: int = 0,
        connection: Optional[List[List[float]]] = None,
    ) -> None:
        """Visualize the two trees with invalid edges highlighted.
        
        Args:
            tree_start: Start tree
            tree_goal: Goal tree
            iteration: Current iteration number for title
            connection: Optional candidate path to display
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 10))
        self._draw_trees_on_axis(ax, tree_start, tree_goal, iteration, connection)
        plt.show()

    def visualize_all_proposed_paths(
        self,
        tree_start: SearchTree,
        tree_goal: SearchTree,
        iteration_paths: List[Tuple[int, List[List[float]], bool]],
    ) -> None:
        """Visualize all proposed paths from each iteration on a single graph.
        
        Args:
            tree_start: Start tree
            tree_goal: Goal tree
            iteration_paths: List of (iteration_number, path, has_invalid_edge) tuples
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        import matplotlib.patches as mpatches
        
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Draw final trees (blue and green for reference)
        for tree, color_valid in [(tree_start, "blue"), (tree_goal, "green")]:
            positions = {node_id: tree.position(node_id) for node_id in tree.node_ids}
            
            # Valid edges
            valid_edges = []
            invalid_edges = []
            for u, v in tree.graph.edges():
                status = tree.graph[u][v].get('status', 'unknown')
                if status == 'invalid':
                    invalid_edges.append([positions[u], positions[v]])
                else:
                    valid_edges.append([positions[u], positions[v]])
            
            # Draw valid edges (light)
            if valid_edges:
                collection = LineCollection(valid_edges, colors=color_valid, linewidths=0.8, alpha=0.3)
                ax.add_collection(collection)
            
            # Draw invalid edges in red
            if invalid_edges:
                collection = LineCollection(invalid_edges, colors="red", linewidths=1.5, alpha=0.5, linestyle="--")
                ax.add_collection(collection)
            
            # Draw nodes (light)
            xs = [pos[0] for pos in positions.values()]
            ys = [pos[1] for pos in positions.values()]
            ax.scatter(xs, ys, c=color_valid, s=15, alpha=0.4, edgecolors="black", linewidths=0.3)
        
        # Color map for iterations
        colors = plt.cm.tab20(np.linspace(0, 1, len(iteration_paths)))
        
        # Draw all proposed paths
        legend_handles = []
        for idx, (iteration, path, has_invalid) in enumerate(iteration_paths):
            path_array = np.array(path)
            linestyle = "--" if has_invalid else "-"
            linewidth = 2.5 if has_invalid else 2.0
            alpha = 0.6 if has_invalid else 0.7
            
            ax.plot(path_array[:, 0], path_array[:, 1], 
                   color=colors[idx], linewidth=linewidth, linestyle=linestyle, 
                   alpha=alpha, marker="o", markersize=3, label=f"Iter {iteration}")
            
            # Create legend entry
            label_text = f"Iter {iteration}"
            if has_invalid:
                label_text += " (invalid)"
            legend_handles.append(mpatches.Patch(color=colors[idx], label=label_text))
        
        # Annotate start and goal
        ax.scatter([tree_start.position(tree_start.root)[0]], 
                  [tree_start.position(tree_start.root)[1]], 
                  c="cyan", s=150, edgecolors="black", zorder=5, marker="s", label="Start")
        ax.scatter([tree_goal.position(tree_goal.root)[0]], 
                  [tree_goal.position(tree_goal.root)[1]], 
                  c="magenta", s=150, edgecolors="black", zorder=5, marker="s", label="Goal")
        
        ax.set_title("All Proposed Paths from SBL Planning\n(Background: Final trees | Dashed: Invalid paths)")
        ax.set_xlim(-1, 22)
        ax.set_ylim(-1, 22)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
        ax.legend(handles=legend_handles + [
            mpatches.Patch(color="cyan", label="Start"),
            mpatches.Patch(color="magenta", label="Goal"),
            mpatches.Patch(color="lightblue", alpha=0.3, label="T_start (final)"),
            mpatches.Patch(color="lightgreen", alpha=0.3, label="T_goal (final)")
        ], loc="upper left", fontsize=9)
        plt.show()
    def _path_contains_invalid_segment(
        self,
        tree_start: SearchTree,
        tree_goal: SearchTree,
        connection: List[List[float]],
    ) -> bool:
        """Check whether a candidate path uses any previously invalidated edge segment.
        
        Checks edge status in the actual tree graphs.
        """
        for idx in range(len(connection) - 1):
            config1 = tuple(connection[idx])
            config2 = tuple(connection[idx + 1])
            
            # Check if this segment is marked as invalid in either tree
            for u, v in tree_start.graph.edges():
                u_pos = tuple(tree_start.position(u))
                v_pos = tuple(tree_start.position(v))
                
                if (u_pos == config1 and v_pos == config2) or \
                   (u_pos == config2 and v_pos == config1):
                    if tree_start.graph[u][v].get('status') == 'invalid':
                        return True
            
            for u, v in tree_goal.graph.edges():
                u_pos = tuple(tree_goal.position(u))
                v_pos = tuple(tree_goal.position(v))
                
                if (u_pos == config1 and v_pos == config2) or \
                   (u_pos == config2 and v_pos == config1):
                    if tree_goal.graph[u][v].get('status') == 'invalid':
                        return True
        
        return False
    
    def plan_path(
        self,
        start: List[float],
        goal: List[float],
        config: Optional[Dict[str, float]] = None,
        max_iter: int = 10,
        animate: bool = True,
    ) -> Tuple[Optional[nx.Graph], Optional[nx.Graph], Optional[List[List[float]]]]:

        # Work with a fully merged configuration dictionary.
        merged_config = self._merge_config(config)

        tree_start, tree_goal = self.init_trees(start, goal, merged_config)

        # Collect all proposed paths for visualization
        iteration_paths = []  # List of (iteration, path, has_invalid_edge)

        for iteration in range(max_iter):
            print(f"iteration={iteration}")
            tree_start, tree_goal, connection = self.iterate_trees(tree_start, tree_goal, merged_config)
            
            if not connection:
                # No candidate path in this iteration, keep growing the trees.
                continue

            # Track whether this path has invalid segments
            has_invalid = self._path_contains_invalid_segment(tree_start, tree_goal, connection)
            iteration_paths.append((iteration, connection, has_invalid))
            
            if has_invalid:
                print(f"  -> Candidate path contains invalid segments, skipping")
                continue

            collision_start_node_id = self.collision_check_solution(connection, kappa_max=10)

            if collision_start_node_id is None:
                # No collision along the candidate path: solution found.
                print(f"  -> Solution found!")
                if animate:
                    self.visualize_all_proposed_paths(tree_start, tree_goal, iteration_paths)
                return tree_start.graph, tree_goal.graph, connection

            # Collision found: mark the corresponding edge segment as invalid in the tree graphs
            # so that it is ignored in future candidate paths.
            if 0 <= collision_start_node_id < len(connection) - 1:
                a = connection[collision_start_node_id]
                b = connection[collision_start_node_id + 1]
                self._mark_edge_invalid(tree_start, a, b)
                self._mark_edge_invalid(tree_goal, a, b)
                print(f"  -> Collision detected at edge index {collision_start_node_id}, marked as invalid")

        # No valid path found within the iteration limit.
        print(f"No valid path found within {max_iter} iterations")
        if animate and iteration_paths:
            self.visualize_all_proposed_paths(tree_start, tree_goal, iteration_paths)
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