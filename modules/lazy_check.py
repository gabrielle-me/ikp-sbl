import random
from typing import Dict, List, Optional, Tuple

import networkx as nx

try:
    from lecture_examples.IPPRMBase import PRMBase
except ImportError:
    from IPPRMBase import PRMBase

try:
    from modules.search_tree import BidirectionalSBL, SearchTree
except ImportError:
    from search_tree import BidirectionalSBL, SearchTree


class LazySBL(BidirectionalSBL):
    """Lazy SBL planner that delays node and edge collision checks until a candidate path is found."""

    def __init__(self, coll_checker):
        super(LazySBL, self).__init__(coll_checker)

    def _validate_candidate_path(self, path: List[List[float]]) -> bool:
        """Validate all nodes and edges in a candidate path before accepting it."""
        if len(path) < 2:
            return False

        for pos in path:
            if self._collisionChecker.pointInCollision(pos):
                return False

        for start_pos, end_pos in zip(path[:-1], path[1:]):
            if self._collisionChecker.lineInCollision(start_pos, end_pos):
                return False

        return True

    def grow_trees(
        self,
        start: List[float],
        goal: List[float],
        config: Optional[Dict[str, float]] = None,
    ) -> Tuple[nx.Graph, nx.Graph, Optional[List[List[float]]]]:
        """Grow two trees and only validate the candidate path once a connection is found."""
        config = self._merge_config(config)
        checked_start, checked_goal = self._checkStartGoal([start], [goal])
        tree_start = SearchTree(checked_start[0])
        tree_goal = SearchTree(checked_goal[0])

        for _ in range(int(config["max_nodes"])):
            if random.random() < 0.5:
                active, passive = tree_start, tree_goal
            else:
                active, passive = tree_goal, tree_start

            new_node_id = self._expand_tree(active, config)
            connection = self._try_connect(active, passive, new_node_id, config)
            if connection is not None and self._validate_candidate_path(connection):
                return tree_start.graph, tree_goal.graph, connection

        return tree_start.graph, tree_goal.graph, None

    def build_trees(
        self,
        start: List[float],
        goal: List[float],
        config: Optional[Dict[str, float]] = None,
    ) -> Tuple[nx.Graph, nx.Graph]:
        tree_start, tree_goal, _ = self.grow_trees(start, goal, config)
        return tree_start, tree_goal
