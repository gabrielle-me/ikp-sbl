from dataclasses import dataclass, asdict
from typing import Optional
import pandas as pd


@dataclass
class PlannerStats:
    """Tracks performance metrics for the motion planner."""
    # Success rate
    success: bool = False
    
    # Timing metrics
    planning_time: float = 0.0
    time_to_first_candidate: Optional[float] = None
    time_to_first_valid_path: Optional[float] = None
    
    # Total generated nodes per tree
    total_nodes_start_tree: int = 0
    total_nodes_goal_tree: int = 0
    
    # Edges
    edges_unchecked: int = 0
    edges_valid: int = 0
    edges_invalid: int = 0
    
    # Collision checks
    point_collision_tests: int = 0
    line_tests: int = 0
    aborted_adaptive_tests: int = 0
    
    # Path metrics
    path_length: float = 0.0
    candidate_paths_checked: int = 0

    def to_dict(self) -> dict:
        """Exports stats to a dictionary for pandas/csv."""
        return asdict(self)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Helper to match IPPerfMonitor's expected output format for plotting."""
        data = self.to_dict()
        return pd.DataFrame([data])