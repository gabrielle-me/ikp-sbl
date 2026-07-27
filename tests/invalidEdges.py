import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path.cwd().parent
sys.path.insert(0, str(REPO_ROOT))

REPO_ROOT

# Import custom modules
from planners.SBL import BidirectionalSBL
from lecture_examples.IPEnvironment import CollisionChecker
from modules import IPVISsbl

# Set global variables for better overview and easier adjustments
START = [1.0, 1.0] # Start position
GOAL = [3.0, 1.0] # Goal position
SCENE_LIMITS = np.array([[0,22],[0,22]])
CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "sbl_checkpoints.json"
GIF_PATH = REPO_ROOT / "gifs"

def validate():
    fig,ax = plt.subplots(figsize=(5,5))
    cc = CollisionChecker({})
    planner = BidirectionalSBL(cc,{"checkpoint_path":str(CHECKPOINT_PATH),
                                "iterations":1,
                                "goal_bias":0.0})
    start_tree, goal_tree = planner.init_trees(START,GOAL)
    custom_node_id=start_tree.add_node([2.0,1.0],start_tree.root)
    start_tree.invalidate_edge(start_tree.root,custom_node_id)
    for i in range(10):
        start_tree, goal_tree, start_path, goal_path = planner.iterate_trees(start_tree,goal_tree)
        if start_path and goal_path:
            break
    IPVISsbl.plot_iteration(ax,start_tree,goal_tree,start_path+goal_path[::-1],bridge_index=len(start_path)-1)
