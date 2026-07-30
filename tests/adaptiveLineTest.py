import sys
from pathlib import Path
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import numpy as np

# own modules
REPO_ROOT = Path.cwd().parent
sys.path.insert(0, str(REPO_ROOT))

REPO_ROOT
from modules.adaptiveLocalCollisionCheck import *
from lecture_examples.IPEnvironment import CollisionChecker
from modules.SBLvis import draw_obstacles

scene = {"wall": Polygon([(5, 5), (5, 10), (10, 10), (10, 5)])}
    
checker = CollisionChecker(scene)
node1 = np.array([5.3,4])
node2 = np.array([4,10])

max_checked_points = 50

def unit_test():
    fig,ax = plt.subplots(figsize=(5,5))

    epsilon = np.linalg.norm(node1-node2) / 50
    adaptive_line_checker = AdaptiveLineChecker(checker, {"steps":max_checked_points,
                                                        "epsilon":epsilon,
                                                        "return_points": True})
    collision, checkedPoints = adaptive_line_checker(node1,node2)
    checker.lineInCollision(node1,node2, steps=max_checked_points)


    draw_obstacles(ax,scene)
    visualize_checked_points(ax, node1, node2,checkedPoints,collision)

    ax.tick_params(axis="both", which="both", length=0)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")
    plt.show()