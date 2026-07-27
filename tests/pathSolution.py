import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path.cwd().parent
sys.path.insert(0, str(REPO_ROOT))

REPO_ROOT

# Import custom modules
from planners.SBL import BidirectionalSBL
from modules import randomScene, IPVISsbl
from lecture_examples.IPPerfMonitor import IPPerfMonitor

SCENE_LIMITS = np.array([[0,22],[0,22]])
STEPS = 50

def validate(n_scenes:int=200):
    fig,ax = plt.subplots(figsize=(5,5))
    time_naive = 0.
    n_naive = 0
    time_adaptive = 0.
    n_adaptive = 0
    n_missed_collisions = 0
    n_no_path = 0
    SHOW_MISSED_COLLISIONS = False

    IPPerfMonitor.clearData()
    for i in range(n_scenes):
        benchmark = randomScene.create_random_benchmark(SCENE_LIMITS)
        planner_naive = BidirectionalSBL(benchmark.collisionChecker,{"count_edge_checks":True,
                                                                    "collision_check": {
                                                                        "adaptive": False,
                                                                        "steps": STEPS,}})
        planner_adaptive = BidirectionalSBL(benchmark.collisionChecker,{"count_edge_checks":True,
                                                                        "collision_check": {
                                                                            "adaptive":True,
                                                                            "steps": STEPS,
                                                                            "epsilon":0.0001}})
        path_naive = planner_naive.planPath(list(benchmark.startList[0]), list(benchmark.goalList[0]))
        time, n= IPPerfMonitor.get_time("__call__")
        time_naive += time
        n_naive +=n
        IPPerfMonitor.clearData()
        path_adaptive = planner_adaptive.planPath(list(benchmark.startList[0]), list(benchmark.goalList[0]))
        time, n= IPPerfMonitor.get_time("__call__")
        time_adaptive += time
        n_adaptive +=n
        edge_check_counts = planner_adaptive.collision_check_counter

        # 14.2 Check that no edge is checked for collisions twice
        assert np.any(np.array(list(edge_check_counts.values()))<2), f"Edge doublechecked for collision {edge_check_counts.values()}"

        if path_adaptive:
            # 14.4 Check that path goes from start to goal point
            assert np.all(path_adaptive[0].coordinates == benchmark.startList[0]), "Start points not matching"
            assert np.all(path_adaptive[-1].coordinates == benchmark.goalList[0]), "Goal points not matching"


            # 14.5 Check that all edges on path are collision free (complete line test as reference)
            colliding_edges = []
            for i_node in range(len(path_adaptive)-2):
                assert not benchmark.collisionChecker.pointInCollision(path_adaptive[i_node].coordinates), f"Colliding node {path_adaptive[i_node]}"
                if benchmark.collisionChecker.lineInCollisionExact(path_adaptive[i_node].coordinates,path_adaptive[i_node+1].coordinates):
                    colliding_edges.append(path_adaptive[i_node:i_node+2])
            if colliding_edges:
                n_missed_collisions += 1
                if SHOW_MISSED_COLLISIONS:
                    fig = plt.figure(figsize=(10, 10))
                    ax = fig.add_subplot(1, 1, 1)
                    IPVISsbl.sblVisualize(planner_adaptive,path_adaptive,ax)
                    ax.set_title(f"Colliding edges {colliding_edges}")
                    plt.show()
        else:
            n_no_path += 1

    print("\nResults\n-------")
    print("No edge checked for collision twice or more")
    print("All solutions lead from start to goal point")
    print(f"Missed collisions: {n_missed_collisions} ({n_missed_collisions*100/n_scenes}%)")
    print(f"No path found: {n_no_path} ({n_no_path*100/n_scenes}%)")


    time_avg_naive = time_naive / n_naive
    time_avg_adaptive = time_adaptive / n_adaptive
    plt.bar(["naive","adaptive"],[time_avg_naive*1_000,time_avg_adaptive*1_000])
    plt.title(f"Edge collision check: {np.round((1-time_avg_adaptive/time_avg_naive)*100,2)}% time saving in adaptive mode")
    plt.ylabel("Duration per edge / ms")
    plt.show()
    
