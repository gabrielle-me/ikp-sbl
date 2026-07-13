import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from shapely.geometry import Polygon
import networkx as nx
from matplotlib.collections import LineCollection
import numpy as np
from typing import Dict



def draw_obstacles(ax:Axes, scene: Dict):
    for obstacle in scene.values():
        x, y = obstacle.exterior.xy
        ax.fill(x, y, facecolor="lightcoral", edgecolor="red", alpha=0.5)


def plot_tree(ax:Axes, tree, color, node_size=20, alpha=0.6):
    positions = nx.get_node_attributes(tree, "pos")
    if not positions:
        return
    edges = list(tree.edges())
    if edges:
        lines = [[positions[u], positions[v]] for u, v in edges]
        collection = LineCollection(lines, colors=color, linewidths=1.2, alpha=alpha)
        ax.add_collection(collection)
    xs = [pos[0] for pos in positions.values()]
    ys = [pos[1] for pos in positions.values()]
    ax.scatter(xs, ys, c=color, s=node_size, alpha=alpha, edgecolors="black", linewidths=0.5)


def plot_path(ax:Axes, path, color="black"):
    if not path:
        return
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], marker="o", color=color, linewidth=2.5, markersize=5)