from matplotlib.axes import Axes
from shapely.geometry import Polygon
from shapely import box, plotting
import networkx as nx
from matplotlib.collections import LineCollection
import numpy as np
from typing import Dict



def draw_obstacles(ax:Axes, scene: Dict):
    for obstacle in scene.values():
        x, y = obstacle.exterior.xy
        ax.fill(x, y, facecolor="lightcoral", edgecolor="red", alpha=0.5)

def drawScene(ax: Axes, content:Dict, limits:np.ndarray, figsize=(10,10), starts=None, goals=None, lines=None):
    for key, value in content.items():
        plotting.plot_polygon(value, add_points=False, ax=ax, color="red")
    if lines:
        for start, end, color in lines:
            ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=2)
    if starts:
        ax.scatter([p[0] for p in starts], [p[1] for p in starts], color="green", s=80, label="start")
    if goals:
        ax.scatter([p[0] for p in goals], [p[1] for p in goals], color="orange", s=80, label="goal")

def _as_graph(tree):
    if hasattr(tree, "graph"):
        return tree.graph
    return tree


def plot_tree(ax:Axes, tree, color="blue", node_size=20, alpha=0.6, tree_type: str = None):
    """Plot a search tree with edge styling based on planner status.

    Edge colors:
        - unknown edge: yellow
        - valid edge:   green
        - invalid edge: red
        - edge with collision (``collision`` attribute set to True): purple

    Line style encodes which tree is plotted:
        - ``tree_type == "start"``: dashed
        - ``tree_type == "goal"``: dotted
        - otherwise: solid

    The ``color`` argument controls the node color (kept for backward
    compatibility with earlier plotting code).
    """
    tree = _as_graph(tree)
    positions = nx.get_node_attributes(tree, "pos")
    if not positions:
        return

    # Group edge segments by status / collision flag
    edges_with_data = list(tree.edges(data=True))
    if edges_with_data:
        segments_by_key = {
            "unknown": [],
            "valid": [],
            "invalid": [],
            "collision": [],
            "unreachable": [],
        }

        for u, v, data in edges_with_data:
            if data.get("collision", False):
                key = "collision"
            else:
                key = data.get("status", "unknown")
                if key not in segments_by_key:
                    key = "unknown"
            segments_by_key[key].append([positions[u], positions[v]])

        color_map = {
            "unknown": "yellow",
            "unreachable": "orange",
            "valid": "green",
            "invalid": "red",
            "collision": "purple",
        }
        line_style_map = {
            "unknown": "-",
            "unreachable": ":",
            "valid": "-",
            "invalid": "-",
            "collision": "-",
        }


        for key, segments in segments_by_key.items():
            if not segments:
                continue

            # Use a slightly different alpha for unreachable parts
            # of the tree to make them visually distinguishable.
            edge_alpha = 0.7 if key == "invalid" else alpha

            collection = LineCollection(
                segments,
                colors=color_map[key],
                linewidths=1.2,
                alpha=edge_alpha,
                linestyles=line_style_map[key],
            )
            ax.add_collection(collection)

    xs = [pos[0] for pos in positions.values()]
    ys = [pos[1] for pos in positions.values()]
    ax.scatter(xs, ys, c=color, s=node_size, alpha=alpha, edgecolors="black", linewidths=0.5,label=tree_type)


def plot_path(ax:Axes, path, color="black", annotateOrder = True, collision_index: int = None, collision_color: str = "purple"):
    if not path:
        return
    path = np.array(path)

    if collision_index is None or collision_index < 0 or collision_index >= len(path) - 1:
        ax.plot(path[:, 0], path[:, 1], marker="o", color=color, linewidth=2.5, markersize=5)
    else:
        for idx in range(len(path) - 1):
            segment_color = collision_color if idx == collision_index else color
            ax.plot(
                path[idx:idx + 2, 0],
                path[idx:idx + 2, 1],
                marker="o",
                color=segment_color,
                linewidth=2.5,
                markersize=5,
            )

    if annotateOrder:
        annotatePathOrder(ax,path)

def annotatePathOrder(ax,path):
        for idx, point in enumerate(path):
            ax.annotate(str(idx), (point[0], point[1]), textcoords='offset points', xytext=(5, 5), fontsize=8)


def plot_iteration(
    ax: Axes,
    start_tree,
    goal_tree,
    path=None,
    collision: bool = False,
    collision_index: int = None,
):
    """Draw one planning iteration with bidirectional trees and the current path."""
    plot_tree(ax, start_tree, color="blue", node_size=35, tree_type="start")
    plot_tree(ax, goal_tree, color="cyan", node_size=35, tree_type="goal")
    
    if collision:
        colliding_edge_p1 = path[collision_index]
        colliding_edge_p2 = path[collision_index+1]
        ax.plot([colliding_edge_p1[0],colliding_edge_p2[0]],[colliding_edge_p1[1],colliding_edge_p2[1]],color="purple")


    if path:
        annotatePathOrder(
            ax,
            path,
        )

    ax.set_aspect("equal", adjustable="box")