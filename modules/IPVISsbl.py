from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely import box, plotting
import networkx as nx
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
from typing import Dict, List, Any, Optional
from modules.SearchTree import SearchTree
from planners import SBL
from modules.node import Node


def sblVisualize(planner:SBL.BidirectionalSBL,solution:List[Node],ax:Axes, nodeSize: Optional[int] = 100):
    """ Draw graph, obstacles and solution in a axis environment of matplotib.
    """
    # get a list of positions of all nodes by returning the content of the attribute 'pos'
    collChecker = planner._collisionChecker

    collChecker.drawObstacles(ax)

    # Prefer an explicitly provided solution path; fall back to the
    # planner's internal path for backward compatibility.
    plot_iteration(ax, planner.startTree, planner.goalTree, path=solution)
    
    """
    pos = nx.get_node_attributes(graph,'pos')
    
    # draw graph (nodes colorized by degree)
    nx.draw_networkx_nodes(graph, pos,  cmap=plt.cm.Blues, ax = ax, node_size=nodeSize)
    nx.draw_networkx_edges(graph,pos,
                                ax = ax
                                 )
    Gcc = sorted(nx.connected_components(graph), key=len, reverse=True)
    G0=graph.subgraph(Gcc[0])# = largest connected component

    # how largest connected component
    nx.draw_networkx_edges(G0,pos,
                               edge_color='b',
                               width=3.0, ax = ax
                            )

    
    # draw nodes based on solution path
    Gsp = nx.subgraph(graph,planner.path)
    nx.draw_networkx_nodes(Gsp,pos,
                            node_size=nodeSize*1.5,
                             node_color='g',  ax = ax)
        
    # draw edges based on solution path
    nx.draw_networkx_edges(Gsp,pos,alpha=0.8,edge_color='g',width=10,  ax = ax)
        
    # draw start and goal
    if "start" in graph.nodes(): 
        nx.draw_networkx_nodes(graph,pos,nodelist=["start"],
                                   node_size=nodeSize*1.5,
                                   node_color='#00dd00',  ax = ax)
        nx.draw_networkx_labels(graph,pos,labels={"start": "S"},  ax = ax)


    if "goal" in graph.nodes():
        nx.draw_networkx_nodes(graph,pos,nodelist=["goal"],
                                   node_size=nodeSize*1.5,
                                   node_color='#DD0000',  ax = ax)
        nx.draw_networkx_labels(graph,pos,labels={"goal": "G"},  ax = ax)
    """



def draw_obstacles(ax:Axes, scene: Dict):
    for obstacle in scene.values():
        x, y = obstacle.exterior.xy
        ax.fill(x, y, facecolor="lightcoral", edgecolor="red", alpha=0.5)

def drawScene(ax: Axes, content:Dict, starts=None, goals=None, lines=None):
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


def _path_to_coordinates(path: Any) -> List[np.ndarray]:
    """Normalize a path representation to a list of coordinate arrays.

    The SBL planner now represents candidate paths as ``List[Node]`` objects,
    while checkpoint files and some legacy code use raw coordinate lists.

    This helper accepts both forms and always returns a list of NumPy arrays
    ``[x, y, ...]`` so that plotting routines can work uniformly.
    """

    if path is None:
        return []

    if not path:
        return []

    first = path[0]

    # New representation: list of Node objects with a ``coordinates`` attribute
    if hasattr(first, "coordinates"):
        return [np.asarray(node.coordinates) for node in path]

    # Legacy representation: list of coordinate lists/arrays
    return [np.asarray(p) for p in path]


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


def plot_path(
    ax: Axes,
    path,
    color: str = "green",
    annotateOrder: bool = True,
    collision_index: int = None,
    collision_color: str = "purple",
    line_width: float = 3.0,
):
    coords = _path_to_coordinates(path)
    if not coords:
        return
    path_arr = np.vstack(coords)

    if collision_index is None or collision_index < 0 or collision_index >= len(path_arr) - 1:
        # Full solution path in a single color, thicker than the tree edges
        ax.plot(
            path_arr[:, 0],
            path_arr[:, 1],
            #marker="o",
            color=color,
            linewidth=line_width,
            markersize=5,
        )
    else:
        for idx in range(len(path_arr) - 1):
            segment_color = collision_color if idx == collision_index else color
            ax.plot(
                path_arr[idx:idx + 2, 0],
                path_arr[idx:idx + 2, 1],
                #marker="o",
                color=segment_color,
                linewidth=line_width,
                markersize=5,
            )

    if annotateOrder:
        annotatePathOrder(ax, path_arr)

def annotatePathOrder(ax,path):
        for idx, point in enumerate(path):
            ax.annotate(str(idx), (point[0], point[1]), textcoords='offset points', xytext=(5, 5), fontsize=8)


def plot_iteration(
    ax: Axes,
    start_tree: SearchTree,
    goal_tree: SearchTree,
    path: Optional[List[Node]] = None,
    collision: bool = False,
    collision_index: int = None,
):
    """Draw one planning iteration with bidirectional trees and the current path."""
    plot_tree(ax, start_tree, color="blue", node_size=35, tree_type="start")
    plot_tree(ax, goal_tree, color="cyan", node_size=35, tree_type="goal")
    
    # Normalize path for plotting: may be a ``List[Node]`` or a list of
    # coordinate lists/arrays (from checkpoints). If provided, draw the
    # full solution path thicker than the tree edges and highlight any
    # colliding segment.
    if path is not None:
        plot_path(
            ax,
            path,
            color="green",              # keep solution path in green
            annotateOrder=False,
            collision_index=collision_index if collision else None,
            collision_color="purple",   # colliding edge in purple
            line_width=3.0,              # thicker than tree edges (1.2)
        )
        # Mark start and goal points in special colors based on the path
        coords = _path_to_coordinates(path)
        if coords:
            start = coords[0]
            goal  = coords[-1]
            ax.scatter(
                start[0],
                start[1],
                label="Startpoint",
                c="blue",
                s=80,
            )
            ax.scatter(
                goal[0],
                goal[1],
                label="Goalpoint",
                c="cyan",
                s=80,
            )

    # Build legend: existing scatter markers (start/goal, tree nodes)
    # plus line color coding for edge status.
    existing_handles, existing_labels = ax.get_legend_handles_labels()

    line_handles = [
        Line2D([0], [0], color="yellow", lw=2, label="edge: unknown/unchecked"),
        Line2D([0], [0], color="green", lw=2, label="edge: valid"),
        Line2D([0], [0], color="red", lw=2, label="edge: invalid"),
        Line2D([0], [0], color="purple", lw=2, label="edge: colliding"),
    ]

    handles = existing_handles + line_handles
    labels = existing_labels + [h.get_label() for h in line_handles]
    ax.legend(handles=handles, labels=labels, loc="best")

    ax.set_aspect("equal", adjustable="box")