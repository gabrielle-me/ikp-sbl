"""Checkpoint loading and interactive replay helpers for SBL planning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.animation as mpl_animation
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display

from modules.SearchTree import SearchTree
from modules import IPVISsbl


def load_sbl_checkpoints(checkpoint_path: str) -> Dict[str, Any]:
    path = Path(checkpoint_path)
    with path.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    frames: List[Dict[str, Any]] = []
    for frame in payload.get("frames", []):
        frames.append(
            {
                "iteration": frame["iteration"],
                "collision": frame["collision"],
                "collision_index": frame.get("collision_index"),
                "path": frame.get("path"),
                "start_tree": SearchTree.from_checkpoint(frame["trees"]["start"]),
                "goal_tree": SearchTree.from_checkpoint(frame["trees"]["goal"]),
            }
        )

    return {
        "metadata": payload.get("metadata", {}),
        "frames": frames,
    }


def _resolve_checkpoint_limits(
    frames: List[Dict[str, Any]],
    scene_limits: Optional[np.ndarray] = None,
) -> tuple[List[float], List[float]]:
    if scene_limits is not None:
        return scene_limits[0], scene_limits[1]

    all_x = []
    all_y = []
    for frame in frames:
        for tree in (frame["start_tree"], frame["goal_tree"]):
            positions = [tree.position(node_id) for node_id in tree.node_ids]
            all_x.extend(position[0] for position in positions)
            all_y.extend(position[1] for position in positions)
        if frame["path"]:
            all_x.extend(point[0] for point in frame["path"])
            all_y.extend(point[1] for point in frame["path"])

    if all_x and all_y:
        margin = 1.0
        return [min(all_x) - margin, max(all_x) + margin], [min(all_y) - margin, max(all_y) + margin]

    return [0.0, 1.0], [0.0, 1.0]


def _draw_checkpoint_frame(
    ax,
    frame: Dict[str, Any],
    scene: Optional[Dict[str, Any]],
    xlim: List[float],
    ylim: List[float],
):
    ax.cla()
    if scene is not None:
        IPVISsbl.draw_obstacles(ax, scene)
    IPVISsbl.plot_iteration(
        ax,
        frame["start_tree"],
        frame["goal_tree"],
        frame["path"],
        frame["collision"],
        frame["collision_index"],
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(f"SBL checkpoint frame {frame['iteration']}")
    ax.tick_params(axis="both", which="both", length=0)
    ax.legend(loc="upper left")


def export_gif(
    checkpoint_path: str,
    gif_path: str,
    scene: Optional[Dict[str, Any]] = None,
    scene_limits: Optional[np.ndarray] = None,
    fps: int = 1,
) -> Path:
    """Export a recorded SBL checkpoint sequence as an animated GIF."""
    checkpoint_data = load_sbl_checkpoints(checkpoint_path)
    frames = checkpoint_data["frames"]
    if not frames:
        raise ValueError(f"No frames found in checkpoint file: {checkpoint_path}")

    resolved_xlim, resolved_ylim = _resolve_checkpoint_limits(frames, scene_limits)
    fig, ax = plt.subplots(figsize=(10, 10))

    def update(frame_index: int):
        _draw_checkpoint_frame(ax, frames[frame_index], scene, resolved_xlim, resolved_ylim)
        return ax

    anim = mpl_animation.FuncAnimation(fig, update, frames=len(frames), blit=False, repeat=False)
    output_path = Path(gif_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = mpl_animation.PillowWriter(fps=fps)
    anim.save(str(output_path), writer=writer)
    plt.close(fig)
    return output_path


def animate(
    checkpoint_path: str,
    scene: Optional[Dict[str, Any]] = None,
    scene_limits: Optional[np.ndarray] = None,
):
    """Create an interactive notebook viewer for recorded SBL checkpoints."""
    checkpoint_data = load_sbl_checkpoints(checkpoint_path)
    frames = checkpoint_data["frames"]
    if not frames:
        raise ValueError(f"No frames found in checkpoint file: {checkpoint_path}")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal", adjustable="box")
    slider = widgets.BoundedIntText(
        value=0,
        min=0,
        max=len(frames) - 1,
        step=1,
        description="Frame",
        continuous_update=False,
        #readout=True,
    )
    title = widgets.HTML()
    output = widgets.Output()
    resolved_xlim, resolved_ylim = _resolve_checkpoint_limits(frames, scene_limits)

    def render(frame_index: int) -> None:
        frame = frames[frame_index]
        _draw_checkpoint_frame(ax, frame, scene, resolved_xlim, resolved_ylim)
        with output:
            output.clear_output(wait=True)
            display(fig)
        title.value = (
            f"<b>Iteration:</b> {frame['iteration']} &nbsp; "
            f"<b>Collision:</b> {frame['collision']} &nbsp; "
            f"<b>Collision index:</b> {frame['collision_index']}"
        )

    def on_change(change):
        if change["name"] == "value":
            render(change["new"])

    slider.observe(on_change)
    render(0)
    widget_box = widgets.VBox([slider, title, output])
    plt.close(fig)
    return widget_box
