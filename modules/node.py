import numpy as np


class Node:
    """Lightweight node representation used for path / connection objects.

    Attributes
    ----------
    id:
        Unique identifier of the node within its search tree (typically a UID from
        :class:`modules.SearchTree.SearchTree`).
    tree:
        Name of the tree this node belongs to (e.g. "start" or "goal").
    coordinates:
        Configuration of the node as a NumPy array.
    """

    def __init__(self, id: int, tree: str, coordinates: np.ndarray):
        self.id = id
        self.tree = tree
        self.coordinates = coordinates

    def __repr__(self) -> str:
        return f"Node(id={self.id}, tree={self.tree}, coordinates={self.coordinates})"

    def __sub__(self, other: "Node") -> np.ndarray:
        return other.coordinates - self.coordinates

    def __add__(self, other: "Node") -> np.ndarray:
        return other.coordinates + self.coordinates