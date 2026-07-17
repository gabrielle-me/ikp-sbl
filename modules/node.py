import numpy as np
class Node:
    def __init__(self, id, tree, coordinates:np.ndarray):
        self.id = id
        self.tree = tree
        self.coordinates = coordinates

    def __repr__(self):
        return f"Node {self.coordinates} at {self.tree} tree"