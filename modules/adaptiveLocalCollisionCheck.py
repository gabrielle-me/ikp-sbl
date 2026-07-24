import numpy as np
from typing import Optional, Tuple
from lecture_examples.IPEnvironment import CollisionChecker
from lecture_examples.IPPerfMonitor import IPPerfMonitor
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Dict

MAX_KAPPA = 25

class LineChecker():
    def __init__(self, coll_checker: CollisionChecker, config:Dict):
        self.config = config
        self.checker = coll_checker

    @IPPerfMonitor
    def __call__(self,node1: np.ndarray, node2: np.ndarray):
        return self.checker.lineInCollision(node1, node2)

class AdaptiveLineChecker(LineChecker):
    def __init__(self, coll_checker: CollisionChecker,config:Dict, ):
        super(AdaptiveLineChecker, self).__init__(coll_checker, config)
        self.kappa_max = int(np.log2(config["steps"] + 1))
        if "return_points" in config:
            self.return_points = config["return_points"]
        else:
            self.return_points = False


    @IPPerfMonitor
    def __call__(self,node1: np.ndarray, node2: np.ndarray):
        """
        Iterative collision check for each segment with increasing kappa until kappa_max
        """ 
        relative_vector = node2 - node1
        distance = np.linalg.norm(relative_vector)

        #Init vector for checked points
        #steps = 2**kappa_max - 1
        checked_points = np.zeros((self.config["steps"], len(node1)))

        checked_points_ind = 0
        for kappa in range(1, self.kappa_max+1):
            n_segments = 2**kappa

            # termination condition: small distance between segments -> no collision
            distance_per_segment = distance / n_segments
            if distance_per_segment < self.config["epsilon"]:
                if self.return_points:
                    return False, checked_points[:checked_points_ind]
                else:
                    return False
            
            for n_segment in range(1, n_segments, 2):
                ratio = n_segment/n_segments
                test_point = node1 + ratio*relative_vector
                checked_points[checked_points_ind] = test_point
                checked_points_ind += 1
                collision = self.checker.pointInCollision(test_point)
                if collision:
                    if self.return_points:
                        return True, checked_points[:checked_points_ind]
                    else:
                        return True
        if self.return_points:    
            return False, checked_points[:checked_points_ind]
        else:
            return False



def visualize_checked_points(ax: Axes, node1: np.ndarray, node2: np.ndarray, checkedPoints: np.ndarray, collision: bool, annotateOrder: Optional[bool] = True):
    """Visualize the segment between node1 and node2, the checked points and their order.
    If collision is True, mark the last checked point with a red cross.
    """
    
    # draw line between nodes
    xs = [node1[0], node2[0]]
    ys = [node1[1], node2[1]]
    ax.plot(xs, ys, '-k', label='segment')

    if len(checkedPoints) > 0:
        # color by order
        colors = np.linspace(0.0, 1.0, len(checkedPoints))
        sc = ax.scatter(checkedPoints[:,0], checkedPoints[:,1], c=colors, cmap='viridis', s=40, label='checked points')
        #ax.colorbar(sc, label='order')
        if annotateOrder:
            for idx, point in enumerate(checkedPoints, start=1):
                ax.annotate(str(idx), (point[0], point[1]), textcoords='offset points', xytext=(5, 5), fontsize=8)
        # mark last point if collision
        if collision:
            last = checkedPoints[-1]
            ax.scatter([last[0]], [last[1]], marker='x', c='red', s=100, linewidths=2, label='collision')

    ax.scatter([node1[0]], [node1[1]], c='green', s=50, marker='x', label='node1')
    ax.scatter([node2[0]], [node2[1]], c='blue', s=50, marker='x', label='node2')
    