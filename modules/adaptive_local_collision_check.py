import numpy as np
from IPEnvironment import CollisionChecker

def adaptive_local_collision_check(node1: np.ndarray, node2: np.ndarray, coll_checker: CollisionChecker, kappa: int, kappa_max: int, epsilon: float) -> bool:
    """
    Recursive collision check for each segment with increasing kappa until kappa_max
    """
    relative_vector = node2 - node1
    distance = np.linalg.norm(relative_vector)
    
    # collision check
    middle_point = node1 + relative_vector / 2
    collision = coll_checker.pointInCollision(middle_point)
    if collision:
        return True

    # finish conditions (no collision)
    if distance/2 < epsilon:
        return False
    
    kappa +=1
    if kappa == kappa_max:
        return False
    
    # recursive function call
    collision1 = adaptive_local_collision_check(node1, middle_point, coll_checker, kappa, kappa_max, epsilon)
    if collision1: return True
    return adaptive_local_collision_check(middle_point, node2, coll_checker, kappa, kappa_max, epsilon)
