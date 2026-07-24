from shapely.geometry import Point, Polygon, LineString, MultiPolygon
from shapely.ops import unary_union
from shapely import box
import numpy as np
from typing import Dict, Optional, Tuple
np.random.seed(42)


from lecture_examples.IPBenchmark import Benchmark
from lecture_examples.IPEnvironment import CollisionChecker


def random_polygon(scene_limits: np.ndarray, n_points: int, point_std_dev: float, buffer_ratio: float = 0.05) -> Polygon:
    points = np.zeros((n_points,2))
    points[0] = np.random.uniform(scene_limits[:,0],scene_limits[:,1])
    for i in range(n_points-1):
        new_point = np.random.normal(points[i],point_std_dev)
        points[i+1] = np.clip(new_point, scene_limits[:,0],scene_limits[:,1])
    #display(points)
    polygon = Polygon(points).buffer(point_std_dev*buffer_ratio)
    if isinstance(polygon, MultiPolygon):
        return random_polygon(scene_limits, n_points, point_std_dev, buffer_ratio)
    return polygon

def random_string(scene_limits: np.ndarray, n_points: int, point_std_dev: float, buffer_ratio: float = 0.05) -> LineString:
    points = np.zeros((n_points,2))
    points[0] = np.random.uniform(scene_limits[:,0],scene_limits[:,1])
    for i in range(n_points-1):
        new_point = np.random.normal(points[i],point_std_dev)
        points[i+1] = np.clip(new_point, scene_limits[:,0],scene_limits[:,1])
    #display(points)
    return LineString(points).buffer(point_std_dev*buffer_ratio)

def random_point(scene_limits: np.ndarray, buffer_ratio: float = 0.05) -> Point:
    point  = np.random.uniform(scene_limits[:,0],scene_limits[:,1])
    #display(point)
    buffer_size = np.random.uniform(scene_limits[0,0],scene_limits[0,1]) * buffer_ratio
    return Point(point).buffer(buffer_size)

def create_random_field(scene_limits: np.ndarray, n_polygons: int, n_strings: int, n_points: int) -> dict:
    field = dict()
    for i in range(n_polygons):
        field[f"polygon{i}"] = random_polygon(scene_limits, np.random.randint(3,7),3.)
    
    for i in range(n_strings):
        field[f"string{i}"] = random_string(scene_limits, np.random.randint(2,7),3.)
    
    for i in range(n_points):
        field[f"point{i}"] = random_point(scene_limits)

    return field

def get_free_space(scene:Dict, scene_limits:np.ndarray):
    # 1. Define your total map boundary (e.g., 100x100 space)
    ws = list(scene_limits.T.flatten())
    workspace = box(*ws)

    # 2. Combine all your random obstacles into one geometry
    # (Assuming 'obstacles_list' contains your Points, Polygons, and LineStrings)
    object_list = list(scene.values())
    all_obstacles = unary_union(object_list)

    # 3.  Get the raw free space
    return workspace.difference(all_obstacles)

def get_valid_start_and_goal(free_space, scene_limits: np.ndarray, max_iter=20, min_dist_ratio: Optional[float]=0.25, max_dist_ratio:Optional[float]=0.8):
    diagonal = np.linalg.norm(scene_limits[:,1] - scene_limits[:,0])
    # Ensure we are working with a MultiPolygon structure for consistency
    if free_space.geom_type == 'Polygon':
        components = [free_space]
    else:
        components = list(free_space.geoms)
        
    # Filter out tiny slivers of free space where a robot can't fit
    valid_components = [comp for comp in components if comp.area > 5.0]
    
    if not valid_components:
        raise ValueError("No viable free space found on the map.")
        
    # Pick the largest connected component of free space to ensure a good map size
    largest_free_zone = max(valid_components, key=lambda c: c.area)
    #display(largest_free_zone)
    
    # Function to randomly sample a point inside a polygon
    def sample_point_in_poly(polygon, max_iter=20):
        minx, miny, maxx, maxy = polygon.bounds
        for i in range(max_iter):
            p = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
            if polygon.contains(p):
                return p
        return None

    # Flood/Sample from the exact same connected component
    start_point = sample_point_in_poly(largest_free_zone)
    
    # Ensure start and goal aren't on top of each other
    for i in range(max_iter):
        goal_point = sample_point_in_poly(largest_free_zone)
        if min_dist_ratio<start_point.distance(goal_point)/diagonal < max_dist_ratio:  # Minimum distance requirement      
            return point2array(start_point), point2array(goal_point), largest_free_zone
    return None, None

def point2array(p: Point) -> np.ndarray:
    return p.coords.__array__()[0]

def create_random_benchmark(scene_limits: np.ndarray,
                            name: Optional[str]="Random Benchmark",
                            level: Optional[int]=2,
                            n_polygons: Optional[int]=6,
                            n_strings: Optional[int]=3,
                            n_points: Optional[int]=6,
                            max_iter=20) -> Benchmark:
    for i in range(max_iter):
        description = f"randomly generated scene\nlimits: {scene_limits}\nRandom seed: {np.random.seed}\n#polygons: {n_polygons}\n#strings: {n_strings}\n#points: {n_points}"
        scene = create_random_field(scene_limits, n_polygons, n_strings,n_points)
        free_space = get_free_space(scene, scene_limits)
        start_point, goal_point, free_zone = get_valid_start_and_goal(free_space,scene_limits)
        if isinstance(start_point,np.ndarray) and isinstance(goal_point,np.ndarray):
            cc = CollisionChecker(scene)
            return Benchmark(name,cc,[start_point],[goal_point],description,level)
    raise TimeoutError("No starting / goal point found in max. iterations")
    
