"""Compatibility wrapper module.

This file used to contain the RRT and sampling implementation. To keep
backwards compatibility for imports that expect `path.py`, it re-exports the
cleaner implementations from `sampling.py` and `rrt.py`.
"""
from sampling import random_cube_placement
import sampling as sampling_module
from rrt import Vertex, RRT, construct_rrt
from constants import (
    DEFAULT_NUM_ITER,
    DEFAULT_DISCRETISATION_STEPS,
    DEFAULT_VIZ_DELAY,
    DEFAULT_SAMPLER_SHRINK,
    DEFAULT_PLANE_HALF_WIDTH,
    DEFAULT_TABLE_Z_RANGE,
    PRESETS,
    DEFAULT_REPAIR_ATTEMPTS,
    DEFAULT_GOAL_BIAS,
    DEFAULT_MAX_IK_ATTEMPTS,
    DEFAULT_MAX_TIME_S,
)
from tools import setcubeplacement
import time
import logging
from typing import List

import pinocchio as pin
import numpy as np

logger = logging.getLogger(__name__)

<<<<<<< HEAD
from typing import Callable, Tuple, List
from pinocchio.utils import rotate
from tools import setcubeplacement, distanceToObstacle


#### Helper Functions for Random Cube Placement Sampling ####

def get_table_borders(table: pin.robot_wrapper.RobotWrapper) -> Tuple[np.array, np.array]:
    """ Returns the minimum and maximum x, y, z coordinates of the table.
    Args:
        table: table object
    Returns:
        xyz_min: minimum x, y, z coordinates of the table's leftmost and bottommost corner
        xyz_max: maximum x, y, z coordinates of the table's rightmost and topmost corner
    """
    pin.updateGeometryPlacements(table.model, table.data, table.collision_model, table.collision_data)
    table_surface = table.collision_model.geometryObjects[0].geometry
    oMtable_surface = table.collision_data.oMg[0]
    R, t = oMtable_surface.rotation, oMtable_surface.translation


    half = table_surface.halfSide
    local_corners = np.array([[sx, sy, sz]
                                for sx in [+1, -1]
                                for sy in [+1, -1]
                                for sz in [+1, -1]]) * half
    world_corners = (R @ local_corners.T).T + t

    xyz_min = world_corners.min(axis=0)
    xyz_max = world_corners.max(axis=0)

    return xyz_min, xyz_max



def random_cube_placement(
    robot: pin.robot_wrapper.RobotWrapper, 
    cube: pin.robot_wrapper.RobotWrapper, 
    table: pin.robot_wrapper.RobotWrapper, 
    check_collisions: bool = True
    ) -> pin.pinocchio_pywrap_default.SE3:
    """ Returns a random cube placement on the table that is collision free.
    Args:
        robot: robot object
        cube: cube object
        table: table object
        check_collisions: whether to check for collisions
    Returns:
        valid_placement: random cube placement on the table that is collision free
    """
    xyz_min, xyz_max = get_table_borders(table)
    
    collision = True
    valid_placement = None

    while collision:
        x = np.random.uniform(xyz_min[0], xyz_max[0])
        y = np.random.uniform(xyz_min[1], xyz_max[1])
        # TODO: change harcoded max z to a more clever bounding
        z = np.random.uniform(0.93, 2)

        placement = pin.SE3(rotate('z', 0.), np.array([x, y, z]))
        setcubeplacement(robot, cube, placement)
        if check_collisions:
            collision = pin.computeCollisions(cube.collision_model, cube.collision_data, False)
            valid_placement = placement if not collision else valid_placement
        else:
            valid_placement = placement
            collision = False
    
    return valid_placement


#### Helper Functions and Classes for RRT ####

class Vertex:
    def __init__(self, q, grasping_q, parent):
        self.q = q 
        self.grasping_q = grasping_q
        self.parent = parent

class RRT:
    def __init__(self, 
                 root_q: pin.pinocchio_pywrap_default.SE3, 
                 root_grasping_q: np.array, 
                 robot: pin.robot_wrapper.RobotWrapper, 
                 cube: pin.robot_wrapper.RobotWrapper,
                 get_grasping_poseq: Callable):
        
        self.robot = robot
        self.cube = cube
        self.get_grasping_poseq = get_grasping_poseq

        # initialize the tree
        self.nodes = [
            Vertex(q=root_q, grasping_q=root_grasping_q, parent=None)
            ]

    @staticmethod
    def calc_distance(
        q1: np.array, 
        q2: np.array) -> float:    
     '''Return the euclidian distance between two configurations'''
     return np.linalg.norm(q2 - q1)
    
    @staticmethod
    def lerp(
        q0: np.array, 
        q1: np.array, 
        t: float
        ) -> np.array:
        """
        Performs linear interpolation between the given points and for the step t
        """
        return RRT.get_placement_from_translation((1 - t) * q0 + t * q1)
    
    @staticmethod
    def get_placement_from_translation(translation):
        return pin.SE3(rotate('z', 0.), np.array(translation))

    def add_edge(self, 
                 parent_idx: int,
                 new_q: pin.pinocchio_pywrap_default.SE3, 
                 new_grasping_q: np.array) -> None:
        
        """
        Constructs and adds a new vertex to the tree.
        """
        
        self.nodes.append(
            Vertex(q=new_q,
                   grasping_q=new_grasping_q,
                   parent=self.nodes[parent_idx])
        )

    def get_nearest_vertex(self, 
                           q: pin.pinocchio_pywrap_default.SE3
                           ) -> Tuple[int, Vertex]:
        
        """
        Args:
            q: random configuration of the cube (SE3)
        Returns:
            nearest_vertex_idx: nearest vertex index in the tree
            nearest_vertex: nearest vertex
        """
        nearest_vertex_idx = None
        smallest_distance = None

        for i, vertex in enumerate(self.nodes): 
            dist = self.calc_distance(vertex.q.translation, q.translation)
            if not smallest_distance or dist <= smallest_distance:
                smallest_distance = dist
                nearest_vertex_idx = i

        return nearest_vertex_idx, self.nodes[nearest_vertex_idx]
    
    def get_q_new(self, 
                  q_nearest_vertex: Vertex, 
                  q_rand: pin.pinocchio_pywrap_default.SE3, 
                  discretisation_steps: int,
                  max_delta_q: int = None,
                  ) -> Tuple[pin.pinocchio_pywrap_default.SE3, np.array, bool]:
        
        """ Returns a new configuration that is the closest to the random configuration `q_rand` that can be grasped and is collison free.
        Args:
            q_nearest_vertex: nearest vertex
            q_rand: random configuration
            discretisation_steps: number of discretisation steps
            max_delta_q: maximum delta q
        Returns:
            q_new: new configuration
            grasping_q: grasping pose for the new configuration
            q_new_found: True if the new configuration (other than the `q_nearest_vertex`) is found, False otherwise    
        """
        
        q_new_found = True
        q_end = q_rand.copy()

        dist = self.calc_distance(
            q_nearest_vertex.q.translation, 
            q_rand.translation)
        
        if max_delta_q is not None and dist > max_delta_q:
            q_end = self.lerp(
                q_nearest_vertex.q.translation, 
                q_rand.translation, 
                max_delta_q/dist)

        if self.get_grasping_poseq is not None:
            dt = 1 / discretisation_steps
            for i in range(1, discretisation_steps):
                q = self.lerp(
                    q_nearest_vertex.q.translation, 
                    q_end.translation, 
                    dt*i)
                grasping_q, found_grasping_pose = self.get_grasping_poseq(
                    robot=self.robot, 
                    qcurrent=q_nearest_vertex.grasping_q,
                    cube=self.cube,
                    cubetarget=q,
                )
                if not found_grasping_pose or (found_grasping_pose and distanceToObstacle(self.robot, grasping_q) < 2e-3):
                    if i - 1 == 0:
                        q_new_found = False
                    return self.lerp(
                        q_nearest_vertex.q.translation, 
                        q_end.translation, 
                        dt*(i-1)), grasping_q, q_new_found
        else:
            grasping_q = None
                
        return q_end, grasping_q, q_new_found
    
    def check_edge(
            self,
            q_latest_new: pin.pinocchio_pywrap_default.SE3,
            q_latest_new_grasping_pose: np.array,
            q_goal: pin.pinocchio_pywrap_default.SE3,
            discretisation_steps: int = 1000,
            ) -> bool:


        """ Checks if there is a valid edge between the latest new configuration and the goal configuration.
        Args:
            q_latest_new: latest new configuration
            q_latest_new_grasping_pose: grasping pose for the latest new configuration
            q_goal: goal configuration
            discretisation_steps: number of discretisation steps
        Returns:
            valid_edge: True if there is a valid edge, False otherwise
        """
        
        valid_edge = False
        q_latest_new_vertex = Vertex(
            q=q_latest_new,
            grasping_q=q_latest_new_grasping_pose,
            parent=None
        )
        
        # check if we can get to the q_goal from the q_new directly
        q_new, q_new_grasping_q, q_new_found = self.get_q_new(
            q_nearest_vertex=q_latest_new_vertex,
            q_rand=q_goal,
            discretisation_steps=discretisation_steps,
            max_delta_q=None
        )

        if q_new_found:
            distance_to_target = self.calc_distance(q_new.translation, q_goal.translation)
            if distance_to_target < 1e-3:
                valid_edge = True

        return valid_edge
    
    def get_path_from_rrt(self) -> List[Vertex]:
        """ Returns a list of vertices from the root to the latest vertex.
        If there was a valid edge between the latest vertex and the goal configuration, the latest vertex is the vertex with the goal configuration.
        Returns:
            vertices: list of vertices
        """
        vertices = []
        last_vertex = self.nodes[-1]

        while last_vertex.parent is not None:
            vertices.insert(0, last_vertex)
            last_vertex = last_vertex.parent

        vertices.insert(0, last_vertex)

        return vertices
    
    def get_shortcut_from_rrt_path(self, 
                                   discretisation_steps: int = 100,
                                   ) -> List[Vertex]:
        
        """ Returns a shortcut path from the root to the latest vertex.
        Args:
            discretisation_steps: number of discretisation steps
        Returns:
            short_path: list of vertices from the root to the latest vertex that has no redundant intermediate vertices  
        """

        path_vertices = self.get_path_from_rrt()
        short_path = []
        start_idx = 0
        
        while start_idx < len(path_vertices) - 3:
            current_vertex = path_vertices[start_idx]
            short_path.append(current_vertex)

            # search in the next nodes starting from the furthest one
            next_vertex_idx = len(path_vertices) - 1
            while next_vertex_idx > start_idx + 1:
                next_vertex = path_vertices[next_vertex_idx]
                is_valid_edge = self.check_edge(
                    q_latest_new=current_vertex.q,
                    q_latest_new_grasping_pose=current_vertex.grasping_q,
                    q_goal=next_vertex.q,
                    discretisation_steps=discretisation_steps
                )
                if is_valid_edge:
                    start_idx = next_vertex_idx
                    break
                next_vertex_idx -= 1

            if not is_valid_edge:
                start_idx += 1

        short_path.extend(path_vertices[start_idx:])

        return short_path



#### The main function for constructing the RRT ####

def construct_rrt(robot: pin.robot_wrapper.RobotWrapper,
                  cube: pin.robot_wrapper.RobotWrapper,
                  q_init: np.array, 
                  q_goal: np.array,
                  cubeplacementq0: pin.pinocchio_pywrap_default.SE3,
                  cubeplacementqgoal: pin.pinocchio_pywrap_default.SE3,
                  num_iter: int, 
                  radnom_sampler: Callable,
                  max_delta_q: float = None,
                  discretisation_steps: int = 100, 
                  get_grasping_poseq: Callable = None,
                  ):

    """ Constructs the RRT from the root to the goal configuration.
    Args:
        robot: robot object
        cube: cube object
        q_init: initial grasping pose
        q_goal: goal grasping pose
        cubeplacementq0: initial cube placement
        cubeplacementqgoal: goal cube placement
        num_iter: number of iterations of sampling random cube placements
        radnom_sampler: random cube placement sampler
        max_delta_q: maximum delta q between the current and the end configuration used to get q_new configuration
        discretisation_steps: number of discretisation steps used in the get_q_new and check_edge function
        get_grasping_poseq: grasping pose calculator from the cube placement (inverse geometry).
    Returns:
        rrt: RRT object
    """
    rrt = RRT(
        root_q=cubeplacementq0,
        root_grasping_q=q_init,
        robot=robot,
        cube=cube,
        get_grasping_poseq=get_grasping_poseq
        )

    for _ in range(num_iter):
        random_cp_q = radnom_sampler()
        cp_q_nearest_idx, cp_q_nearest_vertex = rrt.get_nearest_vertex(random_cp_q)

        cp_q_new, cp_q_new_grasping_pose, cp_q_new_found = rrt.get_q_new(
            q_nearest_vertex=cp_q_nearest_vertex,
            q_rand=random_cp_q,
            discretisation_steps=discretisation_steps,
            max_delta_q=max_delta_q)
        
        if cp_q_new_found:
            rrt.add_edge(
                parent_idx=cp_q_nearest_idx,
                new_q=cp_q_new,
                new_grasping_q=cp_q_new_grasping_pose,
            )

            is_valid_edge = rrt.check_edge(
                q_latest_new=cp_q_new,
                q_latest_new_grasping_pose=cp_q_new_grasping_pose,
                q_goal=cubeplacementqgoal,
                discretisation_steps=discretisation_steps)
            
            if is_valid_edge:
                rrt.add_edge(parent_idx=len(rrt.nodes) - 1,
                             new_q=cubeplacementqgoal,
                             new_grasping_q=q_goal)
                
                return rrt
    
    return None

#returns a collision free path from qinit to qgoal under grasping constraints
#the path is expressed as a list of configurations
def computepath(qinit, 
                qgoal, 
                cubeplacementq0, 
=======
__all__ = [
    "random_cube_placement",
    "Vertex",
    "RRT",
    "construct_rrt",
]


def computepath(qinit,
                qgoal,
                cubeplacementq0,
>>>>>>> main
                cubeplacementqgoal,
                robot=None,
                cube=None,
                computeqgrasppose=None,
                num_iter: int = DEFAULT_NUM_ITER,
                discretisation_steps: int = DEFAULT_DISCRETISATION_STEPS,
                max_ik_attempts: int = None,
                max_time_s: float = DEFAULT_MAX_TIME_S,
                post_path_wait: float = 5.0,
                random_sampler=None,
                repair_sampler=None,
                repair_max_attempts: int = 5,
                goal_bias: float = 0.02,
                viz=None,
                viz_delay: float = DEFAULT_VIZ_DELAY):
    """ Returns a collision free path from qinit to qgoal under grasping constraints.
    Args:
        qinit: initial configuration
        qgoal: goal configuration
        cubeplacementq0: initial cube placement
        cubeplacementqgoal: goal cube placement
    Returns:
        path: list of configurations
    """
    robot = robot or globals().get("robot")
    cube = cube or globals().get("cube")
    computeqgrasppose = computeqgrasppose or globals().get("computeqgrasppose")

<<<<<<< HEAD
    NUM_ITER = 500
    RANDOM_SAMPLER = lambda: random_cube_placement(robot=robot, cube=cube, table=table)
    MAX_DELTA_Q = None
    DISCRETISATION_STEPS = 200
=======
    # If no sampler supplied, build a constrained sampler on the vertical plane
    # that intersects the initial and final cube placements.
    sampling_box_ranges = None
    if random_sampler is None:
        p0 = np.array(cubeplacementq0.translation)
        p1 = np.array(cubeplacementqgoal.translation)

        p0_xy = p0[:2]
        p1_xy = p1[:2]
        seg = p1_xy - p0_xy
        L = np.linalg.norm(seg)
        if L < 1e-6:
            u = np.array([1.0, 0.0])
        else:
            u = seg / L
        perp = np.array([-u[1], u[0]])

        half_width = DEFAULT_PLANE_HALF_WIDTH

        # z range uses the default table z range
        zmin, zmax = DEFAULT_TABLE_Z_RANGE

        def plane_sampler():
            while True:
                s = np.random.uniform(-1.0, 2.0)
                t = np.random.uniform(-half_width, half_width)
                xy = (1.0 - s) * p0_xy + s * p1_xy + perp * t
                z = np.random.uniform(zmin, zmax)
                placement = pin.SE3(np.eye(3), np.array([xy[0], xy[1], z]))
                setcubeplacement(robot, cube, placement)
                has_collision = pin.computeCollisions(cube.collision_model, cube.collision_data, False)
                if not has_collision:
                    return placement

        RANDOM_SAMPLER = plane_sampler

        # sampling box for visualization: a thin box centered between p0 and p1
        center_xy = (p0_xy + p1_xy) / 2.0
        center = np.array([center_xy[0], center_xy[1], (zmin + zmax) / 2.0])
        dims_u = max(L, 1e-6)
        dims_perp = 2.0 * half_width
        dims_z = zmax - zmin

        # build axis-aligned x/y range for visualization convenience
        corners_xy = [p0_xy + perp * (-half_width), p0_xy + perp * (half_width),
                      p1_xy + perp * (-half_width), p1_xy + perp * (half_width)]
        xs = [c[0] for c in corners_xy]
        ys = [c[1] for c in corners_xy]
        x_range = (min(xs), max(xs))
        y_range = (min(ys), max(ys))
        z_range = (zmin, zmax)
        sampling_box_ranges = (x_range, y_range, z_range)
    else:
        RANDOM_SAMPLER = random_sampler
    MAX_DELTA_Q = None
>>>>>>> main
    GET_GRAPSING_POSEQ = computeqgrasppose

    print('Searching for a valid path...(this may take some time)')
    # (no sampling-box visualization)
    # Instantiate RRT with desired runtime parameters then run it. This uses
    # the newer instance API so callers can inspect the tree if desired.
    rrt = RRT(root_q=cubeplacementq0,
              root_grasping_q=qinit,
              robot=robot,
              cube=cube,
              get_grasping_poseq=GET_GRAPSING_POSEQ,
              viz=viz,
              viz_delay=viz_delay,
              discretisation_steps=discretisation_steps,
              max_ik_attempts=max_ik_attempts,
              num_iter=num_iter,
              random_sampler=RANDOM_SAMPLER,
              max_delta_q=MAX_DELTA_Q,
              cubeplacementqgoal=cubeplacementqgoal,
              q_goal=qgoal,
              max_time_s=max_time_s,
              progress_log_every=50,
              repair_sampler=repair_sampler,
              repair_max_attempts=repair_max_attempts,
              goal_bias=goal_bias)

    rrt = rrt.run()

    if not rrt:
        print(
            """Valid Path was not found! 
            Try to increase discretisation steps or try with different start and end configurations.""")

        return [], []

    print('Path found! Retrieving shortcut path from RRT...')

    path_vertices = rrt.get_shortcut_from_rrt_path()

    path_configurations = [vertex.grasping_q for vertex in path_vertices]
    cube_placements = [vertex.q for vertex in path_vertices]

    print('Path is ready!')

    # visualise the cube placements by stepping the robot so user sees the grasping poses
    if viz:
        # pause briefly so user sees the final planning state before playback
        time.sleep(post_path_wait)
        for v in path_vertices:
            setcubeplacement(robot, cube, v.q)
            viz.display(v.grasping_q)
            time.sleep(max(0.01, viz_delay))

    # log metrics if rrt stored them
    if hasattr(rrt, 'metrics'):
        logger.info('Path metrics: %s', rrt.metrics)

    return path_configurations, cube_placements


def displaypath(robot, cube, path, cube_placements, dt, viz):
    for q, cube_placement in zip(path, cube_placements):
        setcubeplacement(robot, cube, cube_placement)
        viz.display(q)
        time.sleep(dt)


if __name__ == "__main__":
    import argparse
    from tools import setupwithmeshcat
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from inverse_geometry import computeqgrasppose
    import logging

    parser = argparse.ArgumentParser(description='Run RRT planner for cube placements')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization during planning')
    parser.add_argument('--discretisation', type=int, default=DEFAULT_DISCRETISATION_STEPS, help='Discretisation steps for interpolation')
    parser.add_argument('--num-iter', type=int, default=DEFAULT_NUM_ITER, help='Maximum RRT iterations/samples')
    parser.add_argument('--sampler', choices=['plane', 'smart'], default='plane', help='Which placement sampler to use')
    # smart sampler params
    parser.add_argument('--smart-init-sigma', type=float, default=0.02, help='Initial sigma (m) for smart midpoint sampler')
    parser.add_argument('--smart-scale', type=float, default=1.5, help='Sigma inflation factor for smart sampler')
    parser.add_argument('--smart-max-sigma', type=float, default=0.2, help='Maximum sigma (m) for smart sampler')
    parser.add_argument('--smart-max-attempts', type=int, default=30, help='Max attempts for smart sampler before fallback')
    parser.add_argument('--max-time', type=float, default=None, help='Maximum wall time (s) for planner (abort early)')
    parser.add_argument('--max-ik-attempts', type=int, default=None, help='Maximum IK iterations per interpolation point')
    parser.add_argument('--max-delta-q', type=float, default=None, help='Maximum delta (translation) per extension (meters)')
    parser.add_argument('--viz-delay', type=float, default=DEFAULT_VIZ_DELAY, help='Visualization delay in seconds')
    parser.add_argument('--progress-every', type=int, default=50, help='How many iterations between progress logs')
    parser.add_argument('--repair-attempts', type=int, default=5, help='How many local repair attempts to try when an extension fails')
    parser.add_argument('--smart-global', action='store_true', help='Use the smart sampler as the global sampler instead of repair-only')
    parser.add_argument('--goal-bias', type=float, default=0.02, help='Probability to sample the goal directly on each iteration (0.0-1.0)')
    parser.add_argument('--playback', action='store_true', help='Show the final path even if planning ran with --no-viz')
    parser.add_argument('--preset', choices=['fast', 'default', 'robust'], default=None, help='Use a preset bundle of planner defaults')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # `setupwithmeshcat` returns (robot, cube, viz) in the current helpers.
    # Unpack accordingly and avoid depending on any `table` object.
    robot, cube, viz = setupwithmeshcat()

    q = robot.q0.copy()
    q0, successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz=None)
    qe, successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET, viz=None)

    if not (successinit and successend):
        print("error: invalid initial or end configuration")
        raise SystemExit(1)

    viz_obj = None if args.no_viz else viz
    # Resolve preset values (CLI args override preset). We detect explicit
    # CLI overrides by checking sys.argv for the flag name.
    import sys
    preset_values = PRESETS.get(args.preset) if args.preset is not None else {}
    def use_or_preset(flag_name, current_value):
        # Return CLI-provided value if flag present, otherwise preset or current_value
        if f'--{flag_name}' in sys.argv:
            return current_value
        if preset_values and flag_name in preset_values:
            return preset_values[flag_name]
        return current_value

    resolved_num_iter = use_or_preset('num-iter', args.num_iter)
    resolved_discretisation = use_or_preset('discretisation', args.discretisation)
    resolved_repair_attempts = use_or_preset('repair-attempts', args.repair_attempts)
    resolved_goal_bias = use_or_preset('goal-bias', args.goal_bias)
    resolved_max_ik_attempts = use_or_preset('max-ik-attempts', args.max_ik_attempts if args.max_ik_attempts is not None else DEFAULT_MAX_IK_ATTEMPTS)
    resolved_max_time = use_or_preset('max-time', args.max_time if args.max_time is not None else DEFAULT_MAX_TIME_S)
    # choose sampler: either let computepath build the plane sampler, or build a smart sampler here
    RANDOM_SAMPLER = None
    REPAIR_SAMPLER = None
    if args.sampler == 'smart':
        # build a plane sampler to use as fallback
        p0 = np.array(CUBE_PLACEMENT.translation)
        p1 = np.array(CUBE_PLACEMENT_TARGET.translation)

        p0_xy = p0[:2]
        p1_xy = p1[:2]
        seg = p1_xy - p0_xy
        L = np.linalg.norm(seg)
        if L < 1e-6:
            u = np.array([1.0, 0.0])
        else:
            u = seg / L
        perp = np.array([-u[1], u[0]])

        half_width = DEFAULT_PLANE_HALF_WIDTH
        zmin, zmax = DEFAULT_TABLE_Z_RANGE

        def plane_sampler():
            while True:
                s = np.random.uniform(0.0, 1.0)
                t = np.random.uniform(-half_width, half_width)
                xy = (1.0 - s) * p0_xy + s * p1_xy + perp * t
                z = np.random.uniform(zmin, zmax)
                placement = pin.SE3(np.eye(3), np.array([xy[0], xy[1], z]))
                setcubeplacement(robot, cube, placement)
                has_collision = pin.computeCollisions(cube.collision_model, cube.collision_data, False)
                if not has_collision:
                    return placement

        # If --smart-global selected, use the smart sampler as the global
        # sampler; otherwise keep the plane sampler global and use smart as a
        # local repair strategy. This keeps exploration simple by default but
        # lets the user switch behaviour.
        if args.smart_global:
            RANDOM_SAMPLER = sampling_module.make_smart_midpoint_sampler(
                robot=robot,
                cube=cube,
                p0=CUBE_PLACEMENT,
                p1=CUBE_PLACEMENT_TARGET,
                initial_sigma=args.smart_init_sigma,
                sigma_scale=args.smart_scale,
                max_sigma=args.smart_max_sigma,
                max_attempts=args.smart_max_attempts,
                fallback_sampler=plane_sampler,
                anisotropic=True,
                z_range=DEFAULT_TABLE_Z_RANGE,
                viz=(None if args.no_viz else viz),
                viz_delay=args.viz_delay,
            )
            REPAIR_SAMPLER = None
        else:
            # use the plane sampler as the global sampler and the smart sampler as a
            # local repair strategy. This keeps global exploration simple while the
            # smart sampler is used to attempt local fixes when an extension fails.
            RANDOM_SAMPLER = plane_sampler

            # create the smart sampler closure (used as repair_sampler)
            REPAIR_SAMPLER = sampling_module.make_smart_midpoint_sampler(
                robot=robot,
                cube=cube,
                p0=CUBE_PLACEMENT,
                p1=CUBE_PLACEMENT_TARGET,
                initial_sigma=args.smart_init_sigma,
                sigma_scale=args.smart_scale,
                max_sigma=args.smart_max_sigma,
                max_attempts=args.smart_max_attempts,
                fallback_sampler=plane_sampler,
                anisotropic=True,
                z_range=DEFAULT_TABLE_Z_RANGE,
                viz=(None if args.no_viz else viz),
                viz_delay=args.viz_delay,
            )

    path, cube_placements = computepath(
        q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET,
        robot=robot,
        cube=cube,
        computeqgrasppose=computeqgrasppose,
        num_iter=resolved_num_iter,
        discretisation_steps=resolved_discretisation,
        random_sampler=RANDOM_SAMPLER,
        repair_sampler=REPAIR_SAMPLER,
        repair_max_attempts=resolved_repair_attempts,
        goal_bias=resolved_goal_bias,
        viz=viz_obj,
        viz_delay=args.viz_delay,
        max_ik_attempts=resolved_max_ik_attempts,
        # forward wall-time
        # Note: computepath forwards this into construct_rrt/RRT.run
        max_time_s=resolved_max_time,
    )

    # optionally playback the final path. If --playback is set we show the
    # final trajectory even when planning was run headless with --no-viz.
    if path and (args.playback or not args.no_viz):
        # ensure visualiser is ready (short pause) then play back slowly
        try:
            time.sleep(0.5)
            displaypath(robot, cube, path, cube_placements, dt=0.5, viz=viz)
        except Exception:
            # Best-effort: if viz isn't available or playback fails, continue.
            logger.exception('Playback failed')

    if hasattr(rrt := __import__('rrt'), 'rrt'):
        # noop — ensure rrt module imported so metrics are available in namespace
        pass

    # if rrt metrics exist log them
    try:
        import rrt as _rrt
        if hasattr(_rrt, 'RRT') and hasattr(_rrt, 'construct_rrt'):
            # rrt.metrics is attached to the returned rrt instance; cannot access it here unless we returned it
            pass
    except Exception:
        pass

    print('Done')