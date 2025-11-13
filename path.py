"""Compatibility wrapper module.

This file used to contain the RRT and sampling implementation. To keep
backwards compatibility for imports that expect `path.py`, it re-exports the
cleaner implementations from `sampling.py` and `rrt.py`.
"""
from rrt import Vertex, RRT
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

from typing import Callable, Tuple, List
from pinocchio.utils import rotate
from tools import setcubeplacement, distanceToObstacle

def computepath(qinit,
                qgoal,
                cubeplacementq0,
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
    GET_GRASPPING_POSEQ = computeqgrasppose

    print('Searching for a valid path...(this may take some time)')
    # (no sampling-box visualization)
    # Instantiate RRT with desired runtime parameters then run it. This uses
    # the newer instance API so callers can inspect the tree if desired.
    rrt = RRT(root_q=cubeplacementq0,
              root_grasping_q=qinit,
              robot=robot,
              cube=cube,
              get_grasping_poseq=GET_GRASPPING_POSEQ,
              viz=viz,
              viz_delay=viz_delay,
              discretisation_steps=discretisation_steps,
              max_ik_attempts=max_ik_attempts,
              num_iter=num_iter,
              random_sampler=RANDOM_SAMPLER,
              max_delta_q=MAX_DELTA_Q,
              cubeplacementqgoal=cubeplacementqgoal,
              q_goal=qgoal,
              max_time_s=max_time_s)

    rrt.run()

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
    parser.add_argument('--max-time', type=float, default=None, help='Maximum wall time (s) for planner (abort early)')
    parser.add_argument('--max-ik-attempts', type=int, default=None, help='Maximum IK iterations per interpolation point')
    parser.add_argument('--max-delta-q', type=float, default=None, help='Maximum delta (translation) per extension (meters)')
    parser.add_argument('--viz-delay', type=float, default=DEFAULT_VIZ_DELAY, help='Visualization delay in seconds')
    parser.add_argument('--progress-every', type=int, default=50, help='How many iterations between progress logs')
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
    resolved_goal_bias = use_or_preset('goal-bias', args.goal_bias)
    resolved_max_ik_attempts = use_or_preset('max-ik-attempts', args.max_ik_attempts if args.max_ik_attempts is not None else DEFAULT_MAX_IK_ATTEMPTS)
    resolved_max_time = use_or_preset('max-time', args.max_time if args.max_time is not None else DEFAULT_MAX_TIME_S)
    # choose sampler: either let computepath build the plane sampler, or build a smart sampler here

    path, cube_placements = computepath(
        q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET,
        robot=robot,
        cube=cube,
        computeqgrasppose=computeqgrasppose,
        num_iter=resolved_num_iter,
        discretisation_steps=resolved_discretisation,
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
