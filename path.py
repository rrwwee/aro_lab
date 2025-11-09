"""Compatibility wrapper module.

This file used to contain the RRT and sampling implementation. To keep
backwards compatibility for imports that expect `path.py`, it re-exports the
cleaner implementations from `sampling.py` and `rrt.py`.
"""
from sampling import random_cube_placement
import sampling as sampling_module
from rrt import Vertex, RRT, construct_rrt
from constants import DEFAULT_NUM_ITER, DEFAULT_DISCRETISATION_STEPS, DEFAULT_VIZ_DELAY, DEFAULT_SAMPLER_SHRINK, DEFAULT_PLANE_HALF_WIDTH, DEFAULT_TABLE_Z_RANGE
from tools import setcubeplacement
import time
import logging
from typing import List

import pinocchio as pin
import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "random_cube_placement",
    "Vertex",
    "RRT",
    "construct_rrt",
]


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
                s = np.random.uniform(0.0, 1.0)
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
    GET_GRAPSING_POSEQ = computeqgrasppose

    print('Searching for a valid path...(this may take some time)')
    # (no sampling-box visualization)
    rrt = construct_rrt(
        robot=robot,
        cube=cube,
        q_init=qinit,
        q_goal=qgoal,
        cubeplacementq0=cubeplacementq0,
        cubeplacementqgoal=cubeplacementqgoal,
        num_iter=num_iter,
        random_sampler=RANDOM_SAMPLER,
        repair_sampler=repair_sampler,
        repair_max_attempts=repair_max_attempts,
        goal_bias=goal_bias,
        max_delta_q=MAX_DELTA_Q,
        discretisation_steps=discretisation_steps,
        get_grasping_poseq=GET_GRAPSING_POSEQ,
        viz=viz,
        viz_delay=viz_delay,
        max_ik_attempts=max_ik_attempts,
        max_time_s=None,
        progress_log_every=50,
    )

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
        num_iter=args.num_iter,
        discretisation_steps=args.discretisation,
        random_sampler=RANDOM_SAMPLER,
        repair_sampler=REPAIR_SAMPLER,
        viz=viz_obj,
        viz_delay=args.viz_delay,
        max_ik_attempts=args.max_ik_attempts,
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