#!/usr/bin/env python3
"""Sampling helpers for cube placement on a table.

This module contains pure sampling logic (no RRT) so it can be read,
tested, and understood independently.
"""
from typing import Tuple, Callable

import logging
import numpy as np
import pinocchio as pin
from pinocchio.utils import rotate

from tools import setcubeplacement
from constants import DEFAULT_TABLE_Z_RANGE, DEFAULT_CHECK_COLLISIONS, DEFAULT_VIZ_DELAY, SAMPLE_DISPLAY_EVERY
import time

logger = logging.getLogger(__name__)

# Lightweight profiling counters populated during sampling calls.
# Accessible as sampling.metrics after running the sampler.
metrics = {
    'samples': 0,
    'geom_updates': 0,
    'collision_checks': 0,
    'time_collision_checks': 0.0,
    'time_viz': 0.0,
    'time_sampling_loop': 0.0,
}


def get_table_borders(table: pin.robot_wrapper.RobotWrapper) -> Tuple[np.array, np.array]:
    """Return the axis-aligned bounding box (min, max) of the table collision surface.

    Args:
        table: pinocchio RobotWrapper for the table which contains geometry information.

    Returns:
        (xyz_min, xyz_max): two length-3 numpy arrays with world-frame corner coordinates.
    """
    # ensure geometry placements are up-to-date
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

    logger.debug("Table borders: min=%s max=%s", xyz_min, xyz_max)
    return xyz_min, xyz_max


def random_cube_placement(
    robot: pin.robot_wrapper.RobotWrapper,
    cube: pin.robot_wrapper.RobotWrapper,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_range: Tuple[float, float] = DEFAULT_TABLE_Z_RANGE,
    check_collisions: bool = DEFAULT_CHECK_COLLISIONS,
    sampler: Callable[[], float] = None,
    viz=None,
    viz_delay: float = DEFAULT_VIZ_DELAY,
    display_every: int = SAMPLE_DISPLAY_EVERY,
    collision_check_every: int = 1,
) -> pin.SE3:
    """Sample a random placement (SE3) for the cube on the table surface.

    The function places the cube at a random (x,y) inside the table AABB and a
    z sampled from `z_range`. It sets the cube placement on the robot via
    `setcubeplacement` and optionally checks for collisions.

    Args:
        robot, cube, table: pinocchio wrappers used for updating placements.
        check_collisions: if True, verify the placement is collision-free.
        z_range: tuple (z_min, z_max) for sampling height above world origin.
        sampler: optional callable returning a float in [0,1] used to sample x/y.

    Returns:
        A pin.SE3 placement that is collision-free (if check_collisions) or the
        last sampled placement if checks are disabled.
    """
    sampler = sampler or np.random.random

    sample_count = 0
    last_placement = None
    while True:
        x = np.random.uniform(x_range[0], x_range[1])
        y = np.random.uniform(y_range[0], y_range[1])
        z = np.random.uniform(z_range[0], z_range[1])

        t_loop = time.perf_counter()
        placement = pin.SE3(rotate('z', 0.0), np.array([x, y, z]))
        last_placement = placement

        sample_count += 1
        metrics['samples'] += 1

        # only update the visualiser occasionally to avoid blocking
        if viz is not None and (sample_count % display_every) == 0:
            t_v = time.perf_counter()
            q_disp = getattr(robot, 'q', None)
            if q_disp is not None:
                viz.display(q_disp)
            else:
                if hasattr(viz, 'render'):
                    viz.render()
            dt_v = time.perf_counter() - t_v
            metrics['time_viz'] += dt_v
            time.sleep(viz_delay)

        if not check_collisions:
            metrics['time_sampling_loop'] += time.perf_counter() - t_loop
            return last_placement

        # perform the (more expensive) geometry update and collision check only every N samples
        if (sample_count % max(1, collision_check_every)) != 0:
            # don't call geometry updates for this sample, keep sampling
            metrics['time_sampling_loop'] += time.perf_counter() - t_loop
            continue

        # now set placement on the scene and do collision checking
        from tools import set_and_check_cube_placement
        metrics['geom_updates'] += 1
        has_collision, dt_coll = set_and_check_cube_placement(robot, cube, last_placement)
        metrics['collision_checks'] += 1
        metrics['time_collision_checks'] += dt_coll
        metrics['time_sampling_loop'] += time.perf_counter() - t_loop
        if not has_collision:
            logger.debug("Found collision-free placement after %d samples", sample_count)
            return last_placement
        # otherwise keep sampling
    logger.debug("Sampled placement in collision at sample %d, retrying", sample_count)


def make_smart_midpoint_sampler(robot: pin.robot_wrapper.RobotWrapper,
                                cube: pin.robot_wrapper.RobotWrapper,
                                p0: pin.SE3,
                                p1: pin.SE3,
                                initial_sigma: float = 0.02,
                                sigma_scale: float = 1.5,
                                max_sigma: float = 0.2,
                                max_attempts: int = 30,
                                fallback_sampler: Callable[[], pin.SE3] = None,
                                anisotropic: bool = True,
                                z_range: Tuple[float, float] = DEFAULT_TABLE_Z_RANGE,
                                viz=None,
                                viz_delay: float = DEFAULT_VIZ_DELAY) -> Callable[[], pin.SE3]:
        """Create a sampler that prefers the straight-line midpoint between p0 and p1.

        The returned callable takes no arguments and returns a collision-free
        pin.SE3 placement. It first tries the exact midpoint, then samples from a
        Gaussian around the midpoint with adaptive variance inflation until a
        collision-free placement is found or max_attempts is reached. If nothing
        found, calls fallback_sampler if provided.
        """
        # Precompute useful values
        p0_xy = np.array(p0.translation)[:2]
        p1_xy = np.array(p1.translation)[:2]
        mid_xy = (p0_xy + p1_xy) / 2.0
        mid_z = (np.array(p0.translation)[2] + np.array(p1.translation)[2]) / 2.0

        seg = p1_xy - p0_xy
        L = np.linalg.norm(seg)
        if L < 1e-6:
            u = np.array([1.0, 0.0])
        else:
            u = seg / L
        perp = np.array([-u[1], u[0]])

        def sampler() -> pin.SE3:
            # try exact midpoint first
            placement = pin.SE3(np.eye(3), np.array([mid_xy[0], mid_xy[1], mid_z]))
            # do a timed collision check for the midpoint
            from tools import set_and_check_cube_placement
            metrics['geom_updates'] += 1
            has_collision, dt = set_and_check_cube_placement(robot, cube, placement)
            metrics['collision_checks'] += 1
            metrics['time_collision_checks'] += dt
            if not has_collision:
                return placement

            # deterministic-offset-first: try a small set of deterministic offsets
            # (along the segment and perpendicular) before doing stochastic sampling.
            attempts_total = 1
            # choose a small grid of offsets based on initial_sigma
            base_eps = max(1e-4, initial_sigma)
            eps_factors = [0.5, 1.0, 2.0]
            det_offsets = []
            for f in eps_factors:
                eps = base_eps * f
                # along, perp, and diagonal combinations
                det_offsets.extend([
                    (eps, 0.0), (-eps, 0.0),
                    (0.0, eps), (0.0, -eps),
                    (eps, eps), (eps, -eps), (-eps, eps), (-eps, -eps)
                ])

            for along_eps, perp_eps in det_offsets:
                if attempts_total >= max_attempts:
                    break
                xy = mid_xy + u * along_eps + perp * perp_eps
                # bias z toward midpoint
                zmin, zmax = z_range
                z = np.clip(np.random.normal(mid_z, initial_sigma * 0.5), zmin, zmax)
                placement = pin.SE3(np.eye(3), np.array([xy[0], xy[1], z]))
                from tools import set_and_check_cube_placement
                metrics['geom_updates'] += 1
                has_collision, dt = set_and_check_cube_placement(robot, cube, placement)
                metrics['collision_checks'] += 1
                metrics['time_collision_checks'] += dt
                attempts_total += 1
                if not has_collision:
                    logger.debug("Smart sampler found placement on deterministic offset after %d attempts", attempts_total)
                    return placement

            sigma = initial_sigma
            attempts = attempts_total
            while attempts < max_attempts:
                attempts += 1
                # anisotropic sampling in the xy-plane: smaller along segment, larger perpendicular
                if anisotropic:
                    sigma_along = max(1e-6, sigma * 0.5)
                    sigma_perp = sigma
                    r_along = np.random.normal(0.0, sigma_along)
                    r_perp = np.random.normal(0.0, sigma_perp)
                    xy = mid_xy + u * r_along + perp * r_perp
                else:
                    xy = mid_xy + np.random.normal(0.0, sigma, size=2)

                # sample z from the provided z_range, but bias toward midpoint z
                zmin, zmax = z_range
                z = np.clip(np.random.normal(mid_z, sigma), zmin, zmax)

                placement = pin.SE3(np.eye(3), np.array([xy[0], xy[1], z]))
                from tools import set_and_check_cube_placement
                metrics['geom_updates'] += 1
                has_collision, dt = set_and_check_cube_placement(robot, cube, placement)
                metrics['collision_checks'] += 1
                metrics['time_collision_checks'] += dt
                if not has_collision:
                    logger.debug("Smart sampler found placement after %d attempts (sigma=%.4f)", attempts, sigma)
                    return placement

                # increase variance and try again
                sigma = min(max_sigma, sigma * sigma_scale)

            # fallback if provided
            if fallback_sampler is not None:
                logger.debug("Smart sampler failing after %d attempts; using fallback sampler", attempts)
                return fallback_sampler()

            # as a last resort return the midpoint (may be colliding)
            return placement

        return sampler
