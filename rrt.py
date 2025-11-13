#!/usr/bin/env python3
"""Standard RRT implementation for cube placement planning.

This module contains a focused RRT implementation that works with cube
placements (pin.SE3) and optional grasping-pose callbacks. The implementation
is intentionally minimal and readable.
"""
from typing import Callable, Tuple, List, Optional

import logging
from inverse_geometry import computeqgrasppose
import numpy as np
import pinocchio as pin
from pinocchio.utils import rotate

from constants import EDGE_DISTANCE_TOL, DEFAULT_DISCRETISATION_STEPS, DEFAULT_CHECK_EDGE_STEPS
from tools import setcubeplacement, distanceToObstacle
from constants import DEFAULT_VIZ_DELAY
import time

logger = logging.getLogger(__name__)


from dataclasses import dataclass


@dataclass
class Vertex:
    q: pin.SE3
    grasping_q: Optional[np.ndarray]
    parent: Optional['Vertex'] = None


class RRT:
    """A small, explicit RRT tree that stores vertices of cube placements.

    Attributes:
        nodes: list of Vertex objects, root at index 0.
    """

    def __init__(self,
                 root_q: pin.SE3,
                 root_grasping_q: Optional[np.ndarray],
                 robot: pin.robot_wrapper.RobotWrapper,
                 cube: pin.robot_wrapper.RobotWrapper,
                 get_grasping_poseq: Optional[Callable] = None,
                 viz=None,
                 viz_delay: float = DEFAULT_VIZ_DELAY,
                 discretisation_steps: int = DEFAULT_DISCRETISATION_STEPS,
                 max_ik_attempts: Optional[int] = None,
                 # Run-time parameters (moved into the instance so run() needs no args)
                 num_iter: int = 1000,
                 random_sampler: Optional[Callable] = None,
                 max_delta_q: Optional[float] = None,
                 cubeplacementqgoal: Optional[pin.SE3] = None,
                 q_goal: Optional[np.ndarray] = None,
                 max_time_s: Optional[float] = None):

        self.robot = robot
        self.cube = cube
        self.get_grasping_poseq = get_grasping_poseq

        self.nodes: List[Vertex] = [Vertex(q=root_q, grasping_q=root_grasping_q, parent=None)]
        # store common runtime parameters on the RRT instance so methods don't
        # need to be passed them repeatedly
        self.viz = viz
        self.viz_delay = viz_delay
        self.discretisation_steps = discretisation_steps
        self.max_ik_attempts = max_ik_attempts
        # store run-time configuration so run() requires no arguments
        self.num_iter = num_iter
        self.random_sampler = random_sampler
        self.max_delta_q = max_delta_q
        self.cubeplacementqgoal = cubeplacementqgoal
        self.q_goal = q_goal
        self.max_time_s = max_time_s
        # profiling counters for IK, viz and edge checks

    @staticmethod
    def calc_distance(q1, q2) -> float:
        """Euclidean distance between two translations or SE3s."""
        if hasattr(q1, 'translation'):
            t1 = np.array(q1.translation)
        else:
            t1 = np.array(q1)

        if hasattr(q2, 'translation'):
            t2 = np.array(q2.translation)
        else:
            t2 = np.array(q2)

        return np.linalg.norm(t2 - t1)

    @staticmethod
    def lerp(q0, q1, t: float):
        """Linear interpolation of translations and return a pin.SE3 placement."""
        if hasattr(q0, 'translation'):
            a = np.array(q0.translation)
        else:
            a = np.array(q0)

        if hasattr(q1, 'translation'):
            b = np.array(q1.translation)
        else:
            b = np.array(q1)

        trans = (1 - t) * a + t * b
        return pin.SE3(rotate('z', 0.0), trans)

    def add_edge(self, parent_idx: int, new_q: pin.SE3, new_grasping_q: Optional[np.ndarray]) -> None:
        self.nodes.append(Vertex(q=new_q, grasping_q=new_grasping_q, parent=self.nodes[parent_idx]))

    def get_nearest_vertex(self, q: pin.SE3) -> Tuple[int, Vertex]:
        nearest_idx = 0
        smallest = float('inf')
        for i, v in enumerate(self.nodes):
            dist = self.calc_distance(v.q, q)
            if dist < smallest:
                smallest = dist
                nearest_idx = i
        return nearest_idx, self.nodes[nearest_idx]

    def get_q_new(self,
                  q_nearest_vertex: Vertex,
                  q_rand: pin.SE3,
                  max_delta_q: Optional[float] = None) -> Tuple[pin.SE3, Optional[np.ndarray], bool]:

        q_new_found = True
        q_end = q_rand

        dist = self.calc_distance(q_nearest_vertex.q, q_rand)
        if max_delta_q is not None and dist > max_delta_q:
            q_end = self.lerp(q_nearest_vertex.q, q_rand, max_delta_q / dist)

        grasping_q = None
        if self.get_grasping_poseq is not None:
            # allow caller to override discretisation for this call; otherwise use instance default
            dt = 1.0 / self.discretisation_steps
            # seed IK with the nearest vertex grasping pose and then update the seed
            seed_grasping_q = q_nearest_vertex.grasping_q
            for i in range(1, self.discretisation_steps):
                q = self.lerp(q_nearest_vertex.q, q_end, dt * i)
                # optionally update visualization so user sees the robot attempt
                if self.viz is not None:
                    t_v = time.perf_counter()
                    setcubeplacement(self.robot, self.cube, q)
                    q_disp = getattr(self.robot, 'q', None)
                    if q_disp is not None:
                        self.viz.display(q_disp)
                    else:
                        if hasattr(self.viz, 'render'):
                            self.viz.render()
                    dt_v = time.perf_counter() - t_v
                    time.sleep(self.viz_delay)

                # Use the IK wrapper to call the IK callback in a
                # backwards-compatible way and time the call. We set
                # viz_sleep=False so display calls (if any) don't sleep.
                from ik_utils import call_ik
                grasping_q, found = computeqgrasppose(
                    robot=self.robot,
                    qcurrent=seed_grasping_q,
                    cube=self.cube,
                    cubetarget=q,
                    viz=(self.viz if self.viz is not None else None),
                    max_attempts=(self.max_ik_attempts if self.max_ik_attempts is not None else 1000),
                    viz_sleep=False,
                )
                # update seed for the next interpolation step if IK succeeded
                if found and grasping_q is not None:
                    seed_grasping_q = grasping_q
                if not found or distanceToObstacle(self.robot, grasping_q) < 0.001:
                    # cannot progress past the previous valid step
                    if i - 1 == 0:
                        q_new_found = False
                        return q_nearest_vertex.q, q_nearest_vertex.grasping_q, False
                    return self.lerp(q_nearest_vertex.q, q_end, dt * (i - 1)), grasping_q, True
        return q_end, grasping_q, q_new_found

    def check_edge(self, q_latest_new: pin.SE3, q_latest_new_grasping_pose: Optional[np.ndarray],
                   q_goal: pin.SE3) -> bool:
        # Try to progress from q_latest_new to q_goal using get_q_new semantics
        dummy_vertex = Vertex(q=q_latest_new, grasping_q=q_latest_new_grasping_pose, parent=None)
        t_edge = time.perf_counter()
        q_new, q_new_grasping_q, q_new_found = self.get_q_new(
            q_nearest_vertex=dummy_vertex,
            q_rand=q_goal,
            max_delta_q=None,
        )
        dt_edge = time.perf_counter() - t_edge
        if not q_new_found:
            return False
        distance_to_target = self.calc_distance(q_new, q_goal)
        return distance_to_target < EDGE_DISTANCE_TOL

    def get_path_from_rrt(self) -> List[Vertex]:
        path: List[Vertex] = []
        last = self.nodes[-1]
        while last is not None:
            path.insert(0, last)
            last = last.parent
        return path

    def get_shortcut_from_rrt_path(self) -> List[Vertex]:
        path_vertices = self.get_path_from_rrt()
        if len(path_vertices) <= 2:
            return path_vertices

        short_path: List[Vertex] = []
        start_idx = 0
        while start_idx < len(path_vertices) - 1:
            current_vertex = path_vertices[start_idx]
            found_idx = None
            for next_idx in range(len(path_vertices) - 1, start_idx, -1):
                if self.check_edge(current_vertex.q, current_vertex.grasping_q, path_vertices[next_idx].q):
                    found_idx = next_idx
                    break
            if found_idx is None:
                # no shortcut found; step by one
                short_path.append(current_vertex)
                start_idx += 1
            else:
                short_path.append(current_vertex)
                start_idx = found_idx

        # always append the final vertex
        if path_vertices:
            short_path.append(path_vertices[-1])
        return short_path

    def run(self) -> Optional['RRT']:
        """Grow the RRT using the same algorithm formerly in construct_rrt.

        This is an instance method so callers can instantiate RRT(...) with
        desired runtime parameters and then run the search. Returns self on
        success, or None on timeout/failure.
        """
        # Metrics for logging
        total_samples = 0
        successful_extensions = 0
        start_time = time.perf_counter()

        for i in range(self.num_iter):
            # Abort the run if we've exceeded a user-specified wall-time budget
            if self.max_time_s is not None:
                elapsed_now = time.perf_counter() - start_time
                if elapsed_now > self.max_time_s:
                    print("Stopped because of time")
                    return None

            total_samples += 1
            random_cp_q = self.random_sampler()

            nearest_idx, nearest_vertex = self.get_nearest_vertex(random_cp_q)

            cp_q_new, cp_q_new_grasping_pose, cp_q_new_found = self.get_q_new(
                q_nearest_vertex=nearest_vertex,
                q_rand=random_cp_q,
                max_delta_q=self.max_delta_q,
            )

            if cp_q_new_found:
                self.add_edge(parent_idx=nearest_idx, new_q=cp_q_new, new_grasping_q=cp_q_new_grasping_pose)

                if self.check_edge(cp_q_new, cp_q_new_grasping_pose, self.cubeplacementqgoal):
                    self.add_edge(parent_idx=len(self.nodes) - 1, new_q=self.cubeplacementqgoal, new_grasping_q=self.q_goal)
                    return self

        return None
