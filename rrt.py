#!/usr/bin/env python3
"""Standard RRT implementation for cube placement planning.

This module contains a focused RRT implementation that works with cube
placements (pin.SE3) and optional grasping-pose callbacks. The implementation
is intentionally minimal and readable.
"""
from typing import Callable, Tuple, List, Optional

import logging
import numpy as np
import pinocchio as pin
from pinocchio.utils import rotate

from constants import EDGE_DISTANCE_TOL, DEFAULT_DISCRETISATION_STEPS, DEFAULT_CHECK_EDGE_STEPS
from tools import setcubeplacement, distanceToObstacle
from constants import DEFAULT_VIZ_DELAY
import time
import sampling

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
                 max_time_s: Optional[float] = None,
                 progress_log_every: int = 50,
                 repair_sampler: Optional[Callable] = None,
                 repair_max_attempts: int = 5,
                 goal_bias: float = 0.02):

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
        self.progress_log_every = progress_log_every
        self.repair_sampler = repair_sampler
        self.repair_max_attempts = repair_max_attempts
        self.goal_bias = goal_bias
        # profiling counters for IK, viz and edge checks
        self._profile = {
            'ik_calls': 0,
            'time_ik': 0.0,
            'viz_calls': 0,
            'time_viz': 0.0,
            'edge_checks': 0,
            'time_edge_checks': 0.0,
        }

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
                  discretisation_steps: Optional[int] = None,
                  max_delta_q: Optional[float] = None) -> Tuple[pin.SE3, Optional[np.ndarray], bool]:

        q_new_found = True
        q_end = q_rand

        dist = self.calc_distance(q_nearest_vertex.q, q_rand)
        if max_delta_q is not None and dist > max_delta_q:
            q_end = self.lerp(q_nearest_vertex.q, q_rand, max_delta_q / dist)

        grasping_q = None
        if self.get_grasping_poseq is not None:
            # allow caller to override discretisation for this call; otherwise use instance default
            discretisation_steps = discretisation_steps if discretisation_steps is not None else self.discretisation_steps
            dt = 1.0 / discretisation_steps
            # seed IK with the nearest vertex grasping pose and then update the seed
            seed_grasping_q = q_nearest_vertex.grasping_q
            for i in range(1, discretisation_steps):
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
                    self._profile['viz_calls'] += 1
                    self._profile['time_viz'] += dt_v
                    time.sleep(self.viz_delay)

                # Use the IK wrapper to call the IK callback in a
                # backwards-compatible way and time the call. We set
                # viz_sleep=False so display calls (if any) don't sleep.
                from ik_utils import call_ik
                ik_kwargs = dict(
                    robot=self.robot,
                    qcurrent=seed_grasping_q,
                    cube=self.cube,
                    cubetarget=q,
                    viz=(self.viz if self.viz is not None else None),
                    max_attempts=(self.max_ik_attempts if self.max_ik_attempts is not None else 1000),
                    viz_sleep=False,
                )
                (grasping_q, found), dt_ik = call_ik(self.get_grasping_poseq, **ik_kwargs)
                self._profile['ik_calls'] += 1
                self._profile['time_ik'] += dt_ik
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
                   q_goal: pin.SE3, discretisation_steps: Optional[int] = None) -> bool:
        # Try to progress from q_latest_new to q_goal using get_q_new semantics
        dummy_vertex = Vertex(q=q_latest_new, grasping_q=q_latest_new_grasping_pose, parent=None)
        t_edge = time.perf_counter()
        q_new, q_new_grasping_q, q_new_found = self.get_q_new(
            q_nearest_vertex=dummy_vertex,
            q_rand=q_goal,
            discretisation_steps=discretisation_steps,
            max_delta_q=None,
        )
        dt_edge = time.perf_counter() - t_edge
        self._profile['edge_checks'] += 1
        self._profile['time_edge_checks'] += dt_edge
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

    def get_shortcut_from_rrt_path(self, discretisation_steps: int = DEFAULT_DISCRETISATION_STEPS,
                                   max_ik_attempts: Optional[int] = None) -> List[Vertex]:
        path_vertices = self.get_path_from_rrt()
        if len(path_vertices) <= 2:
            return path_vertices

        short_path: List[Vertex] = []
        start_idx = 0
        while start_idx < len(path_vertices) - 1:
            current_vertex = path_vertices[start_idx]
            # Try to find the furthest vertex we can directly connect to
            found_idx = None
            for next_idx in range(len(path_vertices) - 1, start_idx, -1):
                if self.check_edge(current_vertex.q, current_vertex.grasping_q, path_vertices[next_idx].q,
                                   discretisation_steps=discretisation_steps):
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

        # reset sampling module metrics for this run
        try:
            sampling.metrics = {
                'samples': 0,
                'geom_updates': 0,
                'collision_checks': 0,
                'time_collision_checks': 0.0,
                'time_viz': 0.0,
                'time_sampling_loop': 0.0,
            }
        except Exception:
            pass

        for i in range(self.num_iter):
            # Periodic progress logging so long runs show activity
            if self.progress_log_every and i % self.progress_log_every == 0 and i > 0:
                elapsed_now = time.perf_counter() - start_time
                samples_per_s_now = total_samples / elapsed_now if elapsed_now > 0 else float('inf')
                logger.info("RRT progress: %d/%d iterations, samples: %d, elapsed: %.2fs, samples/s: %.2f",
                            i, self.num_iter, total_samples, elapsed_now, samples_per_s_now)

            # Abort the run if we've exceeded a user-specified wall-time budget
            if self.max_time_s is not None:
                elapsed_now = time.perf_counter() - start_time
                if elapsed_now > self.max_time_s:
                    logger.warning("RRT aborted after %.2fs (max_time_s=%.2fs) — returning no path", elapsed_now, self.max_time_s)
                    # include profiling into self.metrics before returning
                    try:
                        self.metrics = {
                            'total_samples': total_samples,
                            'successful_extensions': successful_extensions,
                            'iterations': i,
                            'time_s': elapsed_now,
                            'samples_per_s': total_samples / elapsed_now if elapsed_now > 0 else float('inf'),
                            'success_rate': successful_extensions / total_samples if total_samples > 0 else 0.0,
                        }
                        self.metrics['profiling'] = self._profile.copy()
                        self.metrics['sampling'] = sampling.metrics.copy()
                    except Exception:
                        pass
                    return None

            # With small probability sample the goal directly (goal bias)
            import random as _random
            if self.goal_bias and _random.random() < self.goal_bias:
                random_cp_q = self.cubeplacementqgoal
            else:
                total_samples += 1
                random_cp_q = self.random_sampler()

            nearest_idx, nearest_vertex = self.get_nearest_vertex(random_cp_q)

            cp_q_new, cp_q_new_grasping_pose, cp_q_new_found = self.get_q_new(
                q_nearest_vertex=nearest_vertex,
                q_rand=random_cp_q,
                discretisation_steps=(self.discretisation_steps if self.discretisation_steps is not None else self.discretisation_steps),
                max_delta_q=self.max_delta_q,
            )

            # If the extension succeeded use it. Otherwise, optionally try a
            # local repair sampler (smart sampler) a few times before continuing.
            if not cp_q_new_found and self.repair_sampler is not None:
                for r_attempt in range(self.repair_max_attempts):
                    total_samples += 1
                    try:
                        repair_target = self.repair_sampler()
                    except Exception:
                        # if the repair sampler misbehaves, stop trying repairs
                        break
                    nearest_idx, nearest_vertex = self.get_nearest_vertex(repair_target)
                    cp_q_new, cp_q_new_grasping_pose, cp_q_new_found = self.get_q_new(
                        q_nearest_vertex=nearest_vertex,
                        q_rand=repair_target,
                        discretisation_steps=(self.discretisation_steps if self.discretisation_steps is not None else self.discretisation_steps),
                        max_delta_q=self.max_delta_q,
                    )
                    if cp_q_new_found:
                        successful_extensions += 1
                        self.add_edge(parent_idx=nearest_idx, new_q=cp_q_new, new_grasping_q=cp_q_new_grasping_pose)
                        # check connection to goal and return if found
                        if self.check_edge(cp_q_new, cp_q_new_grasping_pose, self.cubeplacementqgoal,
                                           discretisation_steps=(self.discretisation_steps if self.discretisation_steps is not None else self.discretisation_steps)):
                            self.add_edge(parent_idx=len(self.nodes) - 1, new_q=self.cubeplacementqgoal, new_grasping_q=self.q_goal)
                            elapsed = time.perf_counter() - start_time
                            # attach metrics to self for later inspection
                            self.metrics = {
                                'total_samples': total_samples,
                                'successful_extensions': successful_extensions,
                                'iterations': i + 1,
                                'time_s': elapsed,
                                'samples_per_s': total_samples / elapsed if elapsed > 0 else float('inf'),
                                'success_rate': successful_extensions / total_samples if total_samples > 0 else 0.0,
                            }
                            try:
                                self.metrics['profiling'] = self._profile.copy()
                            except Exception:
                                self.metrics['profiling'] = {}
                            try:
                                self.metrics['sampling'] = sampling.metrics.copy()
                            except Exception:
                                self.metrics['sampling'] = {}
                            logger.info("RRT found a path after %d iterations (%.3fs)", i + 1, elapsed)
                            logger.info("Samples: %d, successful extensions: %d, success rate: %.3f, samples/s: %.1f",
                                        total_samples, successful_extensions, self.metrics['success_rate'], self.metrics['samples_per_s'])
                            return self
                        break

            if cp_q_new_found:
                successful_extensions += 1
                self.add_edge(parent_idx=nearest_idx, new_q=cp_q_new, new_grasping_q=cp_q_new_grasping_pose)

                if self.check_edge(cp_q_new, cp_q_new_grasping_pose, self.cubeplacementqgoal, discretisation_steps=(self.discretisation_steps if self.discretisation_steps is not None else self.discretisation_steps)):
                    self.add_edge(parent_idx=len(self.nodes) - 1, new_q=self.cubeplacementqgoal, new_grasping_q=self.q_goal)
                    elapsed = time.perf_counter() - start_time
                    # attach metrics to self for later inspection
                    self.metrics = {
                        'total_samples': total_samples,
                        'successful_extensions': successful_extensions,
                        'iterations': i + 1,
                        'time_s': elapsed,
                        'samples_per_s': total_samples / elapsed if elapsed > 0 else float('inf'),
                        'success_rate': successful_extensions / total_samples if total_samples > 0 else 0.0,
                    }
                    # attach profiling collected by the RRT and sampling modules
                    try:
                        self.metrics['profiling'] = self._profile.copy()
                    except Exception:
                        self.metrics['profiling'] = {}
                    try:
                        self.metrics['sampling'] = sampling.metrics.copy()
                    except Exception:
                        self.metrics['sampling'] = {}
                    logger.info("RRT found a path after %d iterations (%.3fs)", i + 1, elapsed)
                    logger.info("Samples: %d, successful extensions: %d, success rate: %.3f, samples/s: %.1f",
                                total_samples, successful_extensions, self.metrics['success_rate'], self.metrics['samples_per_s'])
                    return self

        elapsed = time.perf_counter() - start_time
        # include profiling even on failure
        try:
            self.metrics = {
                'total_samples': total_samples,
                'successful_extensions': successful_extensions,
                'iterations': self.num_iter,
                'time_s': elapsed,
                'samples_per_s': total_samples / elapsed if elapsed > 0 else float('inf'),
                'success_rate': successful_extensions / total_samples if total_samples > 0 else 0.0,
            }
            self.metrics['profiling'] = self._profile.copy()
            self.metrics['sampling'] = sampling.metrics.copy()
        except Exception:
            pass
        logger.info("RRT completed %d iterations without connecting to goal (%.3fs)", self.num_iter, elapsed)
        logger.info("Samples: %d, successful extensions: %d, success rate: %.3f, samples/s: %.1f",
                    total_samples, successful_extensions, (successful_extensions / total_samples) if total_samples > 0 else 0.0,
                    total_samples / elapsed if elapsed > 0 else float('inf'))
        return None


def construct_rrt(robot: pin.robot_wrapper.RobotWrapper,
                  cube: pin.robot_wrapper.RobotWrapper,
                  q_init: np.ndarray,
                  q_goal: np.ndarray,
                  cubeplacementq0: pin.SE3,
                  cubeplacementqgoal: pin.SE3,
                  num_iter: int,
                  random_sampler: Callable,
                  max_delta_q: Optional[float] = None,
                  discretisation_steps: int = DEFAULT_DISCRETISATION_STEPS,
                  get_grasping_poseq: Optional[Callable] = None,
                  viz=None,
                  viz_delay: float = DEFAULT_VIZ_DELAY,
                  max_ik_attempts: Optional[int] = None,
                  max_time_s: Optional[float] = None,
                  progress_log_every: int = 50,
                  repair_sampler: Optional[Callable] = None,
                  repair_max_attempts: int = 5,
                  goal_bias: float = 0.02) -> Optional[RRT]:
    """Construct a simple RRT that samples cube placements using `random_sampler`.

    This is intentionally straightforward: we sample, extend toward the sample,
    and stop when we can connect to the goal.
    """
    # instantiate with run-time parameters and delegate to instance.run()
    rrt = RRT(root_q=cubeplacementq0,
              root_grasping_q=q_init,
              robot=robot,
              cube=cube,
              get_grasping_poseq=get_grasping_poseq,
              viz=viz,
              viz_delay=viz_delay,
              discretisation_steps=discretisation_steps,
              max_ik_attempts=max_ik_attempts,
              num_iter=num_iter,
              random_sampler=random_sampler,
              max_delta_q=max_delta_q,
              cubeplacementqgoal=cubeplacementqgoal,
              q_goal=q_goal,
              max_time_s=max_time_s,
              progress_log_every=progress_log_every,
              repair_sampler=repair_sampler,
              repair_max_attempts=repair_max_attempts,
              goal_bias=goal_bias)

    return rrt.run()
