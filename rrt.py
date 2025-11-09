#!/usr/bin/env python3
"""Standard RRT implementation for cube placement planning.

This module contains a focused RRT implementation that works with cube
placements (pin.SE3) and optional grasping-pose callbacks. The implementation
is intentionally minimal and readable.
"""
from typing import Callable, Tuple, List, Optional

import logging
import numpy as np
                 cube: pin.robot_wrapper.RobotWrapper,
                 get_grasping_poseq: Optional[Callable] = None,
                 viz=None,
                 viz_delay: float = DEFAULT_VIZ_DELAY,
                 discretisation_steps: int = DEFAULT_DISCRETISATION_STEPS,
                 max_ik_attempts: Optional[int] = None):

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
        # profiling counters for IK, viz and edge checks
        self._profile = {
            'ik_calls': 0,
            'time_ik': 0.0,
            'viz_calls': 0,
            'time_viz': 0.0,
            'edge_checks': 0,
            'time_edge_checks': 0.0,
        }
        # runtime parameters that control a run; populated by callers before
        # invoking `run()` so the instance method can operate without args.
        self.random_sampler = None
        self.num_iter = None
        self.max_delta_q = None
        self.max_time_s = None
        self.progress_log_every = 50
        self.repair_sampler = None
        self.repair_max_attempts = 5
        self.goal_bias = 0.02
        # goal / initial placements and grasping configs
        self.q_init = None
        self.q_goal = None
        self.cubeplacementqgoal = None
                                                    total_samples, successful_extensions, self.metrics['success_rate'], self.metrics['samples_per_s'])
                                        return self
                                    break

                        if cp_q_new_found:
                            successful_extensions += 1
                            self.add_edge(parent_idx=nearest_idx, new_q=cp_q_new, new_grasping_q=cp_q_new_grasping_pose)

                            if self.check_edge(cp_q_new, cp_q_new_grasping_pose, cubeplacementqgoal, discretisation_steps=discretisation_steps):
                                self.add_edge(parent_idx=len(self.nodes) - 1, new_q=cubeplacementqgoal, new_grasping_q=q_goal)
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
                            'iterations': num_iter,
                            'time_s': elapsed,
                            'samples_per_s': total_samples / elapsed if elapsed > 0 else float('inf'),
                            'success_rate': successful_extensions / total_samples if total_samples > 0 else 0.0,
                        }
                        self.metrics['profiling'] = self._profile.copy()
                        self.metrics['sampling'] = sampling.metrics.copy()
                    except Exception:
                        pass
                    logger.info("RRT completed %d iterations without connecting to goal (%.3fs)", num_iter, elapsed)
                    logger.info("Samples: %d, successful extensions: %d, success rate: %.3f, samples/s: %.1f",
                                total_samples, successful_extensions, (successful_extensions / total_samples) if total_samples > 0 else 0.0,
                                total_samples / elapsed if elapsed > 0 else float('inf'))
                    return None
        # always append the final vertex
        if path_vertices:
            short_path.append(path_vertices[-1])
        return short_path

    def run(self,
            q_init: np.ndarray,
            q_goal: np.ndarray,
            cubeplacementqgoal: pin.SE3,
            num_iter: int,
            random_sampler: Callable,
            max_delta_q: Optional[float] = None,
            discretisation_steps: int = DEFAULT_DISCRETISATION_STEPS,
            get_grasping_poseq: Optional[Callable] = None,
            max_time_s: Optional[float] = None,
            progress_log_every: int = 50,
            repair_sampler: Optional[Callable] = None,
            repair_max_attempts: int = 5,
            goal_bias: float = 0.02) -> Optional['RRT']:
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

        for i in range(num_iter):
            # Periodic progress logging so long runs show activity
            if progress_log_every and i % progress_log_every == 0 and i > 0:
                elapsed_now = time.perf_counter() - start_time
                samples_per_s_now = total_samples / elapsed_now if elapsed_now > 0 else float('inf')
                logger.info("RRT progress: %d/%d iterations, samples: %d, elapsed: %.2fs, samples/s: %.2f",
                            i, num_iter, total_samples, elapsed_now, samples_per_s_now)

            # Abort the run if we've exceeded a user-specified wall-time budget
            if max_time_s is not None:
                elapsed_now = time.perf_counter() - start_time
                if elapsed_now > max_time_s:
                    logger.warning("RRT aborted after %.2fs (max_time_s=%.2fs) — returning no path", elapsed_now, max_time_s)
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
            if goal_bias and _random.random() < goal_bias:
                random_cp_q = cubeplacementqgoal
            else:
                total_samples += 1
                random_cp_q = random_sampler()

            nearest_idx, nearest_vertex = self.get_nearest_vertex(random_cp_q)
                def run(self) -> Optional['RRT']:
                        repair_target = repair_sampler()
                    except Exception:
                        # if the repair sampler misbehaves, stop trying repairs
                        break
                    nearest_idx, nearest_vertex = self.get_nearest_vertex(repair_target)
                    cp_q_new, cp_q_new_grasping_pose, cp_q_new_found = self.get_q_new(
                        q_nearest_vertex=nearest_vertex,
                        q_rand=repair_target,
                        discretisation_steps=(discretisation_steps if discretisation_steps is not None else self.discretisation_steps),
                        max_delta_q=max_delta_q,
                    )
                    if cp_q_new_found:
                        successful_extensions += 1
                        self.add_edge(parent_idx=nearest_idx, new_q=cp_q_new, new_grasping_q=cp_q_new_grasping_pose)
                        # check connection to goal and return if found
                        if self.check_edge(cp_q_new, cp_q_new_grasping_pose, cubeplacementqgoal,
                                           discretisation_steps=(discretisation_steps if discretisation_steps is not None else self.discretisation_steps)):
                            self.add_edge(parent_idx=len(self.nodes) - 1, new_q=cubeplacementqgoal, new_grasping_q=q_goal)
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

                if self.check_edge(cp_q_new, cp_q_new_grasping_pose, cubeplacementqgoal, discretisation_steps=(discretisation_steps if discretisation_steps is not None else self.discretisation_steps)):
                    self.add_edge(parent_idx=len(self.nodes) - 1, new_q=cubeplacementqgoal, new_grasping_q=q_goal)
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
                'iterations': num_iter,
                'time_s': elapsed,
                'samples_per_s': total_samples / elapsed if elapsed > 0 else float('inf'),
                'success_rate': successful_extensions / total_samples if total_samples > 0 else 0.0,
            }
            self.metrics['profiling'] = self._profile.copy()
            self.metrics['sampling'] = sampling.metrics.copy()
        except Exception:
            pass
        logger.info("RRT completed %d iterations without connecting to goal (%.3fs)", num_iter, elapsed)
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
    rrt = RRT(root_q=cubeplacementq0,
              root_grasping_q=q_init,
              robot=robot,
              cube=cube,
              get_grasping_poseq=get_grasping_poseq,
              viz=viz,
              viz_delay=viz_delay,
              discretisation_steps=discretisation_steps,
              max_ik_attempts=max_ik_attempts)

    # Delegate the main loop to the RRT instance; keep this wrapper for
    # backwards compatibility.
    return rrt.run(q_init=q_init,
                   q_goal=q_goal,
                   cubeplacementqgoal=cubeplacementqgoal,
                   num_iter=num_iter,
                   random_sampler=random_sampler,
                   max_delta_q=max_delta_q,
                   discretisation_steps=discretisation_steps,
                   get_grasping_poseq=get_grasping_poseq,
                   max_time_s=max_time_s,
                   progress_log_every=progress_log_every,
                   repair_sampler=repair_sampler,
                   repair_max_attempts=repair_max_attempts,
                   goal_bias=goal_bias)

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

    for i in range(num_iter):
        # Periodic progress logging so long runs show activity
        if progress_log_every and i % progress_log_every == 0 and i > 0:
            elapsed_now = time.perf_counter() - start_time
            samples_per_s_now = total_samples / elapsed_now if elapsed_now > 0 else float('inf')
            logger.info("RRT progress: %d/%d iterations, samples: %d, elapsed: %.2fs, samples/s: %.2f",
                        i, num_iter, total_samples, elapsed_now, samples_per_s_now)

        # Abort the run if we've exceeded a user-specified wall-time budget
        if max_time_s is not None:
            elapsed_now = time.perf_counter() - start_time
            if elapsed_now > max_time_s:
                logger.warning("RRT aborted after %.2fs (max_time_s=%.2fs) — returning no path", elapsed_now, max_time_s)
                # include profiling into rrt.metrics before returning
                try:
                    rrt.metrics = {
                        'total_samples': total_samples,
                        'successful_extensions': successful_extensions,
                        'iterations': i,
                        'time_s': elapsed_now,
                        'samples_per_s': total_samples / elapsed_now if elapsed_now > 0 else float('inf'),
                        'success_rate': successful_extensions / total_samples if total_samples > 0 else 0.0,
                    }
                    rrt.metrics['profiling'] = rrt._profile.copy()
                    rrt.metrics['sampling'] = sampling.metrics.copy()
                except Exception:
                    pass
                return None
        # With small probability sample the goal directly (goal bias) to
        # encourage connection attempts.
        import random as _random
        if goal_bias and _random.random() < goal_bias:
            random_cp_q = cubeplacementqgoal
        else:
            total_samples += 1
            random_cp_q = random_sampler()
        nearest_idx, nearest_vertex = rrt.get_nearest_vertex(random_cp_q)

        cp_q_new, cp_q_new_grasping_pose, cp_q_new_found = rrt.get_q_new(
            q_nearest_vertex=nearest_vertex,
            q_rand=random_cp_q,
            discretisation_steps=discretisation_steps,
            max_delta_q=max_delta_q,
        )

        # If the extension succeeded use it. Otherwise, optionally try a
        # local repair sampler (smart sampler) a few times before continuing.
        if not cp_q_new_found and repair_sampler is not None:
            for r_attempt in range(repair_max_attempts):
                total_samples += 1
                try:
                    repair_target = repair_sampler()
                except Exception:
                    # if the repair sampler misbehaves, stop trying repairs
                    break
                nearest_idx, nearest_vertex = rrt.get_nearest_vertex(repair_target)
                cp_q_new, cp_q_new_grasping_pose, cp_q_new_found = rrt.get_q_new(
                    q_nearest_vertex=nearest_vertex,
                    q_rand=repair_target,
                    discretisation_steps=discretisation_steps,
                    max_delta_q=max_delta_q,
                )
                if cp_q_new_found:
                    successful_extensions += 1
                    rrt.add_edge(parent_idx=nearest_idx, new_q=cp_q_new, new_grasping_q=cp_q_new_grasping_pose)
                    # check connection to goal and return if found
                    if rrt.check_edge(cp_q_new, cp_q_new_grasping_pose, cubeplacementqgoal,
                                      discretisation_steps=discretisation_steps):
                        rrt.add_edge(parent_idx=len(rrt.nodes) - 1, new_q=cubeplacementqgoal, new_grasping_q=q_goal)
                        elapsed = time.perf_counter() - start_time
                        # attach metrics to rrt for later inspection
                        rrt.metrics = {
                            'total_samples': total_samples,
                            'successful_extensions': successful_extensions,
                            'iterations': i + 1,
                            'time_s': elapsed,
                            'samples_per_s': total_samples / elapsed if elapsed > 0 else float('inf'),
                            'success_rate': successful_extensions / total_samples if total_samples > 0 else 0.0,
                        }
                        try:
                            rrt.metrics['profiling'] = rrt._profile.copy()
                        except Exception:
                            rrt.metrics['profiling'] = {}
                        try:
                            rrt.metrics['sampling'] = sampling.metrics.copy()
                        except Exception:
                            rrt.metrics['sampling'] = {}
                        logger.info("RRT found a path after %d iterations (%.3fs)", i + 1, elapsed)
                        logger.info("Samples: %d, successful extensions: %d, success rate: %.3f, samples/s: %.1f",
                                    total_samples, successful_extensions, rrt.metrics['success_rate'], rrt.metrics['samples_per_s'])
                        return rrt
                    break

        if cp_q_new_found:
            successful_extensions += 1
            rrt.add_edge(parent_idx=nearest_idx, new_q=cp_q_new, new_grasping_q=cp_q_new_grasping_pose)

            if rrt.check_edge(cp_q_new, cp_q_new_grasping_pose, cubeplacementqgoal, discretisation_steps=discretisation_steps):
                rrt.add_edge(parent_idx=len(rrt.nodes) - 1, new_q=cubeplacementqgoal, new_grasping_q=q_goal)
                elapsed = time.perf_counter() - start_time
                # attach metrics to rrt for later inspection
                rrt.metrics = {
                    'total_samples': total_samples,
                    'successful_extensions': successful_extensions,
                    'iterations': i + 1,
                    'time_s': elapsed,
                    'samples_per_s': total_samples / elapsed if elapsed > 0 else float('inf'),
                    'success_rate': successful_extensions / total_samples if total_samples > 0 else 0.0,
                }
                # attach profiling collected by the RRT and sampling modules
                try:
                    rrt.metrics['profiling'] = rrt._profile.copy()
                except Exception:
                    rrt.metrics['profiling'] = {}
                try:
                    rrt.metrics['sampling'] = sampling.metrics.copy()
                except Exception:
                    rrt.metrics['sampling'] = {}
                logger.info("RRT found a path after %d iterations (%.3fs)", i + 1, elapsed)
                logger.info("Samples: %d, successful extensions: %d, success rate: %.3f, samples/s: %.1f",
                            total_samples, successful_extensions, rrt.metrics['success_rate'], rrt.metrics['samples_per_s'])
                return rrt

    elapsed = time.perf_counter() - start_time
    # include profiling even on failure
    try:
        rrt.metrics = {
            'total_samples': total_samples,
            'successful_extensions': successful_extensions,
            'iterations': num_iter,
            'time_s': elapsed,
            'samples_per_s': total_samples / elapsed if elapsed > 0 else float('inf'),
            'success_rate': successful_extensions / total_samples if total_samples > 0 else 0.0,
        }
        rrt.metrics['profiling'] = rrt._profile.copy()
        rrt.metrics['sampling'] = sampling.metrics.copy()
    except Exception:
        pass
    logger.info("RRT completed %d iterations without connecting to goal (%.3fs)", num_iter, elapsed)
    logger.info("Samples: %d, successful extensions: %d, success rate: %.3f, samples/s: %.1f",
                total_samples, successful_extensions, (successful_extensions / total_samples) if total_samples > 0 else 0.0,
                total_samples / elapsed if elapsed > 0 else float('inf'))
    return None
