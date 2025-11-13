import numpy as np
from config import DT
import pinocchio as pin
from pinocchio.utils import rotate
from inverse_geometry import computeqgrasppose
from bezier import Bezier

class RobotTrajectoryWrapper:
    def __init__(
        self,
        q_init,
        q_goal,
        T_max,
        T_min=0.0,
        dt: float = DT
    ):
        self.q_init = q_init
        self.q_goal = q_goal
        self.T_max = T_max
        self.T_min = T_min
        self.dt = dt

        self.q_t = {
            T_min: q_init,
            T_max: q_goal
        }

    def __call__(self, t):
        return self.get_closest_q(t)

    def set_q_t(self, q, t):
        self.q_t[t] = q

    def get_closest_q(self, t):
        if t in self.q_t:
            return self.q_t[t]
        else:
            return self.q_t[min(self.q_t.keys(), key=lambda x: abs(x-t))]

    def derivative(self, order, t):

        q_t  = self.get_closest_q(t)

        # choose neighbors with clamping
        tp = min(t + self.dt, self.T_max)
        tm = max(t - self.dt, self.T_min)
        q_tp = self.get_closest_q(tp)
        q_tm = self.get_closest_q(tm)

        if order == 1:
            if tp == t and tm == t:
                return np.zeros_like(q_t)
            if tp > t and tm < t:  # central diff
                return (q_tp - q_tm) / (tp - tm)
            elif tp > t:  # forward one-sided
                return (q_tp - q_t) / (tp - t)
            else:         # backward one-sided
                return (q_t - q_tm) / (t - tm)

        if order == 2:
            # use symmetric step h when possible
            h_p = max(tp - t, 0.0)
            h_m = max(t - tm, 0.0)
            if h_p > 0 and h_m > 0 and abs(h_p - h_m) < 1e-12:
                h = h_p  # equal steps
                return (q_tp - 2.0*q_t + q_tm) / (h*h)
            # fallback: nonuniform 3-point formula (approx)
            denom = 0.5*(h_p*h_m*(h_p + h_m)) + 1e-12
            a = 2.0 / denom
            return a * ( (q_tp * h_m) - (q_t * (h_p + h_m)) + (q_tm * h_p) )
        raise ValueError("Only derivative orders 1 or 2 supported.")


class CubeBezierTrajectory:
    def __init__(self,
                cube_placements,
                robot,
                cube,
                q_init,
                q_goal,
                T_max,
                T_min=0.0,
                dt: float = DT):

        self.cube_placements = cube_placements
        self.cube_placements_pos = [p.translation for p in cube_placements]

        self.q_of_t_cube = Bezier(self.cube_placements_pos, T_min, T_max)

        self.robot = robot
        self.cube = cube

        self.dt = dt
        self.T_min = T_min
        self.T_max = T_max

        self.robot_q_trajectory = RobotTrajectoryWrapper(
            q_init=q_init,
            q_goal=q_goal,
            T_max=T_max,
            T_min=T_min,
            dt=dt
        )

    def _register_robot_q_t(self, q, t):
        self.robot_q_trajectory.set_q_t(q, t)

    def __call__(self, t):
        cube_pos = self.q_of_t_cube(t)
        cube_placement = pin.SE3(rotate('z', 0.), cube_pos)

        q, success = computeqgrasppose(
            robot=self.robot,
            qcurrent=self.robot_q_trajectory.get_closest_q(t - self.dt),
            cube=self.cube,
            cubetarget=cube_placement)
        if not success:
            print(t)
            print(cube_placement)
            print(self.robot_q_trajectory.get_closest_q(t - self.dt * 2))
            print(q)
            raise ValueError("Failed to compute grasp pose")
        # print(f"t={t:.3f}, success={success}")
        self.robot_q_trajectory.set_q_t(q, t)
        return q

    def get_q_derivative(self, t, order):
        if t not in self.robot_q_trajectory.q_t:
            self.__call__(t)
        prev_t = max(self.T_min, t - self.dt)
        next_t = min(self.T_max, t + self.dt)
        if prev_t not in self.robot_q_trajectory.q_t:
            self.__call__(prev_t)
        if next_t not in self.robot_q_trajectory.q_t:
            self.__call__(next_t)
        return self.robot_q_trajectory.derivative(order, t)

    def derivative(self, order):
        return lambda t: self.get_q_derivative(t, order)


class CubeLinearTrajectory:
    def __init__(self,
                cube_placements,
                robot,
                cube,
                q_init,
                q_goal,
                T_max,
                T_min=0.0,
                dt: float = DT):

        self.cube_placements = cube_placements
        self.cube_placements_pos = [p.translation for p in cube_placements]

        self.robot = robot
        self.cube = cube

        self.T_max = T_max
        self.T_min = T_min

        self.dt = dt

        self.robot_q_trajectory = RobotTrajectoryWrapper(
            q_init=q_init,
            q_goal=q_goal,
            T_max=T_max,
            T_min=T_min,
            dt=dt
        )

        segments = np.diff(self.cube_placements_pos, axis=0)
        self.lengths = np.linalg.norm(segments, axis=1)
        self.cumulative_lengths = np.concatenate([[0], np.cumsum(self.lengths)])
        self.total_length = self.cumulative_lengths[-1]

        # constant speed
        self.v = self.total_length / T_max

    def _register_robot_q_t(self, q, t):
        self.robot_q_trajectory.set_q_t(q, t)

    def _get_cube_pos(self, t):
        # current arc length
        s = t * self.v
        s = np.clip(s, 0, self.total_length)

        # segment index of the current arc length
        segment_idx = np.searchsorted(self.cumulative_lengths, s, side='right') - 1
        segment_idx = min(segment_idx, len(self.lengths) - 1)

        # Local parameter within segment
        s_local = s - self.cumulative_lengths[segment_idx]
        alpha = s_local / self.lengths[segment_idx] if self.lengths[segment_idx] > 0 else 0

        # Interpolate
        p0 = self.cube_placements_pos[segment_idx]
        p1 = self.cube_placements_pos[segment_idx + 1]
        return (1 - alpha) * p0 + alpha * p1

    def __call__(self, t):
        cube_pos = self._get_cube_pos(t)
        cube_placement = pin.SE3(rotate('z', 0.), cube_pos)

        q, success = computeqgrasppose(
            robot=self.robot,
            qcurrent=self.robot_q_trajectory.get_closest_q(t-self.dt * 10),
            cube=self.cube,
            cubetarget=cube_placement)

        if not success:
            print(f'Failed to compute q for t={t}, cube_pos={cube_pos}, ')
        self.robot_q_trajectory.set_q_t(q, t)
        return q

    def get_q_derivative(self, t, order):
        if t not in self.robot_q_trajectory.q_t:
            self.__call__(t)
        prev_t = max(self.T_min, t - self.dt)
        next_t = min(self.T_max, t + self.dt)
        if prev_t not in self.robot_q_trajectory.q_t:
            self.__call__(prev_t)
        if next_t not in self.robot_q_trajectory.q_t:
            self.__call__(next_t)
        return self.robot_q_trajectory.derivative(order, t)

    def derivative(self, order):
        return lambda t: self.get_q_derivative(t, order)



class LinearJointTrajectory:
    """
    Piecewise-linear joint-space trajectory through given waypoints.
    Time is distributed proportional to joint-space segment length,
    with optional per-joint vmax / amax limiting.
    """
    def __init__(self, waypoints, T_max, qdot_max=None, amax=None, eps=1e-9):
        self.W = np.asarray(waypoints, dtype=float)  # (K, nq)
        # assert self.W.ndim == 2 and self.W.shape[0] >= 2, "Need at least 2 waypoints"
        self.nq = self.W.shape[1]
        self.T_max = float(T_max)
        self.eps = eps

        # Per-joint limits
        if qdot_max is None:
            qdot_max = np.full(self.nq, 2.0)  # rad/s default
        if amax is None:
            amax = np.full(self.nq, 5.0)      # rad/s^2 default
        self.qdot_max = np.asarray(qdot_max, dtype=float)
        self.amax = np.asarray(amax, dtype=float)

        # Segment lengths in joint space (∞-norm per joint -> max joint move dominates)
        dQ = np.diff(self.W, axis=0)                       # (K-1, nq)
        seg_norms = np.max(np.abs(dQ), axis=1)             # (K-1,)
        seg_norms = np.maximum(seg_norms, self.eps)
        total = float(np.sum(seg_norms))

        # First pass: proportional time split
        self.seg_times = (seg_norms / total) * self.T_max  # (K-1,)

        # Enforce vmax per joint: T_i >= max_j |dq_ij| / vmax_j
        Tmin_v = np.max(np.abs(dQ) / self.qdot_max, axis=1)
        self.seg_times = np.maximum(self.seg_times, Tmin_v + 1e-6)

        # Optional: very light amax check (not a full trapezoid here)
        # You can tighten with a true trapezoidal time-scaler if needed.

        # Prefix sums -> piecewise timing
        self.t_knots = np.concatenate(([0.0], np.cumsum(self.seg_times)))
        # Guard for round-off
        self.t_knots[-1] = self.T_max

        # Precompute segment slopes (constant velocity per segment)
        self.seg_slopes = np.zeros_like(dQ)
        for i, dt in enumerate(self.seg_times):
            self.seg_slopes[i] = dQ[i] / max(dt, self.eps)

    def _find_seg(self, t):
        if t <= 0.0: return 0, 0.0
        if t >= self.T_max: return len(self.seg_times) - 1, self.seg_times[-1]
        # locate segment index i s.t. t in [t_i, t_{i+1}]
        i = np.searchsorted(self.t_knots, t, side='right') - 1
        i = min(max(i, 0), len(self.seg_times) - 1)
        tau = t - self.t_knots[i]
        return i, tau

    def __call__(self, t):
        """q(t)"""
        t = float(np.clip(t, 0.0, self.T_max))
        i, tau = self._find_seg(t)
        return self.W[i] + self.seg_slopes[i] * tau

    def derivative(self, order):
        if order == 1:
            return lambda t: self.velocity(t)
        elif order == 2:
            return lambda t: self.acceleration(t)
        else:
            return lambda t: np.zeros(self.nq)

    def velocity(self, t):
        """qdot(t): piecewise constant"""
        t = float(np.clip(t, 0.0, self.T_max))
        i, _ = self._find_seg(t)
        return self.seg_slopes[i]

    def acceleration(self, t):
        """qddot(t): zero inside segments (impulses at knots not modeled)"""
        return np.zeros(self.nq)



class GlobalMinJerkLinearJTraj:
    """
    Follow a joint-space polyline with a GLOBAL min-jerk time law s(t) in [0, L].
    - q(t) continuous everywhere
    - |q̇(t)| ramps from ~0 at start to peak mid-way, back to 0 at end
    - q̇ direction still changes at vertices (polyline corners)
    """
    def __init__(self, waypoints, T_max, eps=1e-9):
        W = np.asarray(waypoints, float)
        assert W.ndim == 2 and W.shape[0] >= 2
        self.W = W
        self.nq = W.shape[1]
        self.T = float(T_max)
        self.eps = eps

        # Segment "lengths" in joint space (use ∞-norm so one joint dominates timing)
        dQ = np.diff(W, axis=0)                    # (S, nq)
        seg_L = np.max(np.abs(dQ), axis=1)         # (S,)
        seg_L = np.maximum(seg_L, eps)
        self.dQ = dQ
        self.seg_L = seg_L
        self.cum_L = np.concatenate(([0.0], np.cumsum(seg_L)))
        self.Ltot = float(self.cum_L[-1])

        # Precompute segment directions (unit in ∞-norm sense)
        self.seg_dir = np.divide(dQ, seg_L[:, None], where=seg_L[:, None] > 0)

    # ---- Min-jerk scalar profile s(t) (and derivatives) on [0,T] ----
    # τ in [0,1]; s = L*(10τ^3 - 15τ^4 + 6τ^5)
    def _s(self, t):
        tau = np.clip(t / self.T, 0.0, 1.0)
        tau2, tau3, tau4, tau5 = tau*tau, tau*tau*tau, None, None
        tau4 = tau2*tau2
        tau5 = tau3*tau2
        s = self.Ltot * (10*tau3 - 15*tau4 + 6*tau5)
        sd = self.Ltot * (30*tau*tau - 60*tau2*tau + 30*tau4) / self.T
        sdd = self.Ltot * (60*tau - 180*tau2 + 120*tau3) / (self.T**2)
        return s, sd, sdd

    def _locate_segment(self, s):
        # find i s.t. s in [cum_L[i], cum_L[i+1]]
        i = np.searchsorted(self.cum_L, s, side='right') - 1
        i = int(np.clip(i, 0, len(self.seg_L)-1))
        s_local = s - self.cum_L[i]
        alpha = s_local / max(self.seg_L[i], self.eps)  # in [0,1]
        return i, alpha

    def __call__(self, t):
        s, sd, sdd = self._s(t)
        # Clamp end
        if s >= self.Ltot - 1e-12:
            return self.W[-1]
        i, alpha = self._locate_segment(s)
        return self.W[i] + alpha * self.dQ[i]

    def derivative(self, order):
        if order == 1:
            return lambda t: self.velocity(t)
        if order == 2:
            return lambda t: self.acceleration(t)
        return lambda t: np.zeros(self.nq)

    def velocity(self, t):
        s, sd, sdd = self._s(t)
        if s >= self.Ltot - 1e-12 or sd <= 0:
            return np.zeros(self.nq)
        i, _ = self._locate_segment(s)
        # q̇ = (dq/ds) * ṡ ; along a linear segment, dq/ds = seg_dir[i]
        return self.seg_dir[i] * sd

    def acceleration(self, t):
        s, sd, sdd = self._s(t)
        if s >= self.Ltot - 1e-12:
            return np.zeros(self.nq)
        i, _ = self._locate_segment(s)
        # q̈ = (dq/ds) * s̈  +  (d/ds seg_dir)*ṡ^2 ; second term is zero inside segment
        # (impulses at corners not modeled explicitly)
        return self.seg_dir[i] * sdd



class LinearPathWithTrapezoidalVel:
    """
    Piecewise linear path with smooth velocity profile that slows at corners
    """
    def __init__(self, waypoints, T_max, corner_slowdown=0.1):
        W = np.asarray(waypoints, float)
        self.W = W
        self.nq = W.shape[1]
        self.T = T_max

        # Segment lengths
        dQ = np.diff(W, axis=0)
        seg_L = np.linalg.norm(dQ, axis=1)
        self.seg_L = seg_L
        self.cum_L = np.concatenate(([0.0], np.cumsum(seg_L)))
        self.Ltot = self.cum_L[-1]

        # Direction vectors
        self.seg_dir = dQ / (seg_L[:, None] + 1e-9)

        # Define "corner zones" - last X% of each segment + first X% of next
        self.corner_slowdown = corner_slowdown
        self.slow_distance = corner_slowdown * np.min(seg_L[seg_L > 0]) if len(seg_L) > 0 else 0.1

    def _s(self, t):
        """Min-jerk profile on [0, T] but modified near corners"""
        tau = np.clip(t / self.T, 0.0, 1.0)
        tau2, tau3 = tau*tau, tau*tau*tau
        tau4, tau5 = tau2*tau2, tau3*tau2

        # Base min-jerk
        s_base = self.Ltot * (10*tau3 - 15*tau4 + 6*tau5)
        sd_base = self.Ltot * (30*tau2 - 60*tau3 + 30*tau4) / self.T
        sdd_base = self.Ltot * (60*tau - 180*tau2 + 120*tau3) / (self.T**2)

        # Slow down factor based on proximity to corners
        # slowdown = self._corner_slowdown_factor(s_base)
        slowdown = 1.0

        return s_base, sd_base * slowdown, sdd_base * slowdown

    def _corner_slowdown_factor(self, s):
        """Returns value in (0, 1] that reduces speed near corners"""
        # Find which segment we're in
        i = np.searchsorted(self.cum_L, s, side='right') - 1
        i = int(np.clip(i, 0, len(self.seg_L)-1))

        # Distance from start and end of segment
        s_in_seg = s - self.cum_L[i]
        dist_to_end = self.seg_L[i] - s_in_seg

        # Slow down if near end of segment (approaching corner)
        if dist_to_end < self.slow_distance:
            factor = dist_to_end / self.slow_distance
            return 0.2 + 0.8 * factor  # Reduce to 20% speed at corner

        # Slow down if near start of segment (just left corner)
        if s_in_seg < self.slow_distance and i > 0:
            factor = s_in_seg / self.slow_distance
            return 0.2 + 0.8 * factor

        return 1.0  # Full speed in middle of segments

    def _locate_segment(self, s):
        i = np.searchsorted(self.cum_L, s, side='right') - 1
        i = int(np.clip(i, 0, len(self.seg_L)-1))
        s_local = s - self.cum_L[i]
        alpha = s_local / max(self.seg_L[i], 1e-9)
        return i, alpha

    def __call__(self, t):
        s, _, _ = self._s(t)
        if s >= self.Ltot - 1e-12:
            return self.W[-1]
        i, alpha = self._locate_segment(s)
        return self.W[i] + alpha * (self.W[i+1] - self.W[i])

    def velocity(self, t):
        s, sd, _ = self._s(t)
        if s >= self.Ltot - 1e-12 or sd <= 0:
            return np.zeros(self.nq)
        i, _ = self._locate_segment(s)
        return self.seg_dir[i] * sd

    def acceleration(self, t):
        s, sd, sdd = self._s(t)
        if s >= self.Ltot - 1e-12:
            return np.zeros(self.nq)
        i, _ = self._locate_segment(s)
        return self.seg_dir[i] * sdd

    def derivative(self, order):
        if order == 1:
            return self.velocity
        elif order == 2:
            return self.acceleration
        return lambda t: np.zeros(self.nq)
