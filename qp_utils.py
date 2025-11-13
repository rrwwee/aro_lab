import numpy as np
import quadprog
from typing import List, Dict, Tuple

def assign_time_to_intermediate_points(path: List[np.array]) -> Dict[int, float]:
    # assign time to each point in the path
    # so we can use it in QP cost function
    # compute time proportional to distance between points

    total_dist = 0
    point_idx_time = {}

    for i in range(1, len(path)-1):
        dist = np.linalg.norm(path[i] - path[i-1])
        total_dist += dist
        point_idx_time[i] = dist

    total_dist += np.linalg.norm(path[-1] - path[-2])
    norm_dist = np.cumsum(list(point_idx_time.values()) / total_dist)
    for point_idx, time in point_idx_time.items():
        point_idx_time[point_idx] = norm_dist[point_idx-1]

    return point_idx_time


def quadprog_solve_qp(H, q, G=None, h=None, C=None, d=None, verbose=False):
    """
    min (1/2)x' P x + q' x
    subject to  G x <= h
    subject to  C x  = d
    """
    qp_G = 0.5 * (H + H.T)  # make sure P is symmetric
    qp_a = -q
    qp_C = None
    qp_b = None
    meq = 0
    if C is not None:
        if G is not None:
            qp_C = -np.vstack([C, G]).T
            qp_b = -np.hstack([d, h])
        else:
            qp_C = -C.transpose()
            qp_b = -d
        meq = C.shape[0]
    elif G is not None:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
    res = quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)
    if verbose:
        return res
    return res[0]


def bernstein_deg4(t: float) -> np.array:
    """Quartic Bernstein basis [B0..B4] at scalar t."""
    # as the equation looks like this
    # p(t) = (1-t)^4*p0 + 4(1-t)^3*t*p1 + 6(1-t)^2*t^2*p2 + 4(1-t)*t^3*p3 + t^4*p4
    # so we need to compute the coefficients for each term
    # B0 = (1-t)^4
    # B1 = 4*(1-t)^3*t
    # B2 = 6*(1-t)^2*t^2
    # B3 = 4*(1-t)*t^3
    # B4 = t^4
    tt = t
    omt = 1.0 - tt
    B0 = omt**4
    B1 = 4 * omt**3 * tt
    B2 = 6 *  omt**2 * tt**2
    B3 = 4 * omt * tt**3
    B4 = tt**4
    return np.array([B0, B1, B2, B3, B4], dtype=np.float64)

def build_H_q_quartic_bezier(
    q0: np.array,
    qe: np.array,
    t_samples: np.array,
    g_samples: np.array,
    l2_reg: float = 0.0
) -> Tuple[np.array, np.array]:


    """
    t_samples: (N,) in [0,1]
    g_samples: (N, 15) targets
    Returns H (45x45), q (45,)
    """

    ## p(t)=P0​B0​ + (P1​B1 ​+ P2​B2 ​+ P3​B3 ​) + P4​B4
    ## need to be p(t) = Ax + b
    ## where b = P0​B0​  + P4​B4

    ## variables are [P1_dimensions, P2_dimensions, P3_dimensions]
    ## and A = [
    ##     [B1, 0, 0, 0, ...., B2, 0, 0, 0, ...., B3, 0, 0, 0,0...],
    ##     [0, B1, 0, 0, ...., 0, B2, 0, 0, ...., 0, B3, 0, 0, ...., 0],
    ##     [0, 0, B1, 0, ...., 0, 0, B2, 0, ...., 0, 0, B3, 0, ...., 0],
    ##                              .....
    ##     [0, 0, 0, B1, ...., 0, 0, 0, B2, ...., 0, 0, 0, B3, ...., 0],
    ##     [0, 0, 0, 0, ...., B3, 0, 0, 0, ...., 0, 0, 0, 0, ...., B3]
    ## ]

    # we need to compute the coefficients for each term
    t_samples = np.asarray(t_samples, dtype=np.float64).ravel()
    g_samples = np.asarray(g_samples, dtype=np.float64)
    assert g_samples.shape == (t_samples.size, 15)

    d = 15
    n_ctrl = 3
    nvars = d * n_ctrl

    H = np.zeros((nvars, nvars), dtype=np.float64)
    q = np.zeros(nvars, dtype=np.float64)

    # samples in the computed trajectory we want to fit
    # Should look like this
    # ||p(t) - g(t)||^2
    # ||A x + b - g(t)||^2
    # ||A x - (g(t) - b)||^2
    # 1/2 x.T A.T A x + (-A.T (g(t) - b)) x
    # 1/2 x.T A.T A x + (A.T (b - g(t))) x

    for ti, gi in zip(t_samples, g_samples):
        B = bernstein_deg4(ti)
        b = B[0] * q0 + B[-1] * qe
        A_i = np.concatenate([np.eye(d) * b for b in B[1:-1]], axis=1)# (15, 45): per-dim applies same B across control blocks
        H += A_i.T @ A_i               # accumulate quadratic term
        q += A_i.T @ (b - gi)     # accumulate linear term

    if l2_reg > 0.0:
        # Optional small ridge for PD Hessian (helps quadprog)
        H += 2.0 * l2_reg * np.eye(nvars, dtype=np.float64)

    return H, q


def accel_equalities_known_endpoints(
    q0: np.array,
    qe: np.array
) -> Tuple[np.array, np.array]:
    d = 15
    C_start = np.concatenate([np.eye(d) * i for i in [-2., 1., 0.]], axis=1)  # P2 - 2P1 = -P0
    C_end   = np.concatenate([np.eye(d) * i for i in [ 0., 1., -2.]], axis=1) # P2 - 2P3 = -P4
    C = np.vstack([C_start, C_end])                        # (30,45)
    dvec = np.concatenate([-q0, -qe])                      # (30,)
    return C, dvec


def construct_collision_constraints(
    discretization_steps,
    distance_to_obstacle_function,
    distance_to_obstacle_threshold):

    ## we need to construct inequality constraints for the collision avoidance
    ## form of inequality constraint is
    ## distance_to_obstacle_function(x) <= distance_to_obstacle_threshold
    ## distance_function(p(t)) <= distance_to_obstacle_threshold
    ## distance_function(Ax+b) <= distance_to_obstacle_threshold

    pass


def get_bezier_control_points(
    path: List[np.array]) -> List[np.array]:

    # assign specific times to intermediate points
    # this is based on the distance between points in the path
    point_times = assign_time_to_intermediate_points(path)

    q0 = path[0]
    qe = path[-1]

    t_samples = np.array(list(point_times.values()))
    g_samples = path[1:-1]

    l2_reg = 1e-6

    H, q = build_H_q_quartic_bezier(q0, qe, t_samples, g_samples, l2_reg=l2_reg)
    C, d = accel_equalities_known_endpoints(q0, qe)
    res = quadprog_solve_qp(H, q, C=C, d=d)

    if res is None or res.size == 0:
        return

    control_points = []
    for i in range(0, len(res), 15):
        control_points.append(res[i:i+15])

    return [q0] + control_points + [qe]
