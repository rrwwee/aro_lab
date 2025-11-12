# ARO Lab Overview

See demos in `demo/*.mp4`

## 1. Inverse Geometry (`inverse_geometry.py`)

- `computeqgrasppose` iteratively integrates `q` with steps of `DT = EPSILON * 10`, stopping when both grippers align with their hooks.
- Three stacked tasks share null spaces: left hand tracks `LEFT_HOOK`, right hand follows `RIGHT_HOOK` without disturbing the left hand, and a postural term keeps the robot near `q0`.
- Each candidate pose must respect joint limits, avoid collisions, and keep the hooks aligned before returning `success=True`.

## 2. Motion Planning (`path.py`)

- `random_cube_placement` samples cube poses within table bounds until it finds one that is collision-free.
- `construct_rrt` grows a tree of cube placements and matching grasp configurations from `computeqgrasppose`.
- Before adding a vertex, we run inverse geometry, ensure no collisions, and keep at least `2e-3` m clearance from obstacles using `distanceToObstacle`.
- `get_shortcut_from_rrt_path` removes redundant waypoints once the goal is reached.

## 3. Dynamics (`control.py`)

- **3.1 Trajectory Planning**
  - **3.1.1 Piecewise-linear:** interpolate between the RRT waypoints, assign each segment a duration proportional to the joint-space distance, and use `GlobalMinJerkLinearJTraj` to get a smooth speed profile that rises from near-zero, peaks mid-way, and returns to zero. Other velocity profiles are listed in `cube_trajectory_wrapper.py`, but this one performed best.
  - **3.1.2 Bezier curves:** solve for a 4th-degree Bezier trajectory via QP by setting `USE_BEZIER = True` in `control.py`. <span style="color:red;font-weight:bold">It behaves well only when the RRT search space is not heavily constrained</span>; the demo in `demo/control_with_bezier.mp4` shows a successful run under those conditions.

- **3.2 Torque Control Law**
  - Operates entirely in joint configuration space: the planner delivers `q`, `q̇`, `q̈` targets, and we compute `tau_base` from model-based dynamics using those signals.
  - Adds task-space correction forces so the total torque is `tau_total = tau_base + tau_r + tau_l`, nudging each hand toward the other to maintain a firm, cooperative grasp on the cube.