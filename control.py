#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import pickle
import numpy as np

from zmq.constants import THREAD_AFFINITY_CPU_REMOVE
from bezier import Bezier
from typing import List
import pinocchio as pin
from cube_trajectory_wrapper import GlobalMinJerkLinearJTraj
from tools import setcubeplacement
from config import LEFT_HAND, RIGHT_HAND
from qp_utils import get_bezier_control_points
import pybullet as pyb
    
# in my solution these gains were good enough for all joints but you might want to tune this.
Kp = 1000               # proportional gain (P of PD)
Kv = 30 * np.sqrt(Kp)   # derivative gain (D of PD)

Kx_lin = 2000  # Very stiff position control
Dx_lin = 5 * np.sqrt(Kx_lin)

Kx_rot = 1000
Dx_rot = 10          

# Whether to use Bezier curve for trajectory generation
USE_BEZIER = False

if USE_BEZIER:
    Kp = 1000               # proportional gain (P of PD)
    Kv = 30 * np.sqrt(Kp)   # derivative gain (D of PD)


def controllaw(sim, robot, trajs, tcurrent):
    """Joint trajectory tracking with end-effector force correction"""
    q, vq = sim.getpybulletstate()

    q_target = trajs[0](tcurrent)
    vq_target = trajs[1](tcurrent)
    vvq_target = trajs[2](tcurrent)
    
    vvq_des = vvq_target + Kp * (q_target - q) + Kv * (vq_target - vq)
    M = robot.mass(q)
    h = robot.nle(q, vq)
    tau_base = M @ vvq_des + h

    cube_pos, cube_quat = pyb.getBasePositionAndOrientation(sim.cubeId)
    R = np.array(pyb.getMatrixFromQuaternion(cube_quat)).reshape(3, 3)
    cube_current_SE3 = pin.SE3(R, np.array(cube_pos))

    # cube_target_SE3 = pin.SE3(rotate('z', 0.), cube_trajs(tcurrent))
    setcubeplacement(robot, cube, cube_current_SE3)

    # Forward kinematics
    model, data = robot.model, robot.data
    left_id = model.getFrameId(LEFT_HAND)
    right_id = model.getFrameId(RIGHT_HAND)

    pin.forwardKinematics(model, data, trajs[0](0))
    pin.updateFramePlacements(model, data)

    oMLhand_desired = data.oMf[left_id]   # From joint trajectory
    oMRhand_desired = data.oMf[right_id]

    
    pin.forwardKinematics(model, data, q, vq)
    pin.updateFramePlacements(model, data)
    pin.computeJointJacobians(model, data, q)
    
    # Jacobians
    J_l = pin.computeFrameJacobian(model, data, q, left_id, 
                                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    J_r = pin.computeFrameJacobian(model, data, q, right_id,
                                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                                    
    oMLhand_actual = data.oMf[left_id]
    oMRhand_actual = data.oMf[right_id]

    error_pos_hands = oMLhand_actual.translation - oMRhand_actual.translation
    
    # Orientation errors
    R_err_l = oMLhand_desired.rotation @ oMLhand_actual.rotation.T
    err_rot_l = pin.log3(R_err_l)

    R_err_r = oMRhand_desired.rotation @ oMRhand_actual.rotation.T
    err_rot_r = pin.log3(R_err_r)

    # Velocities
    v_l = pin.getFrameVelocity(model, data, left_id, 
                               pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    vel_l = np.hstack([v_l.linear, v_l.angular])
    v_r = pin.getFrameVelocity(model, data, right_id,
                               pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    vel_r = np.hstack([v_r.linear, v_r.angular])
    
    # Compute correction forces
    F_l = np.hstack([
        -Kx_lin * error_pos_hands - Dx_lin * vel_l[:3],
        Kx_rot * err_rot_l - Dx_rot * vel_l[3:]
    ])
    F_r = np.hstack([
        Kx_lin * error_pos_hands - Dx_lin * vel_r[:3],
        Kx_rot * err_rot_r - Dx_rot * vel_r[3:]
    ])
    
    # Map to joint torques
    tau_l = J_l[:, :].T @ F_l
    tau_r = J_r[:, :].T @ F_r

    tau_l = np.clip(tau_l, -200.0, 200.0)
    tau_r = np.clip(tau_r, -200.0, 200.0)

    # Combine: base tracking + end-effector correction
    tau_total = tau_base + tau_r + tau_l
    sim.step(tau_total)
 
if __name__ == "__main__":
        
    from tools import setupwithpybullet, setupwithpybulletandmeshcat, rununtil
    from config import DT
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET    
    from inverse_geometry import computeqgrasppose
    from path import computepath
    
    robot, sim, cube = setupwithpybullet()
    
    q0, successinit = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
    qe, successend = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT_TARGET,  None)


    if not successinit or not successend:
        raise RuntimeError('Failed to successfully compute grasp pose!')

    path, cube_placements = computepath(
        q0,
        qe,
        CUBE_PLACEMENT,
        CUBE_PLACEMENT_TARGET,
        robot=robot,
        cube=cube,
        computeqgrasppose=computeqgrasppose)

    sim.setqsim(q0)

    def maketraj(points, T):
        q_of_t = Bezier(points,t_max=T)
        vq_of_t = q_of_t.derivative(1)
        vvq_of_t = vq_of_t.derivative(1)
        return q_of_t, vq_of_t, vvq_of_t


    def maketraj_with_joint_space_linear(
            waypoints: List[np.array],
            T: int):
        joint_space_trajectory = GlobalMinJerkLinearJTraj(
            waypoints=waypoints,
            T_max=T,
        )

        return (
            joint_space_trajectory,
            joint_space_trajectory.derivative(1),
            joint_space_trajectory.derivative(2)
        )
    
    total_time=10.0

    if USE_BEZIER:
        control_points = get_bezier_control_points(path)
        if control_points is None:
            raise RuntimeError('Failed to successfully compute Bezier control points!')
        trajs = maketraj(control_points, total_time)
    else:
        trajs = maketraj_with_joint_space_linear(
            waypoints=path,
            T=total_time
        )

    tcur = 0.
    
    while tcur < total_time:
        rununtil(
            controllaw, 
            DT, 
            sim, 
            robot, 
            trajs, 
            tcur)
        tcur += DT

    