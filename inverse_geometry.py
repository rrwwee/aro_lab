#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import pinocchio as pin 
import numpy as np
from numpy.linalg import pinv,inv,norm,svd,eig
from tools import collision, getcubeplacement, setcubeplacement, jointlimitsviolated
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

from tools import setcubeplacement

def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    setcubeplacement(robot, cube, cubetarget)

    q = qcurrent.copy()
    DT = EPSILON * 10
    Kpost = 1 # penalty term for initial postural change

    max_updates = 1000
    updates_done = 0
    q_opt_found = False

    oMLhook = getcubeplacement(cube, LEFT_HOOK)
    oMRhook = getcubeplacement(cube, RIGHT_HOOK)

    while updates_done < max_updates:

        pin.framesForwardKinematics(robot.model, robot.data, q)
        pin.computeJointJacobians(robot.model, robot.data, q)

        left_hand_frameid = robot.model.getFrameId(LEFT_HAND)
        oMLhand = robot.data.oMf[left_hand_frameid]

        right_hand_frameid = robot.model.getFrameId(RIGHT_HAND)
        oMRhand = robot.data.oMf[right_hand_frameid]

        # FIRST TASK: moving left hand to the left hook
        
        # get the homogeneous matrix that takes from left hand frame to hook frame
        lhandMLhook = oMLhand.inverse() * oMLhook

        # 6D spatial vector that is needed to be applied in 1 second 
        # so the left hand's frame will be in the position of hook frame's position
        lhand_nu_lhand = pin.log(lhandMLhook).vector

        # get the 6D vector in the world frame
        lhand_nu_world = oMLhand.action @ lhand_nu_lhand

        # Get corresponding jacobian for lhand
        o_Jlhand = pin.computeFrameJacobian(
            robot.model,
            robot.data,
            q,
            left_hand_frameid, 
            pin.WORLD
            )
        
        # get the joint velocities to move left hand coordinates to the hook position
        lhand_vq = pinv(o_Jlhand) @ lhand_nu_world

        # get the null-space projector matrix of the left hand Jacobian
        # any vector multiplied by Plhand is projected into the null space of o_Jlhand
        # meaning if that vector is multiplied by o_Jlhand = 0 => lhand is not moved
        Plhand = np.eye(robot.nv) - pinv(o_Jlhand) @ o_Jlhand

        # SECOND TASK: moving right hand to the right hook
        # Get corresponding jacobian for rhand
        o_Jrhand = pin.computeFrameJacobian(
            robot.model,
            robot.data,
            q,
            right_hand_frameid, 
            pin.WORLD
            )
        
        o_Jrhand_lhand_null = o_Jrhand @ Plhand

        # Spatial velocity of right hand frame
        rhand_nu_world_from_lhand_vq = o_Jrhand @ lhand_vq

        # get the right hook into the right hand frame
        rhandMLhook = oMRhand.inverse() * oMRhook

        # get the 6D spacial velocity matrix 
        rhand_nu_rhand = pin.log(rhandMLhook).vector

        # get the 6D vector in the world frame
        rhand_nu_world = oMRhand.action @ rhand_nu_rhand

        # substract the spatial velocity that from left hand
        rhand_nu_world -= rhand_nu_world_from_lhand_vq

        # calculate right hand joint spatial velocities projected 
        # into the left hand jacobian matrix's null space
        rhand_vq_null = pinv(o_Jrhand_lhand_null) @ rhand_nu_world


        # THIRD TASK: add postural bias 
        # as the initial configuration of the robot is natural, add penalty of getting far from the q_ref 
        Plhandrhand = Plhand - pinv(o_Jrhand_lhand_null) @ o_Jrhand @ Plhand
        delta_q_to_ref = pin.difference(robot.model, q, robot.q0)
        v_post = Kpost * delta_q_to_ref
        postural_vq_null = Plhandrhand @ v_post

        # combine both hands' joint velocities
        vq = lhand_vq + rhand_vq_null + postural_vq_null


        # check if the angle and translation to the targets are small enough 
        # so we can early stop the optimization
        rotation_to_left_hook = norm(lhand_nu_lhand[:3])
        translation_to_left_hook = norm(lhand_nu_lhand[3:])

        rotation_to_right_hook = norm(rhand_nu_world[:3])
        translation_to_right_hook = norm(rhand_nu_world[3:])

        left_hook_close_enough = rotation_to_left_hook <= EPSILON and translation_to_left_hook <= EPSILON / 10
        right_hook_close_enough = rotation_to_right_hook <= EPSILON and translation_to_right_hook <= EPSILON / 10

        if left_hook_close_enough and right_hook_close_enough:
            if not collision(robot, q) and not jointlimitsviolated(robot, q):
                q_opt_found = True
                break

        # get the q by integrating the current q and unit time joint velocities multiplied by DT
        q = pin.integrate(robot.model, q, vq * DT)
        updates_done += 1

        if viz:
            viz.display(q)

    return q, q_opt_found
            
if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    robot, cube, viz = setupwithmeshcat()
    
    q = robot.q0.copy()
    
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    
    updatevisuals(viz, robot, cube, q0)
    
    
    
