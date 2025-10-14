#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import pinocchio as pin 
import numpy as np
from numpy.linalg import pinv,inv,norm,svd,eig
import time
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

from tools import setcubeplacement

def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    q = qcurrent.copy()

    lhand_id = robot.model.getFrameId(LEFT_HAND)
    rhand_id = robot.model.getFrameId(RIGHT_HAND)
    
    oMlhook = getcubeplacement(cube, LEFT_HOOK)
    oMrhook = getcubeplacement(cube, RIGHT_HOOK)

    success = False

    for i in range(1000):
        pin.framesForwardKinematics(robot.model, robot.data, q)
        pin.computeJointJacobians(robot.model, robot.data, q)

        oMlhand = robot.data.oMf[lhand_id]
        oMrhand = robot.data.oMf[rhand_id]

        lhandMlhook = oMlhand.inverse()*oMlhook
        rhandMrhook = oMrhand.inverse()*oMrhook
        
        nu_L = pin.log(lhandMlhook).vector
        nu_R = pin.log(rhandMrhook).vector

        J_L = pin.computeFrameJacobian(robot.model, robot.data, q, lhand_id, pin.LOCAL)
        J_R = pin.computeFrameJacobian(robot.model, robot.data, q, rhand_id, pin.LOCAL)

        JL_p = pinv(J_L)
        v1 = JL_p @ nu_L
        N1 = np.eye(robot.nv) - JL_p @ J_L

        JRN = J_R @ N1
        JRN_p = pinv(JRN)

        v2 = JRN_p @ (nu_R - J_R @ v1)
        vq = v1 + N1 @ v2

        N2 = N1 - JRN_p @ JRN @ N1
        v3 = pin.difference(robot.model, q, robot.q0)

        vq += N2 @ v3

        q = pin.integrate(robot.model,q, vq)

        lerror = norm(oMlhand.translation - oMlhook.translation)
        rerror = norm(oMrhand.translation - oMrhook.translation)

        if viz:
            viz.display(q)
            time.sleep(1)
        
        if lerror < EPSILON and rerror < EPSILON and not collision(robot, q):
            success = True
            break
    
    return q, success

            
if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    robot, cube, viz = setupwithmeshcat()
    
    q = robot.q0.copy()
    
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    
    updatevisuals(viz, robot, cube, q0)
    
    
    
