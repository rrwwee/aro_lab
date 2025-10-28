#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import numpy as np

from bezier import Bezier
    
# in my solution these gains were good enough for all joints but you might want to tune this.
Kp = 300.               # proportional gain (P of PD)
Kv = 2 * np.sqrt(Kp)   # derivative gain (D of PD)

def controllaw(sim, robot, trajs, tcurrent, cube):
    q, vq = sim.getpybulletstate()

    q_target = trajs[0](tcurrent)
    vq_target = trajs[1](tcurrent)


    error = q_target - q
    vvq_des = Kp * error + Kv * vq_target

    # data = robot.data

    # Mass matrix
    M = robot.mass(q)            
    h = robot.nle(q, vq)

    torques = M @ vvq_des + h

    # print('Order: ', sim.bulletCtrlJointsInPinOrder)
     
    # torques = [torques[i] for i in sim.bulletCtrlJointsInPinOrder]
    sim.step(torques)

if __name__ == "__main__":
        
    from tools import setupwithpybullet, setupwithpybulletandmeshcat, rununtil
    from config import DT
    
    robot, sim, table, cube = setupwithpybullet()
    
    
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET    
    from inverse_geometry import computeqgrasppose
    from path import computepath
    
    q0,successinit = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
    qe,successend = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT_TARGET,  None)
    path, cube_placements = computepath(
        q0,
        qe,
        CUBE_PLACEMENT, 
        CUBE_PLACEMENT_TARGET,
        robot=robot, 
        cube=cube,
        table=table,
        computeqgrasppose=computeqgrasppose)


    print('Success Init: ', successinit)
    print('Success End: ', successend)

    
    #setting initial configuration
    sim.setqsim(q0)
    
    
    #TODO this is just an example, you are free to do as you please.
    #In any case this trajectory does not follow the path 
    #0 init and end velocities
    def maketraj(q0,q1,path, T): #TODO compute a real trajectory !
        poimtlist = [q0, q0] + path + [q1, q1]
        # point = [q0, q0, q1, q1]
        q_of_t = Bezier(poimtlist,t_max=T)
        vq_of_t = q_of_t.derivative(1)
        vvq_of_t = vq_of_t.derivative(1)
        return q_of_t, vq_of_t, vvq_of_t
    
    
    #TODO this is just a random trajectory, you need to do this yourself
    total_time=10.
    trajs = maketraj(q0, qe, path, total_time)   
    
    tcur = 0.
    
    
    while tcur < total_time:
        rununtil(
            controllaw, 
            DT, 
            sim, 
            robot, 
            trajs, 
            tcur, 
            cube)
        tcur += DT
    
    
    