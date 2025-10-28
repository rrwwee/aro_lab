#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:44:32 2023

@author: stonneau
"""

import pinocchio as pin
import numpy as np
from numpy.linalg import pinv

from config import LEFT_HAND, RIGHT_HAND
import time

from typing import Callable, Tuple, List
from pinocchio.utils import rotate
from tools import setcubeplacement


#### Helper Functions for Random Cube Placement Sampling ####

def get_table_borders(table: pin.robot_wrapper.RobotWrapper) -> Tuple[np.array, np.array]:
    """ Returns the minimum and maximum x, y, z coordinates of the table.
    Args:
        table: table object
    Returns:
        xyz_min: minimum x, y, z coordinates of the table's leftmost and bottommost corner
        xyz_max: maximum x, y, z coordinates of the table's rightmost and topmost corner
    """
    pin.updateGeometryPlacements(table.model, table.data, table.collision_model, table.collision_data)
    table_surface = table.collision_model.geometryObjects[0].geometry
    oMtable_surface = table.collision_data.oMg[0]
    R, t = oMtable_surface.rotation, oMtable_surface.translation


    half = table_surface.halfSide
    local_corners = np.array([[sx, sy, sz]
                                for sx in [+1, -1]
                                for sy in [+1, -1]
                                for sz in [+1, -1]]) * half
    world_corners = (R @ local_corners.T).T + t

    xyz_min = world_corners.min(axis=0)
    xyz_max = world_corners.max(axis=0)

    return xyz_min, xyz_max



def random_cube_placement(
    robot: pin.robot_wrapper.RobotWrapper, 
    cube: pin.robot_wrapper.RobotWrapper, 
    table: pin.robot_wrapper.RobotWrapper, 
    check_collisions: bool = True
    ) -> pin.pinocchio_pywrap_default.SE3:
    """ Returns a random cube placement on the table that is collision free.
    Args:
        robot: robot object
        cube: cube object
        table: table object
        check_collisions: whether to check for collisions
    Returns:
        valid_placement: random cube placement on the table that is collision free
    """
    xyz_min, xyz_max = get_table_borders(table)
    
    collision = True
    valid_placement = None

    while collision:
        x = np.random.uniform(xyz_min[0], xyz_max[0])
        y = np.random.uniform(xyz_min[1], xyz_max[1])
        # TODO: change harcoded max z to a more clever bounding
        z = np.random.uniform(0.93, 2)

        placement = pin.SE3(rotate('z', 0.), np.array([x, y, z]))
        setcubeplacement(robot, cube, placement)
        if check_collisions:
            collision = pin.computeCollisions(cube.collision_model, cube.collision_data, False)
            valid_placement = placement if not collision else valid_placement
        else:
            valid_placement = placement
            collision = False
    
    return valid_placement


#### Helper Functions and Classes for RRT ####

class Vertex:
    def __init__(self, q, grasping_q, parent):
        self.q = q 
        self.grasping_q = grasping_q
        self.parent = parent

class RRT:
    def __init__(self, 
                 root_q: pin.pinocchio_pywrap_default.SE3, 
                 root_grasping_q: np.array, 
                 robot: pin.robot_wrapper.RobotWrapper, 
                 cube: pin.robot_wrapper.RobotWrapper,
                 get_grasping_poseq: Callable):
        
        self.robot = robot
        self.cube = cube
        self.get_grasping_poseq = get_grasping_poseq

        # initialize the tree
        self.nodes = [
            Vertex(q=root_q, grasping_q=root_grasping_q, parent=None)
            ]

    @staticmethod
    def calc_distance(
        q1: np.array, 
        q2: np.array) -> float:    
     '''Return the euclidian distance between two configurations'''
     return np.linalg.norm(q2 - q1)
    
    @staticmethod
    def lerp(
        q0: np.array, 
        q1: np.array, 
        t: float
        ) -> np.array:
        """
        Performs linear interpolation between the given points and for the step t
        """
        return RRT.get_placement_from_translation((1 - t) * q0 + t * q1)
    
    @staticmethod
    def get_placement_from_translation(translation):
        return pin.SE3(rotate('z', 0.), np.array(translation))

    def add_edge(self, 
                 parent_idx: int,
                 new_q: pin.pinocchio_pywrap_default.SE3, 
                 new_grasping_q: np.array) -> None:
        
        """
        Constructs and adds a new vertex to the tree.
        """
        
        self.nodes.append(
            Vertex(q=new_q,
                   grasping_q=new_grasping_q,
                   parent=self.nodes[parent_idx])
        )

    def get_nearest_vertex(self, 
                           q: pin.pinocchio_pywrap_default.SE3
                           ) -> Tuple[int, Vertex]:
        
        """
        Args:
            q: random configuration of the cube (SE3)
        Returns:
            nearest_vertex_idx: nearest vertex index in the tree
            nearest_vertex: nearest vertex
        """
        nearest_vertex_idx = None
        smallest_distance = None

        for i, vertex in enumerate(self.nodes): 
            dist = self.calc_distance(vertex.q.translation, q.translation)
            if not smallest_distance or dist <= smallest_distance:
                smallest_distance = dist
                nearest_vertex_idx = i

        return nearest_vertex_idx, self.nodes[nearest_vertex_idx]
    
    def get_q_new(self, 
                  q_nearest_vertex: Vertex, 
                  q_rand: pin.pinocchio_pywrap_default.SE3, 
                  discretisation_steps: int,
                  max_delta_q: int = None,
                  ) -> Tuple[pin.pinocchio_pywrap_default.SE3, np.array, bool]:
        
        """ Returns a new configuration that is the closest to the random configuration `q_rand` that can be grasped and is collison free.
        Args:
            q_nearest_vertex: nearest vertex
            q_rand: random configuration
            discretisation_steps: number of discretisation steps
            max_delta_q: maximum delta q
        Returns:
            q_new: new configuration
            grasping_q: grasping pose for the new configuration
            q_new_found: True if the new configuration (other than the `q_nearest_vertex`) is found, False otherwise    
        """
        
        q_new_found = True
        q_end = q_rand.copy()

        dist = self.calc_distance(
            q_nearest_vertex.q.translation, 
            q_rand.translation)
        
        if max_delta_q is not None and dist > max_delta_q:
            q_end = self.lerp(
                q_nearest_vertex.q.translation, 
                q_rand.translation, 
                max_delta_q/dist)

        if self.get_grasping_poseq is not None:
            dt = 1 / discretisation_steps
            for i in range(1, discretisation_steps):
                q = self.lerp(
                    q_nearest_vertex.q.translation, 
                    q_end.translation, 
                    dt*i)
                grasping_q, found_grasping_pose = self.get_grasping_poseq(
                    robot=self.robot, 
                    qcurrent=q_nearest_vertex.grasping_q,
                    cube=self.cube,
                    cubetarget=q,
                )
                if not found_grasping_pose:
                    if i - 1 == 0:
                        q_new_found = False
                    return self.lerp(
                        q_nearest_vertex.q.translation, 
                        q_end.translation, 
                        dt*(i-1)), grasping_q, q_new_found
        else:
            grasping_q = None
                
        return q_end, grasping_q, q_new_found
    
    def check_edge(
            self,
            q_latest_new: pin.pinocchio_pywrap_default.SE3,
            q_latest_new_grasping_pose: np.array,
            q_goal: pin.pinocchio_pywrap_default.SE3,
            discretisation_steps: int = 1000,
            ) -> bool:


        """ Checks if there is a valid edge between the latest new configuration and the goal configuration.
        Args:
            q_latest_new: latest new configuration
            q_latest_new_grasping_pose: grasping pose for the latest new configuration
            q_goal: goal configuration
            discretisation_steps: number of discretisation steps
        Returns:
            valid_edge: True if there is a valid edge, False otherwise
        """
        
        valid_edge = False
        q_latest_new_vertex = Vertex(
            q=q_latest_new,
            grasping_q=q_latest_new_grasping_pose,
            parent=None
        )
        
        # check if we can get to the q_goal from the q_new directly
        q_new, q_new_grasping_q, q_new_found = self.get_q_new(
            q_nearest_vertex=q_latest_new_vertex,
            q_rand=q_goal,
            discretisation_steps=discretisation_steps,
            max_delta_q=None
        )

        if q_new_found:
            distance_to_target = self.calc_distance(q_new.translation, q_goal.translation)
            if distance_to_target < 1e-3:
                valid_edge = True

        return valid_edge
    
    def get_path_from_rrt(self) -> List[Vertex]:
        """ Returns a list of vertices from the root to the latest vertex.
        If there was a valid edge between the latest vertex and the goal configuration, the latest vertex is the vertex with the goal configuration.
        Returns:
            vertices: list of vertices
        """
        vertices = []
        last_vertex = self.nodes[-1]

        while last_vertex.parent is not None:
            vertices.insert(0, last_vertex)
            last_vertex = last_vertex.parent

        vertices.insert(0, last_vertex)

        return vertices
    
    def get_shortcut_from_rrt_path(self, 
                                   discretisation_steps: int = 100,
                                   ) -> List[Vertex]:
        
        """ Returns a shortcut path from the root to the latest vertex.
        Args:
            discretisation_steps: number of discretisation steps
        Returns:
            short_path: list of vertices from the root to the latest vertex that has no redundant intermediate vertices  
        """

        path_vertices = self.get_path_from_rrt()
        short_path = []
        start_idx = 0
        
        while start_idx < len(path_vertices) - 3:
            current_vertex = path_vertices[start_idx]
            short_path.append(current_vertex)

            # search in the next nodes starting from the furthest one
            next_vertex_idx = len(path_vertices) - 1
            while next_vertex_idx > start_idx + 1:
                next_vertex = path_vertices[next_vertex_idx]
                is_valid_edge = self.check_edge(
                    q_latest_new=current_vertex.q,
                    q_latest_new_grasping_pose=current_vertex.grasping_q,
                    q_goal=next_vertex.q,
                    discretisation_steps=discretisation_steps
                )
                if is_valid_edge:
                    start_idx = next_vertex_idx
                    break
                next_vertex_idx -= 1

            if not is_valid_edge:
                start_idx += 1

        short_path.extend(path_vertices[start_idx:])

        return short_path



#### The main function for constructing the RRT ####

def construct_rrt(robot: pin.robot_wrapper.RobotWrapper,
                  cube: pin.robot_wrapper.RobotWrapper,
                  q_init: np.array, 
                  q_goal: np.array,
                  cubeplacementq0: pin.pinocchio_pywrap_default.SE3,
                  cubeplacementqgoal: pin.pinocchio_pywrap_default.SE3,
                  num_iter: int, 
                  radnom_sampler: Callable,
                  max_delta_q: float = None,
                  discretisation_steps: int = 100, 
                  get_grasping_poseq: Callable = None,
                  ):

    """ Constructs the RRT from the root to the goal configuration.
    Args:
        robot: robot object
        cube: cube object
        q_init: initial grasping pose
        q_goal: goal grasping pose
        cubeplacementq0: initial cube placement
        cubeplacementqgoal: goal cube placement
        num_iter: number of iterations of sampling random cube placements
        radnom_sampler: random cube placement sampler
        max_delta_q: maximum delta q between the current and the end configuration used to get q_new configuration
        discretisation_steps: number of discretisation steps used in the get_q_new and check_edge function
        get_grasping_poseq: grasping pose calculator from the cube placement (inverse geometry).
    Returns:
        rrt: RRT object
    """
    rrt = RRT(
        root_q=cubeplacementq0,
        root_grasping_q=q_init,
        robot=robot,
        cube=cube,
        get_grasping_poseq=get_grasping_poseq
        )

    for _ in range(num_iter):
        random_cp_q = radnom_sampler()
        cp_q_nearest_idx, cp_q_nearest_vertex = rrt.get_nearest_vertex(random_cp_q)

        cp_q_new, cp_q_new_grasping_pose, cp_q_new_found = rrt.get_q_new(
            q_nearest_vertex=cp_q_nearest_vertex,
            q_rand=random_cp_q,
            discretisation_steps=discretisation_steps,
            max_delta_q=max_delta_q)
        
        if cp_q_new_found:
            rrt.add_edge(
                parent_idx=cp_q_nearest_idx,
                new_q=cp_q_new,
                new_grasping_q=cp_q_new_grasping_pose,
            )

            is_valid_edge = rrt.check_edge(
                q_latest_new=cp_q_new,
                q_latest_new_grasping_pose=cp_q_new_grasping_pose,
                q_goal=cubeplacementqgoal,
                discretisation_steps=discretisation_steps)
            
            if is_valid_edge:
                rrt.add_edge(parent_idx=len(rrt.nodes) - 1,
                             new_q=cubeplacementqgoal,
                             new_grasping_q=q_goal)
                
                return rrt
    
    return None

#returns a collision free path from qinit to qgoal under grasping constraints
#the path is expressed as a list of configurations
def computepath(qinit, qgoal, cubeplacementq0, cubeplacementqgoal):
    """ Returns a collision free path from qinit to qgoal under grasping constraints.
    Args:
        qinit: initial configuration
        qgoal: goal configuration
        cubeplacementq0: initial cube placement
        cubeplacementqgoal: goal cube placement
    Returns:
        path: list of configurations
    """

    global robot
    global cube 
    global table
    global computeqgrasppose

    NUM_ITER = 200
    RANDOM_SAMPLER = lambda: random_cube_placement(robot=robot, cube=cube, table=table)
    MAX_DELTA_Q = None
    DISCRETISATION_STEPS = 100
    GET_GRAPSING_POSEQ = computeqgrasppose

    print('Searching for a valid path...(this may take some time)')
    rrt = construct_rrt(
        robot=robot,
        cube=cube,
        q_init=qinit,
        q_goal=qgoal,
        cubeplacementq0=cubeplacementq0,
        cubeplacementqgoal=cubeplacementqgoal,
        num_iter=NUM_ITER,
        radnom_sampler=RANDOM_SAMPLER,
        max_delta_q=MAX_DELTA_Q,
        discretisation_steps=DISCRETISATION_STEPS,
        get_grasping_poseq=GET_GRAPSING_POSEQ,
    )

    if not rrt:
        print(
            """Valid Path was not found! 
            Try to increase discretisation steps or try with different start and end configurations.""")
        
        return [], []

    print('Path found! Retrieving shortcut path from RRT...')

    path_vertices = rrt.get_shortcut_from_rrt_path()

    path_configurations = [vertex.grasping_q for vertex in path_vertices]
    cube_placements = [vertex.q for vertex in path_vertices]

    print('Path is ready!')

    return path_configurations, cube_placements

def displaypath(robot, cube, path, cube_placements, dt, viz):
    for q, cube_placement in zip(path, cube_placements):
        setcubeplacement(robot, cube,cube_placement)
        viz.display(q)
        time.sleep(dt)


if __name__ == "__main__":
    from tools import setupwithmeshcat
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from inverse_geometry import computeqgrasppose
    
    robot, cube, table, viz = setupwithmeshcat()
    
    q = robot.q0.copy()
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    
    if not(successinit and successend):
        print ("error: invalid initial or end configuration")
    
    path, cube_placements = computepath(q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)
    
    displaypath(robot, cube, path, cube_placements, dt=0.5, viz=viz) #you ll probably want to lower dt