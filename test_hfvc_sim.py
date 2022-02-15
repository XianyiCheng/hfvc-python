import os
from pathlib import Path

import numpy as np
from matplotlib import cm
import open3d as o3d
from autolab_core import RigidTransform
from action_sampler import *

np.set_printoptions(precision=4,suppress=True, linewidth=120)

# input: initial pose, contact points in the object frame, goal pose, contact points in the goal frame, robot contacts
# output: pick a robot-object contact points, hfvc

if __name__ == '__main__':
    path_to_franka_ee_collision_pts = './data/ee_collision_pts.npz'
    path_to_data = './data/sugar_box_s0_v1/2022-01-17_19-49-24/data'


    franka_ee_collision_pts = np.load(path_to_franka_ee_collision_pts)['0.0']
    data_dir = Path(path_to_data)

    for item_number in os.listdir(data_dir):
        print(f'initial state {item_number}')
        initial_state_dir = data_dir / item_number / 'initial_state'
        subgoal_states_dir = data_dir / item_number / 'subgoal_states'

        # load init state data

        ## raw point cloud data
        init_pc = np.load(initial_state_dir / 'pc.npz')
        init_pts = init_pc['pts']
        init_segs = init_pc['segs']
        init_normals = init_pc['normals']

        ## obj env cts
        # for initial state
        # everything before the for loop is initial state
        obj_env_cts = np.load(initial_state_dir / 'obj_env_cts.npz') # same index, same contact point (with obj_env_cts_subgoal)
        ct_pts_local = obj_env_cts['ct_pts_local'] # n x 3
        ct_pts_world = obj_env_cts['ct_pts_world'] # n x 3
        cts_mask = obj_env_cts['cts_mask'] # n x 1, binary, true: in contact

        ## obj pose
        T_obj_world = RigidTransform.load(str(initial_state_dir / 'T_obj_world.tf'))

        # visualize init state data
        obj_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(init_pts[init_segs == 2]))
        obj_pcd.paint_uniform_color(cm.tab10.colors[0])
        shelf_bottom_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(init_pts[init_segs == 3]))
        shelf_bottom_pcd.paint_uniform_color(cm.tab10.colors[1])
        shelf_walls_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(init_pts[init_segs >= 4]))
        shelf_walls_pcd.paint_uniform_color(cm.tab10.colors[2])

        obj_env_cts_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ct_pts_world))
        obj_env_cts_color = np.zeros((len(ct_pts_world), 3))
        obj_env_cts_color[cts_mask] = np.array([1, 0, 0]) # label in-ct as red
        obj_env_cts_pcd.colors = o3d.utility.Vector3dVector(obj_env_cts_color)

        T_obj_world_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        T_obj_world_axes.transform(T_obj_world.matrix)

        origin_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        o3d.visualization.draw_geometries([
            obj_pcd, shelf_bottom_pcd, shelf_walls_pcd, obj_env_cts_pcd, T_obj_world_axes, origin_axes
        ])

        # load subgoal state data
        for subgoal_number in os.listdir(subgoal_states_dir):
            subgoal_state_dir = subgoal_states_dir / subgoal_number


            pc_subgoal = np.load(subgoal_state_dir / 'pc.npz') # point cloud to the subgoal
            obj_env_cts_subgoal = np.load(subgoal_state_dir / 'obj_env_cts.npz')
            cts_mask = obj_env_cts_subgoal['cts_mask']
            T_obj_world_subgoal = RigidTransform.load(str(subgoal_state_dir / 'T_obj_world.tf'))

            # in the following it is just visualization
            obj_pcd_subgoal = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc_subgoal['pts'][pc_subgoal['segs'] == 2]))
            obj_pcd_subgoal.paint_uniform_color(cm.tab10.colors[0])

            obj_env_cts_pcd_subgoal = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj_env_cts_subgoal['ct_pts_world']))
            obj_env_cts_color_subgoal = np.zeros((len(ct_pts_world), 3))
            obj_env_cts_color_subgoal[cts_mask] = np.array([1, 0, 0]) # label in-ct as red
            obj_env_cts_pcd_subgoal.colors = o3d.utility.Vector3dVector(obj_env_cts_color_subgoal)

            T_obj_world_axes_subgoal = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            T_obj_world_axes_subgoal.transform(T_obj_world_subgoal.matrix)

            # robot object relationship
            # for subgoal: did collision filtering
            # every loop is the contact point from a robot pose
            obj_robot_cts_dir = subgoal_state_dir / 'obj_robot_cts'
            for robot_obj_ct_number in os.listdir(obj_robot_cts_dir):
                print(f'subgoal {subgoal_number} | robot-obj ct {robot_obj_ct_number}')
                obj_robot_ct_dir = obj_robot_cts_dir / robot_obj_ct_number

                robot_obj_cts = np.load(obj_robot_ct_dir / 'cts.npz') # n x 3, in the (?) frame
                # robot_obj_cts is a dictionary
                    # init_ct_pt_world: robot object contact point for the initial state in the world frame
                    # init_ct_normal_world: normal (of above)
                    # final_ct_pts_world: robot object contact point for the subgoal state in the world frame
                    # final_ct_normals_world: normal (of above)
                    # ct_pt_local: robot object contact point for the initial & the subgoal state in the object frame
                    # ct_normal_local: normal (of above)

                T_ee_world = RigidTransform.load(str(obj_robot_ct_dir / 'T_ee_world.tf')) # the end-effector pose at the subgoal state in the world frame

                franka_ee_collision_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector((franka_ee_collision_pts)))
                franka_ee_collision_pcd.transform(T_ee_world.matrix)

                o3d.visualization.draw_geometries([
                    obj_pcd_subgoal, obj_env_cts_pcd_subgoal, T_obj_world_axes_subgoal,
                    franka_ee_collision_pcd,
                    shelf_bottom_pcd, shelf_walls_pcd, origin_axes
                ])

                # extract data for hfvc
                mu_h = 0.8
                mu_e = 0.5
                F_external = np.array([0,0,-9.8,0,0,0, 0,0,0,0,0,0]) 
                T_WO = np.identity(4)
                T_WO[0:3,0:3] = T_obj_world.rotation
                T_WO[0:3, 3] = T_obj_world.translation

                T_WO_goal = np.identity(4)
                T_WO_goal[0:3,0:3] = T_obj_world_subgoal.rotation
                T_WO_goal[0:3, 3] = T_obj_world_subgoal.translation

                T_WH_goal = np.identity(4)
                T_WH_goal[0:3,0:3] = T_ee_world.rotation
                T_WH_goal[0:3, 3] = T_ee_world.translation

                results = compute_hfvc(T_WO, T_WO_goal, T_WH_goal, obj_env_cts, obj_env_cts_subgoal, robot_obj_cts, mu_h, mu_e, F_external)
                if results == False:
                    print("No hfvc solution \n")
                else:
                    print("Hfvc results: \n velocity control dimension ", results[0], 
                        ",\n force control dimension ", results[1], 
                        ",\n velocity control magnitute ", results[2],
                        ",\n force control magnitute ", results[3],
                        ",\n control matrix \n", results[4])
                print("\n")
