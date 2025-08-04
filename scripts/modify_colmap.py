import os
import sys
import copy

import numpy as np
import open3d as o3d

abs_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(abs_path) + "/..")

from read_write_model import (
    read_model,
    write_model,
    Point3D,
    Image,
    qvec2rotmat,
    rotmat2qvec,
)



def colmapP3d2o3d(points3D):
    # get xyz, rgb
    xyz_list = [point.xyz for point in points3D.values()]
    xyz_array = np.array(xyz_list)

    rgb_list = [point.rgb for point in points3D.values()]
    rgb_array = np.array(rgb_list)

    # convert to Open3D pcd
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_array)
    # normalize to [0, 1]
    pcd.colors = o3d.utility.Vector3dVector(rgb_array / 255.0)
    # add normals
    # pcd.normals = o3d.utility.Vector3dVector(np.zeros_like(xyz_array))
    # o3d.visualization.draw_geometries([pcd])

    return pcd


def o3d2colmapP3d(pcd_o3d, Points3D):
    # get original xyz, rgb
    xyz_array = np.asarray(pcd_o3d.points)
    rgb_array = np.asarray(pcd_o3d.colors) * 255.0

    # rgb
    rgb_array = np.round(rgb_array).astype(np.uint8)

    # convert to Colmap Points3D
    for i, (point_id, point) in enumerate(points3D.items()):
        new_point3D = Point3D(
            id=point.id,
            xyz=xyz_array[i],
            rgb=rgb_array[i],
            error=point.error,
            image_ids=point.image_ids,
            point2D_idxs=point.point2D_idxs,
        )
        points3D[point_id] = new_point3D

    # return Points3D


def poses_to_pcd(poses):
    pcd = o3d.geometry.PointCloud()
    xyz = np.zeros((len(poses), 3))
    xyz = [pose[:3, 3] for pose in poses]
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.paint_uniform_color([0, 1, 0])
    return pcd


def poses_vis(pose_list, trans_mat=np.eye(4), pcd_scene=o3d.geometry.PointCloud()):

    poses = copy.deepcopy(pose_list)
    pose_show = []
    # pose_show.append(o3d.geometry.TriangleMesh.create_coordinate_frame(2))
    scale = 0.5
    pose_cur = o3d.geometry.TriangleMesh.create_coordinate_frame(0.5 * scale)
    pose_cur.transform(poses[0])
    pose_show.append(pose_cur)

    for i in range(1, len(poses)):
        pose_cur = o3d.geometry.TriangleMesh.create_coordinate_frame(0.2 * scale)
        pose_cur.transform(poses[i])
        pose_show.append(pose_cur)

    # # show neighbor lines
    # lineset = pcd_tools.create_neighbor_lineset(poses)
    # pose_show.append(lineset)

    # show positions
    pcd = poses_to_pcd(poses)
    pose_show.append(pcd)

    pose_show.append(pcd_scene)

    coor_sfm = o3d.geometry.TriangleMesh.create_coordinate_frame(0.5 * scale)
    coor_ori = o3d.geometry.TriangleMesh.create_coordinate_frame(1.5 * scale)
    coor_ori.transform(np.linalg.inv(trans_mat))
    pose_show.append(coor_sfm)
    pose_show.append(coor_ori)
    # pose_show.append(coor_cam)

    o3d.visualization.draw_geometries(pose_show, "pose_show")


if __name__ == "__main__":
    # ********** you need to modify the dir include data procceed by colmap **********
    dir_model = "videos/piper76"
    dir_data = f"{dir_model}/sparse/0_ori"
    dir_data_new = f"{dir_model}/sparse/0"
    print(dir_data_new)
    if not os.path.exists(dir_data_new):
        os.makedirs(dir_data_new)
        print("make new folder successfully!!!")

    # ********** read data **********
    cameras, images, points3D = read_model(dir_data, ".bin")
    # convert colmap points3D to o3d pcd
    pcd_o3d = colmapP3d2o3d(points3D)
    o3d.io.write_point_cloud(
        f"{dir_data_new}/points3D_o3d_ori.ply",
        pcd_o3d,
        write_ascii=True,
    )
    # ********** modify data **********
    # scale and transform matrix
    mat_scale = np.eye(4)
    trans_mat = np.eye(4)  # identity matrix
    # 机械臂
    matrix = np.array([
    [ 0.118498,  0.011758, -0.028000, -0.089684],
    [ 0.029933, -0.026174,  0.115685, -0.383849],
    [ 0.005129, -0.118915, -0.028232,  0.537512],
    [ 0.000000,  0.000000,  0.000000,  1.000000]
])

    R = matrix[0:3, 0:3]
    T = matrix[0:3, 3]
    # 进行奇异值分解
    U, Σ, Vt = np.linalg.svd(R)

    # 计算旋转矩阵
    rotation = U.dot(Vt)

    matrix_new = np.eye(4)
    matrix_new[0:3,0:3] = rotation
    matrix_new[0:3,3] = T/Σ
    print(matrix_new)

    trans_mat = matrix_new

    # scale = 0.10396246
    mat_scale[0, 0] = Σ[0]
    mat_scale[1, 1] = Σ[1]
    mat_scale[2, 2] = Σ[2]

    print(mat_scale)
    print(trans_mat)

    poses = []
    for idx, key in enumerate(images):
        extr = images[key]
        # get original pose
        R = qvec2rotmat(extr.qvec)
        T = np.array(extr.tvec)

        # transform pose
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = T
        pose = np.linalg.inv(pose)
        # print("ori_pose: \n", pose)
        pose = trans_mat @ pose
        # print("pose1: \n", pose)


        # scale pose
        pose_temp = mat_scale @ pose
        # print("scale_pose: \n", pose_temp)
        pose[:3, 3] = pose_temp[:3, 3]
        # print("pose2: \n", pose)

        # save pose
        poses.append(pose)
        # print(pose)
        # print("inv: ")
        # print(np.linalg.inv(pose))

        # update image in colmap data
        pose_back = np.linalg.inv(pose)
        # print("pose_back:\n", pose_back)
        # print("-----------------------------")
        image_new = Image(
            camera_id=images[key].camera_id,
            id=images[key].id,
            name=images[key].name,
            point3D_ids=images[key].point3D_ids,
            qvec=rotmat2qvec(pose_back[:3, :3]),
            tvec=pose_back[:3, 3],
            xys=images[key].xys,
        )
        images[key] = image_new
        

    # tansform and scale points
    pcd_o3d.transform(mat_scale @ trans_mat)

    # visualize poses
    poses_vis(poses, pcd_scene=pcd_o3d)

    print(len(pcd_o3d.points))
    print(len(pcd_o3d.colors))

    # # save new pcd
    # o3d.io.write_point_cloud(
    #     f"{dir_data_new}/points3D.ply",
    #     pcd_o3d,
    #     write_ascii=True,
    # )

    # *********** save new data ***********
    # convert o3d pcd to colmap points3D
    o3d2colmapP3d(pcd_o3d, points3D)

    # update colmap data
    write_model(cameras, images, points3D, path=dir_data_new, ext=".bin")

    # read new data
    cameras, images, points3D_new = read_model(dir_data_new, ".bin")
    pcd_o3d_new = colmapP3d2o3d(points3D_new)
    o3d.visualization.draw_geometries([pcd_o3d_new], "pcd_o3d_new")
