import os
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import math
from mesh_to_sdf import get_surface_point_cloud, scale_to_unit_sphere, sample_sdf_near_surface, utils
from mesh_to_sdf.scan import Scan, get_camera_transform_looking_at_origin
import trimesh
import scipy.io
from tqdm import tqdm
import argparse
import skimage, skimage.measure

def get_train_split(train_split_path, animal_kind):
    train_split_file = open(os.path.join(train_split_path, "{}.txt".format(animal_kind)), 'r')
    train_split = []
    for line in train_split_file.readlines():
        train_split.append(line.strip())
    train_split_file.close()
    return train_split

def get_test_split(test_split_path, animal_kind):
    test_split_file = open(os.path.join(test_split_path, "{}.txt".format(animal_kind)), 'r')
    test_split = []
    for line in test_split_file.readlines():
        test_split.append(line.strip())
    test_split_file.close()
    return test_split

def scale_to_unit_sphere_global(mesh_cur, scale):
    if isinstance(mesh_cur, trimesh.Scene):
        mesh_cur = mesh_cur.dump().sum()
    # if isinstance(mesh_all, trimesh.Scene):
    #     mesh_all = mesh_all.dump().sum()
    vertices = mesh_cur.vertices
    vertices /= scale
    return trimesh.Trimesh(vertices=vertices, faces=mesh_cur.faces, process=False)


def write_data(shapes_info, output_folder, human_kind, mesh_all_vertices, scale):
    for shape_info in tqdm(shapes_info):
        obj_file_name = shape_info['obj_file_name']
        output_folder_in_sphere = os.path.join(output_folder, human_kind, 'free_space_pts')
        output_folder_near_surface = os.path.join(output_folder, human_kind, 'surface_pts_n_normal')
        output_folder_scaled_mesh = os.path.join(output_folder, human_kind, 'mesh')
        output_folder_geodesic_mesh = os.path.join(output_folder, human_kind, 'mesh_for_geodesic_distance')
        output_folder_partial_points = os.path.join(output_folder, human_kind, 'partial_points')
        os.makedirs(output_folder_in_sphere, exist_ok=True)
        os.makedirs(output_folder_near_surface, exist_ok=True)
        os.makedirs(output_folder_scaled_mesh, exist_ok=True)
        os.makedirs(output_folder_geodesic_mesh, exist_ok=True)
        os.makedirs(output_folder_partial_points, exist_ok=True)
        output_name_in_sphere = os.path.join(output_folder_in_sphere, '{}.mat'.format(obj_file_name))
        output_name_near_surface = os.path.join(output_folder_near_surface, '{}.mat'.format(obj_file_name))
        output_name_scaled_mesh = os.path.join(output_folder_scaled_mesh, '{}.obj'.format(obj_file_name))
        output_name_geodesic_mesh = os.path.join(output_folder_geodesic_mesh, '{}.ply'.format(obj_file_name))
        output_name_partial_points = os.path.join(output_folder_partial_points, '{}.mat'.format(obj_file_name))

        if os.path.exists(output_name_in_sphere):
            continue
        vertices = mesh_all_vertices[shape_info['idx'][0]:shape_info['idx'][1]]
        mesh = trimesh.Trimesh(vertices=vertices, faces=shape_info['faces'], process=False)
        mesh_scale = scale_to_unit_sphere_global(mesh, scale)
        #1. scaled mesh
        mesh_scale.export(output_name_scaled_mesh)

        #2. sdf
        cloud = get_surface_point_cloud(mesh_scale, surface_point_method='scan', scan_count=100, scan_resolution=1600,
                                        bounding_radius=1, )
        vertices_free, sdf = cloud.sample_sdf_near_surface(500000, sign_method='depth')
        points_in_sphere = np.concatenate([vertices_free, sdf[:, None]], axis=1)
        vertices_surface, normals = cloud.get_random_surface_points(500000)
        scipy.io.savemat(output_name_in_sphere, {'p_sdf': points_in_sphere})
        points_near_surface = np.concatenate([vertices_surface, normals], axis=1)
        scipy.io.savemat(output_name_near_surface, {'p': points_near_surface})

        #3. partial points
        bounding_radius = 1
        camera_transform = get_camera_transform_looking_at_origin(0, math.pi / 2, camera_distance=2 * bounding_radius)
        scan = Scan(mesh_scale,
                    camera_transform=camera_transform,
                    resolution=400,
                    calculate_normals=True,
                    fov=1.0472,
                    z_near=bounding_radius * 1,
                    z_far=bounding_radius * 3
                    )
        vertices_surface, normals = scan.points, scan.normals
        points_near_surface = np.concatenate([vertices_surface, normals], axis=1)
        scipy.io.savemat(output_name_partial_points, {'p': points_near_surface})

        #4. mesh for geodesic distance
        mesh_scale.remove_degenerate_faces()
        m_split_list = mesh_scale.split(only_watertight=False).tolist()
        m_split_list.sort(key=lambda x: x.vertices.shape[0], reverse=True)
        mesh_split = m_split_list[0]
        ply_mesh = trimesh.exchange.ply.export_ply(mesh_split, encoding='ascii')
        output_file = open(output_name_geodesic_mesh, "wb+")
        output_file.write(ply_mesh)
        output_file.close()


def generate_train_data(dataset_folder, output_folder, human_kind):
    train_split_path = "./split/train/"
    train_split = get_train_split(train_split_path, human_kind)

    vertices_all_list = []
    shapes_info = []
    point_num = 0
    for action in sorted(os.listdir(dataset_folder)):
        if not action.startswith(human_kind):
            continue
        human_kind_action_path = os.path.join(dataset_folder, action)
        for human_action_obj_file in sorted(os.listdir(human_kind_action_path)):
            if human_action_obj_file[:-4] in train_split:
                m = trimesh.load(os.path.join(human_kind_action_path, human_action_obj_file), process=False)
                vertices_centroid = m.vertices - m.bounding_box.centroid
                nv = vertices_centroid.shape[0]
                start_idx = point_num
                point_num += nv
                vertices_all_list.append(vertices_centroid)
                shape_info = {'action': action,
                              'obj_file_name': human_action_obj_file[:-4],
                              'faces': m.faces,
                              'idx': [start_idx, point_num]}
                shapes_info.append(shape_info)
    mesh_all_vertices = np.array(vertices_all_list).reshape(point_num, 3)
    scale = np.max(np.linalg.norm(mesh_all_vertices, axis=1))
    print(human_kind)
    write_data(shapes_info, output_folder, human_kind, mesh_all_vertices, scale)


def generate_test_data(dataset_folder, output_folder, human_kind):
    train_split_path = "./split/train/"
    train_split = get_train_split(train_split_path, human_kind)

    test_split_path = "./split/eval/"
    test_split = get_test_split(test_split_path, human_kind)

    vertices_all_list = []
    vertices_all_list_scale = []
    shapes_info = []
    point_num = 0
    for action in sorted(os.listdir(dataset_folder)):
        if not action.startswith(human_kind):
            continue
        human_kind_action_path = os.path.join(dataset_folder, action)
        for human_action_obj_file in sorted(os.listdir(human_kind_action_path)):
            m = trimesh.load(os.path.join(human_kind_action_path, human_action_obj_file), process=False)
            vertices_centroid = m.vertices - m.bounding_box.centroid
            nv = vertices_centroid.shape[0]
            if human_action_obj_file[:-4] in train_split:
                vertices_all_list_scale.append(vertices_centroid)
            if human_action_obj_file[:-4] in test_split:
                vertices_all_list.append(vertices_centroid)
                start_idx = point_num
                point_num += nv
                shape_info = {'action': action,
                              'obj_file_name': human_action_obj_file[:-4],
                              'faces': m.faces,
                              'idx': [start_idx, point_num]}
                shapes_info.append(shape_info)

    mesh_all_vertices_scale = np.array(vertices_all_list_scale).reshape(-1, 3)
    scale = np.max(np.linalg.norm(mesh_all_vertices_scale, axis=1))
    mesh_all_vertices = np.array(vertices_all_list).reshape(point_num, 3)
    print(human_kind)
    write_data(shapes_info, output_folder, human_kind, mesh_all_vertices, scale)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', type=str, default='../../dataset/D-FAUST/mesh')
    parser.add_argument('--output_folder', type=str, default='../../dataset/D-FAUST/{}_sdf_with_scale_noise')
    parser.add_argument('--human_kind', type=str, default='50002')
    parser.add_argument('--mode', type=str, default='train')
    opt = parser.parse_args()
    

    if opt.mode == "train":
        generate_train_data(opt.dataset_folder, opt.output_folder, opt.human_kind)
    else:
        generate_test_data(opt.dataset_folder, opt.output_folder, opt.human_kind)



