import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
#os.environ["PYOPENGL_PLATFORM"] = "egl"
import numpy as np
import math
from mesh_to_sdf import get_surface_point_cloud, scale_to_unit_sphere,sample_sdf_near_surface,utils
from mesh_to_sdf.scan import Scan, get_camera_transform_looking_at_origin
import trimesh
import scipy.io
from tqdm import tqdm
import argparse
import skimage, skimage.measure
def anime_read( filename):
    """
    filename: .anime file
    return:
        nf: number of frames in the animation
        nv: number of vertices in the mesh (mesh topology fixed through frames)
        nt: number of triangle face in the mesh
        vert_data: [nv, 3], vertice data of the 1st frame (3D positions in x-y-z-order)
        face_data: [nt, 3], riangle face data of the 1st frame
        offset_data: [nf-1,nv,3], 3D offset data from the 2nd to the last frame
    """
    f = open(filename, 'rb')
    nf = np.fromfile(f, dtype=np.int32, count=1)[0]
    nv = np.fromfile(f, dtype=np.int32, count=1)[0]
    nt = np.fromfile(f, dtype=np.int32, count=1)[0]
    vert_data = np.fromfile(f, dtype=np.float32, count=nv * 3)
    face_data = np.fromfile(f, dtype=np.int32, count=nt * 3)
    offset_data = np.fromfile(f, dtype=np.float32, count=-1)
    vert_data = vert_data.reshape((-1, 3))
    face_data = face_data.reshape((-1, 3))
    offset_data = offset_data.reshape((nf - 1, nv, 3))
    return nf, nv, nt, vert_data, face_data, offset_data

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

def write_data(shapes_info, output_folder, animal_kind, mesh_all_vertices, scale):
    for shape_info in tqdm(shapes_info):
        kind = shape_info['name']
        frame = shape_info['frame']

        output_folder_in_sphere = os.path.join(output_folder, animal_kind, 'free_space_pts')
        output_folder_near_surface = os.path.join(output_folder, animal_kind, 'surface_pts_n_normal')
        output_folder_scaled_mesh = os.path.join(output_folder, animal_kind, 'mesh')
        output_folder_geodesic_mesh = os.path.join(output_folder, animal_kind, 'mesh_for_geodesic_distance')
        output_folder_partial_points = os.path.join(output_folder, animal_kind, 'partial_points')
        os.makedirs(output_folder_in_sphere, exist_ok=True)
        os.makedirs(output_folder_near_surface, exist_ok=True)
        os.makedirs(output_folder_scaled_mesh, exist_ok=True)
        os.makedirs(output_folder_geodesic_mesh, exist_ok=True)
        os.makedirs(output_folder_partial_points, exist_ok=True)
        output_name_in_sphere = os.path.join(output_folder_in_sphere, kind + '_{}.mat'.format(frame))
        output_name_near_surface = os.path.join(output_folder_near_surface, kind + '_{}.mat'.format(frame))
        output_name_scaled_mesh = os.path.join(output_folder_scaled_mesh, kind + '_{}.obj'.format(frame))
        output_name_geodesic_mesh = os.path.join(output_folder_geodesic_mesh, kind + '_{}.ply'.format(frame))
        output_name_partial_points = os.path.join(output_folder_partial_points, kind + '_{}.mat'.format(frame))

        if os.path.exists(output_name_in_sphere):
            continue
        vertices = mesh_all_vertices[shape_info['idx'][0]:shape_info['idx'][1]]
        mesh = trimesh.Trimesh(vertices=vertices, faces=shape_info['faces'], process=False)
        mesh_scale = scale_to_unit_sphere_global(mesh, scale)
        #1. scaled mesh
        mesh_scale.export(output_name_scaled_mesh)

        #2. sdf
        cloud = get_surface_point_cloud(mesh_scale, surface_point_method='scan', scan_count=20, scan_resolution=400,
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




def generate_train_data(dataset_folder, output_folder, animal_kind):
    train_split_path = "../split/train/"
    train_split = get_train_split(train_split_path, animal_kind)

    vertices_all_list = []
    shapes_info = []
    point_num = 0
    for kind in sorted(os.listdir(dataset_folder)):
        if not kind.startswith(animal_kind):
            continue
        animal_kind_anime = os.path.join(dataset_folder, kind, kind + '.anime')
        nf, nv, nt, vert_data, face_data, offset_data = anime_read(animal_kind_anime)
        for frame in range(nf):
            if "{}_{}".format(kind, frame) in train_split:
                vertices = vert_data if frame == 0 else vert_data + offset_data[frame - 1]
                m = trimesh.Trimesh(vertices=vertices, faces=face_data, process=False)
                vertices_centroid = m.vertices - m.bounding_box.centroid
                start_idx = point_num
                point_num += nv
                vertices_all_list.append(vertices_centroid)
                shape_info = {'name': kind,
                              'frame': frame,
                              'faces': m.faces,
                              'idx': [start_idx, point_num]}
                shapes_info.append(shape_info)

    mesh_all_vertices = np.array(vertices_all_list).reshape(point_num, 3)
    scale = np.max(np.linalg.norm(mesh_all_vertices, axis=1))
    print(animal_kind)
    write_data(shapes_info, output_folder, animal_kind, mesh_all_vertices, scale)
def generate_test_data(dataset_folder, output_folder, animal_kind):
    train_split_path = "../split/train/"
    train_split = get_train_split(train_split_path, animal_kind)

    test_split_path = "../split/eval/"
    test_split = get_test_split(test_split_path, animal_kind)

    vertices_all_list = []
    vertices_all_list_scale = []
    shapes_info = []
    point_num = 0
    for kind in sorted(os.listdir(dataset_folder)):
        if not kind.startswith(animal_kind):
            continue
        animal_kind_anime = os.path.join(dataset_folder, kind, kind + '.anime')
        nf, nv, nt, vert_data, face_data, offset_data = anime_read(animal_kind_anime)
        for frame in range(nf):
            vertices = vert_data if frame == 0 else vert_data + offset_data[frame - 1]
            m = trimesh.Trimesh(vertices=vertices, faces=face_data, process=False)
            vertices_centroid = m.vertices - m.bounding_box.centroid
            if "{}_{}".format(kind, frame) in train_split:
                vertices_all_list_scale.append(vertices_centroid)
            if "{}_{}".format(kind, frame) in test_split:
                vertices_all_list.append(vertices_centroid)
                start_idx = point_num
                point_num += nv
                shape_info = {'name': kind,
                              'frame': frame,
                              'faces': m.faces,
                              'idx': [start_idx, point_num]}
                shapes_info.append(shape_info)

    mesh_all_vertices_scale = np.array(vertices_all_list_scale).reshape(-1, 3)
    scale = np.max(np.linalg.norm(mesh_all_vertices_scale, axis=1))
    mesh_all_vertices = np.array(vertices_all_list).reshape(point_num, 3)
    print(animal_kind)
    write_data(shapes_info, output_folder, animal_kind, mesh_all_vertices, scale)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', type=str, default='../../disk/dataset/DeformingThings4D/animals')
    parser.add_argument('--output_folder', type=str, default='../../disk/dataset/DeformingThings4D/animals_sdf_scale')
    parser.add_argument('--animal_kind', type=str, default='deer2MB')
    parser.add_argument('--mode', type=str, default='train')
    opt = parser.parse_args()

    if opt.mode == "train":
        generate_train_data(opt.dataset_folder, opt.output_folder, opt.animal_kind)
    else:
        generate_test_data(opt.dataset_folder, opt.output_folder, opt.animal_kind)



