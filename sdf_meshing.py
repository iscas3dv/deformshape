'''Create mesh from SDF
'''

import logging
import numpy as np
import plyfile
import skimage.measure
import time
import torch
from collections import OrderedDict

'''Adapted from the DeepSDF repository https://github.com/facebookresearch/DeepSDF
'''

def create_mesh(model, filename,subject_idx=0, embedding=None, N=128, max_batch=64 ** 3, offset=None, scale=None,level=0.0,get_color=True):
    start = time.time()
    ply_filename = filename

    model.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0
    if embedding is None:
        if subject_idx<0:
            embedding=model.template_code[None,:]
        else:
            subject_idx = torch.Tensor([subject_idx]).squeeze().long().cuda()[None,...]
            embedding = model.get_latent_code(subject_idx)
            print(subject_idx.shape,embedding.shape)

    while head < num_samples:
        print(head)
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()[None,...]
        samples[head : min(head + max_batch, num_samples), 3] = (
            model.inference(sample_subset,embedding)
            .squeeze()#.squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch

    # head = 0

    # while head < num_samples:
    #     print(head)
    #     sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()[None,...]
    #     samples[head : min(head + max_batch, num_samples), 3] = (
    #         model.recurrent_inference(sample_subset,subject_idx)
    #         .squeeze()#.squeeze(1)
    #         .detach()
    #         .cpu()
    #     )
    #     head += max_batch
    # embedding = model.get_latent_code(torch.Tensor([subject_idx]).squeeze().long().cuda()[None,...])

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print("sampling takes: %f" % (end - start))
    if not get_color:
        return convert_sdf_samples_to_ply(
                    sdf_values.data.cpu(),
                    voxel_origin,
                    voxel_size,
                    ply_filename + ".ply",
                    offset,
                    scale,
                    level
                )
    else:
        convert_sdf_samples_with_color_to_ply(
            model,
            subject_idx,
            embedding,
            sdf_values.data.cpu(),
            voxel_origin,
            voxel_size,
            ply_filename + ".ply",
            offset,
            scale,
            level,
            False
        )
    
def warped_mesh(model, filename,subject_idx=0,target_idx=0, N=128, max_batch=64 ** 3, offset=None, scale=None,level=0.0):
    start = time.time()
    ply_filename = filename

    model.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    subject_idx = torch.Tensor([subject_idx]).squeeze().long().cuda()[None,...]
    embedding = model.get_latent_code(subject_idx)

    head = 0

    while head < num_samples:
        print(head)
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()[None,...]
        samples[head : min(head + max_batch, num_samples), 3] = (
            model.inference_stage1(sample_subset,subject_idx)
            .squeeze()#.squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print("sampling takes: %f" % (end - start))

    start_time = time.time()

    numpy_3d_sdf_tensor = sdf_values.data.cpu().numpy()

    verts, faces, normals, values = skimage.measure.marching_cubes(
            numpy_3d_sdf_tensor, level=level, spacing=[voxel_size] * 3
        )


    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_origin[2] + verts[:, 2]

    target_idx = torch.Tensor([target_idx]).squeeze().long().cuda()[None,...]
    max_batch=64 ** 3

    head = 0
    warp_points = np.zeros_like(mesh_points)
    num_samples = mesh_points.shape[0]
    while head < num_samples:
        print(head)
        sample_subset = torch.from_numpy(mesh_points[head : min(head + max_batch, num_samples), 0:3]).float().cuda()[None,...]

        points = (
            model.warp_shape(sample_subset,subject_idx,target_idx)
            .squeeze()#.squeeze(1)
            .detach()
            .cpu()
        )
        warp_points[head : min(head + max_batch, num_samples), 0:3]=points
        
        head += max_batch

    mesh_points=warp_points
    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_filename_out=ply_filename + ".ply"
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )

def create_template_mesh(model, filename, N=128, max_batch=64 ** 3, offset=None, scale=None,level=0.0, get_color=True):
    start = time.time()
    ply_filename = filename

    model.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        print(head)
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()[None,...]
        samples[head : min(head + max_batch, num_samples), 3] = (
            model.inference_template(sample_subset)
            .squeeze()#.squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print("sampling takes: %f" % (end - start))


    convert_template_sdf_samples_with_color_to_ply(
        model,
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        offset,
        scale,
        level
    )

def get_mesh_verts(mesh_points,embedding,model):
    model.eval()
    mesh_verts = np.zeros_like(mesh_points)
    num_samples = mesh_points.shape[0]

    print('num_samples',num_samples)
    max_batch=64 ** 3


    head = 0
    while head < num_samples:
        print(head)
        sample_subset = torch.from_numpy(mesh_points[head : min(head + max_batch, num_samples), 0:3]).float().cuda()[None,...]
        mesh_verts[head : min(head + max_batch, num_samples), 0:3] = (
                model.get_template_coords(sample_subset,embedding)
                .squeeze()#.squeeze(1)
                .detach()
                .cpu()
            )
        return mesh_verts

def get_mesh_color(mesh_points,subject_idx,embedding,model,part_color=False):

    model.eval()
    rgb_list=np.array([[255,255,255],[246, 18, 18], [253, 53, 31], [252, 58, 10], [248, 108, 48], [243, 112, 25], [247, 139, 30], [250, 167, 41], [249, 190, 54], [251, 203, 9], [254, 230, 15], [250, 250, 46], [231, 252, 44], [212, 254, 45], [185, 254, 24], [170, 251, 48], [128, 250, 6], [103, 251, 4], [106, 245, 47], [69, 250, 24], [64, 251, 44], [19, 245, 19], [7, 248, 31], [20, 251, 66], [17, 247, 86], [38, 246, 121], [30, 253, 141], [52, 251, 171], [12, 252, 180], [5, 251, 202], [45, 248, 228],\
        [246, 18, 18], [253, 53, 31], [252, 58, 10], [248, 108, 48], [243, 112, 25], [247, 139, 30], [250, 167, 41], [249, 190, 54], [251, 203, 9], [254, 230, 15], [250, 250, 46], [231, 252, 44], [212, 254, 45], [185, 254, 24], [170, 251, 48], [128, 250, 6], [103, 251, 4], [106, 245, 47], [69, 250, 24], [64, 251, 44], [19, 245, 19], [7, 248, 31], [20, 251, 66], [17, 247, 86], [38, 246, 121], [30, 253, 141], [52, 251, 171], [12, 252, 180], [5, 251, 202], [45, 248, 228]],\
        dtype=np.float32)/255
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    # mesh_points.requires_grad = False
    # rgb_list=np.array([[243, 38, 38], [249, 33, 19], [247, 38, 10], [254, 59, 16], [248, 100, 53], [244, 97, 34], [249, 121, 50], [251, 125, 33], [249, 140, 40], [246, 152, 43], [254, 162, 23], [247, 169, 18], [254, 184, 5], [245, 204, 59], [243, 207, 19], [254, 232, 30], [252, 243, 20], [241, 245, 43], [233, 253, 10], [219, 249, 30], [210, 251, 49], [191, 245, 39], [186, 254, 42], [173, 251, 47], [156, 246, 40], [148, 246, 50], [124, 253, 23], [123, 246, 48], [111, 244, 49], [94, 248, 40], [73, 246, 29], [51, 246, 20], [67, 251, 51], [12, 248, 7], [44, 250, 52], [47, 250, 67], [35, 247, 69], [14, 246, 65], [25, 252, 89], [15, 248, 94], [6, 254, 106], [47, 253, 142], [39, 245, 146], [40, 247, 160], [41, 251, 175], [17, 253, 182], [6, 254, 194], [32, 244, 206], [28, 245, 219], [35, 247, 234], [51, 248, 248], [4, 239, 253], [31, 226, 252], [20, 210, 252], [42, 199, 248], [11, 180, 252], [51, 176, 247], [26, 153, 244], [31, 144, 249], [32, 133, 250], [25, 112, 244], [38, 109, 245], [11, 79, 253], [1, 57, 253], [4, 44, 252], [40, 61, 252], [28, 37, 246], [10, 5, 250], [61, 45, 248], [67, 38, 245], [67, 20, 253], [89, 34, 246], [108, 39, 253], [102, 12, 249], [138, 48, 254], [130, 10, 250], [161, 46, 251], [167, 32, 250], [174, 20, 247], [188, 21, 247], [201, 32, 244], [219, 6, 254], [228, 21, 246], [247, 48, 251], [251, 4, 241], [249, 14, 225], [250, 32, 215], [247, 52, 205], [245, 9, 179], [246, 40, 176], [254, 23, 162], [254, 46, 158], [249, 55, 148], [244, 38, 124], [254, 33, 112], [246, 12, 82], [243, 18, 72], [254, 8, 52], [244, 28, 54], [251, 15, 29]],\
    #     dtype=np.float32)/255
    mesh_colors = np.zeros_like(mesh_points)
    num_samples = mesh_points.shape[0]

    print('num_samples',num_samples)
    max_batch=64 ** 3


    head = 0
    while head < num_samples:
        print(head)
        sample_subset = torch.from_numpy(mesh_points[head : min(head + max_batch, num_samples), 0:3]).float().cuda()[None,...]

        if part_color:
            part_id = (
                model.get_part_id(sample_subset,embedding)
                .squeeze()#.squeeze(1)
                .detach()
                .cpu()
            )
            mesh_colors[head : min(head + max_batch, num_samples), 0:3]=rgb_list[part_id]
        else:
            mesh_colors[head : min(head + max_batch, num_samples), 0:3] = (
                model.get_template_coords(sample_subset,embedding)
                .squeeze()#.squeeze(1)
                .detach()
                .cpu()
            )
        # else:
        #     error1=(
        #         model.get_alignment_error(sample_subset,embedding,time_step)
        #     )
        #     error1=(error1.squeeze()#.squeeze(1)
        #         .detach()
        #         .cpu()
        #     )


        #     #c=((error-min_val)/(max_val-min_val))[:,None]
        #     c1=(error1/0.03).clamp(max=1)[:,None]

        #     mesh_colors[head : min(head + max_batch, num_samples), 0:3]=torch.cat([c1,torch.zeros_like(c1),1-c1],dim=1)
        head += max_batch

    mesh_colors = np.clip(mesh_colors/2+0.5,0,1) # normalize color to 0-1

    return mesh_colors


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
    level=0.0,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    # verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    # try:
    #     verts, faces, normals, values = skimage.measure.marching_cubes(
    #         numpy_3d_sdf_tensor, level=level, spacing=[voxel_size] * 3
    #     )
    # except:
    #     pass
    verts, faces, normals, values = skimage.measure.marching_cubes(
            numpy_3d_sdf_tensor, level=level, spacing=[voxel_size] * 3
        )
    # verts, faces = marching_cubes(pytorch_3d_sdf_tensor.transpose(0,2).contiguous(), 0)
    # verts*=voxel_size
    # verts=verts.cpu().numpy()

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )
    return verts


def convert_sdf_samples_with_color_to_ply(
    model,
    subject_idx,
    embedding,
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
    level=0.0,
    part_color=True,
):
    """
    Convert sdf samples to .ply with color-coded template coordinates

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes(
            numpy_3d_sdf_tensor, level=level, spacing=[voxel_size] * 3
        )
    except:
        pass

    # verts, faces = marching_cubes(pytorch_3d_sdf_tensor.transpose(0,2).contiguous(), 0)
    # verts*=voxel_size
    # verts=verts.cpu().numpy()

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset


    mesh_colors = get_mesh_color(mesh_points,subject_idx,embedding,model,part_color)
    mesh_colors = np.clip(mesh_colors*255,0,255).astype(np.uint8)

    #trans_verts = get_mesh_verts(mesh_points,embedding,model)

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    #trans_verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    colors_tuple = np.zeros((num_verts,), dtype=[("red", "u1"), ("green", "u1"), ("blue", "u1")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])
    
    # for i in range(0, num_verts):
    #     trans_verts_tuple[i] = tuple(trans_verts[i, :])

    for i in range(0, num_verts):
        colors_tuple[i] = tuple(mesh_colors[i, :])

    verts_all = np.empty(num_verts,verts_tuple.dtype.descr + colors_tuple.dtype.descr)
    #trans_verts_all = np.empty(num_verts,trans_verts_tuple.dtype.descr + colors_tuple.dtype.descr)

    for prop in verts_tuple.dtype.names:
        verts_all[prop] = verts_tuple[prop]

    for prop in colors_tuple.dtype.names:
        verts_all[prop] = colors_tuple[prop]

    # for prop in trans_verts_tuple.dtype.names:
    #     trans_verts_all[prop] = trans_verts_tuple[prop]

    # for prop in colors_tuple.dtype.names:
    #     trans_verts_all[prop] = colors_tuple[prop]


    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_all, "vertex")
    #el_trans_verts = plyfile.PlyElement.describe(trans_verts_all, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces],text=True)
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    # file_name=ply_filename_out.replace('.ply','_translation.ply')
    # print(file_name)
    # ply_data = plyfile.PlyData([el_trans_verts, el_faces],text=True)
    # logging.debug("saving translation mesh to %s" % (file_name))
    # ply_data.write(file_name)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )

def convert_template_sdf_samples_with_color_to_ply(
    model,
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
    level=0.0,
):
    """
    Convert sdf samples to .ply with color-coded template coordinates

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes(
            numpy_3d_sdf_tensor, level=level, spacing=[voxel_size] * 3
        )
    except:
        pass

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset


    mesh_colors = get_template_color(mesh_points,model)
    mesh_colors = np.clip(mesh_colors*255,0,255).astype(np.uint8)

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    colors_tuple = np.zeros((num_verts,), dtype=[("red", "u1"), ("green", "u1"), ("blue", "u1")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    for i in range(0, num_verts):
        colors_tuple[i] = tuple(mesh_colors[i, :])

    verts_all = np.empty(num_verts,verts_tuple.dtype.descr + colors_tuple.dtype.descr)

    for prop in verts_tuple.dtype.names:
        verts_all[prop] = verts_tuple[prop]

    for prop in colors_tuple.dtype.names:
        verts_all[prop] = colors_tuple[prop]


    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_all, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces],text=True)
    
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )

def get_template_color(mesh_points,model):

    model.eval()
    rgb_list=np.array([[255,255,255],[246, 18, 18], [253, 53, 31], [252, 58, 10], [248, 108, 48], [243, 112, 25], [247, 139, 30], [250, 167, 41], [249, 190, 54], [251, 203, 9], [254, 230, 15], [250, 250, 46], [231, 252, 44], [212, 254, 45], [185, 254, 24], [170, 251, 48], [128, 250, 6], [103, 251, 4], [106, 245, 47], [69, 250, 24], [64, 251, 44], [19, 245, 19], [7, 248, 31], [20, 251, 66], [17, 247, 86], [38, 246, 121], [30, 253, 141], [52, 251, 171], [12, 252, 180], [5, 251, 202], [45, 248, 228],\
        [246, 18, 18], [253, 53, 31], [252, 58, 10], [248, 108, 48], [243, 112, 25], [247, 139, 30], [250, 167, 41], [249, 190, 54], [251, 203, 9], [254, 230, 15], [250, 250, 46], [231, 252, 44], [212, 254, 45], [185, 254, 24], [170, 251, 48], [128, 250, 6], [103, 251, 4], [106, 245, 47], [69, 250, 24], [64, 251, 44], [19, 245, 19], [7, 248, 31], [20, 251, 66], [17, 247, 86], [38, 246, 121], [30, 253, 141], [52, 251, 171], [12, 252, 180], [5, 251, 202], [45, 248, 228]],\
        dtype=np.float32)/255
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    # mesh_points.requires_grad = False
    # rgb_list=np.array([[243, 38, 38], [249, 33, 19], [247, 38, 10], [254, 59, 16], [248, 100, 53], [244, 97, 34], [249, 121, 50], [251, 125, 33], [249, 140, 40], [246, 152, 43], [254, 162, 23], [247, 169, 18], [254, 184, 5], [245, 204, 59], [243, 207, 19], [254, 232, 30], [252, 243, 20], [241, 245, 43], [233, 253, 10], [219, 249, 30], [210, 251, 49], [191, 245, 39], [186, 254, 42], [173, 251, 47], [156, 246, 40], [148, 246, 50], [124, 253, 23], [123, 246, 48], [111, 244, 49], [94, 248, 40], [73, 246, 29], [51, 246, 20], [67, 251, 51], [12, 248, 7], [44, 250, 52], [47, 250, 67], [35, 247, 69], [14, 246, 65], [25, 252, 89], [15, 248, 94], [6, 254, 106], [47, 253, 142], [39, 245, 146], [40, 247, 160], [41, 251, 175], [17, 253, 182], [6, 254, 194], [32, 244, 206], [28, 245, 219], [35, 247, 234], [51, 248, 248], [4, 239, 253], [31, 226, 252], [20, 210, 252], [42, 199, 248], [11, 180, 252], [51, 176, 247], [26, 153, 244], [31, 144, 249], [32, 133, 250], [25, 112, 244], [38, 109, 245], [11, 79, 253], [1, 57, 253], [4, 44, 252], [40, 61, 252], [28, 37, 246], [10, 5, 250], [61, 45, 248], [67, 38, 245], [67, 20, 253], [89, 34, 246], [108, 39, 253], [102, 12, 249], [138, 48, 254], [130, 10, 250], [161, 46, 251], [167, 32, 250], [174, 20, 247], [188, 21, 247], [201, 32, 244], [219, 6, 254], [228, 21, 246], [247, 48, 251], [251, 4, 241], [249, 14, 225], [250, 32, 215], [247, 52, 205], [245, 9, 179], [246, 40, 176], [254, 23, 162], [254, 46, 158], [249, 55, 148], [244, 38, 124], [254, 33, 112], [246, 12, 82], [243, 18, 72], [254, 8, 52], [244, 28, 54], [251, 15, 29]],\
    #     dtype=np.float32)/255
    mesh_colors = np.zeros_like(mesh_points)
    num_samples = mesh_points.shape[0]

    print('num_samples',num_samples)
    max_batch=64 ** 3


    head = 0
    while head < num_samples:
        print(head)
        sample_subset = torch.from_numpy(mesh_points[head : min(head + max_batch, num_samples), 0:3]).float().cuda()[None,...]

        part_id = (
            model.get_template_part_id(sample_subset)
            .squeeze()#.squeeze(1)
            .detach()
            .cpu()
        )
        mesh_colors[head : min(head + max_batch, num_samples), 0:3]=rgb_list[part_id]
        
        head += max_batch

    mesh_colors = np.clip(mesh_colors/2+0.5,0,1) # normalize color to 0-1

    return mesh_colors