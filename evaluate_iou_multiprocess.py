import os
import yaml
import trimesh
from tqdm import tqdm
import numpy as np
import mesh_to_sdf 
import multiprocessing as mul
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# config_files = ["configs/eval_trainset/50002.yml",
#                 "configs/eval_trainset/50004.yml",
#                 "configs/eval_trainset/50007.yml",
#                 "configs/eval_trainset/50009.yml",
#                 "configs/eval_trainset/50020.yml",
#                 "configs/eval_trainset/50021.yml",
#                 "configs/eval_trainset/50022.yml",
#                 "configs/eval_trainset/50025.yml",
#                 "configs/eval_trainset/50026.yml",
#                 "configs/eval_trainset/50027.yml"
#                 ]
config_files = ["configs/eval/50002.yml" ]
total_error_list = []
sample_point_count = 500000
voxel_resolution = 128
for config_file in config_files:
    with open(config_file,'r') as stream:
        meta_params = yaml.safe_load(stream)
    fitting_results_path = os.path.join(meta_params["logging_root"],meta_params["experiment_name"])
    gt_mesh_path = meta_params["mesh_path"]
    with open(meta_params['eval_split'],'r') as file:
        all_names = file.read().split('\n')
        all_names = [name for name in all_names if len(name)>3]
    # error_list = []
    def calculate_iou(idx):
        name=all_names[idx]
        fitted_shape = trimesh.load(os.path.join(fitting_results_path,name,"mesh.ply"))
        fitted_point_cloud=mesh_to_sdf.surface_point_cloud.create_from_scans(
            fitted_shape, bounding_radius=1, 
            scan_count=20, 
            scan_resolution=400, 
            calculate_normals=False)
        if os.path.exists(os.path.join(gt_mesh_path,name+".ply")):
            gt_shape = trimesh.load(os.path.join(gt_mesh_path,name+".ply"))
        else:
            gt_shape = trimesh.load(os.path.join(gt_mesh_path,name+".obj"))
        vert = gt_shape.vertices
        gt_shape = trimesh.Trimesh(vertices=vert,faces=gt_shape.faces)
        gt_point_cloud=mesh_to_sdf.surface_point_cloud.create_from_scans(
            gt_shape, bounding_radius=1, 
            scan_count=20, 
            scan_resolution=400, 
            calculate_normals=False)
        grid_sample = mesh_to_sdf.utils.get_raster_points(voxel_resolution)
        fitted_inner_idx = ~fitted_point_cloud.is_outside(grid_sample)
        gt_inner_idx = ~gt_point_cloud.is_outside(grid_sample)
        iou = float(np.sum((fitted_inner_idx*gt_inner_idx).astype(int)))/float(np.sum((fitted_inner_idx+gt_inner_idx).astype(int)))
        return iou
    pool = mul.Pool(32)
    error_list = pool.map(calculate_iou, list(range(len(all_names))))
    print(len(error_list))
    print(np.mean(error_list))
    total_error_list.extend(error_list)
print("total:",np.mean(total_error_list))

