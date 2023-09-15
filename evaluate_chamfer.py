import os
import torch
import yaml
from calculate_chamfer_distance import compute_chamfer
import trimesh
from tqdm import tqdm
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

config_files = ["configs/generate_all/50002.yml"]

total_error_list = []
for config_file in config_files:
    with open(config_file,'r') as stream:
        meta_params = yaml.safe_load(stream)
    fitting_results_path = os.path.join(meta_params["logging_root"],meta_params["experiment_name"])
    gt_mesh_path = meta_params["mesh_path"]
    with open(meta_params['eval_split'],'r') as file:
        all_names = file.read().split('\n')
        all_names = [name for name in all_names if len(name)>3]
    error_list = []
    for name in tqdm(all_names,mininterval=1):
        fitted_shape = trimesh.load(os.path.join(fitting_results_path,name,"mesh.ply"))
        if os.path.exists(os.path.join(gt_mesh_path,name+".obj")):
            gt_shape = trimesh.load(os.path.join(gt_mesh_path,name+".obj"))
        else:
            gt_shape = trimesh.load(os.path.join(gt_mesh_path,name+".ply"))
        error_list.append(compute_chamfer(fitted_shape.vertices,gt_shape.vertices))
    print(len(error_list))
    print(np.mean(error_list))
    total_error_list.extend(error_list)
print("total:",np.mean(total_error_list))

