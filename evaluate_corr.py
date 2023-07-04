# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''Evaluation script for DIF-Net.
'''

import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import yaml
import numpy as np
import utils
import torch
from flexfield import FlexField
import trimesh
from scipy.io import loadmat
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# config_files = ["configs/eval_cloud/50002.yml",
#                 "configs/eval_cloud/50004.yml",
#                 "configs/eval_cloud/50007.yml",
#                 "configs/eval_cloud/50009.yml",
#                 "configs/eval_cloud/50020.yml",
#                 "configs/eval_cloud/50021.yml",
#                 "configs/eval_cloud/50022.yml",
#                 "configs/eval_cloud/50025.yml",
#                 "configs/eval_cloud/50026.yml",
#                 "configs/eval_cloud/50027.yml"]
# config_files = ["configs/eval/bear3EP.yml"]
config_files = ["configs/eval/50002.yml"]

total_error_list = []
with torch.no_grad():
    for config_file in config_files:
        with open(config_file,'r') as stream:
            meta_params = yaml.safe_load(stream)

        meta_params['expand'] = 0

        # define DIF-Net
        model = FlexField(**meta_params)
        state_dict=torch.load(meta_params['checkpoint_path'])
        filtered_state_dict={k:v for k,v in state_dict.items() if k.find('detach')==-1}
        model.load_state_dict(filtered_state_dict)
        model.cuda()

        # create save path
        root_path = os.path.join(meta_params['logging_root'], meta_params['experiment_name'])
        utils.cond_mkdir(root_path)


        # load names for evaluation subjects
        with open(meta_params['eval_corr_split'],'r') as file:
            all_names = file.read().split('\n')
        all_names = [name for name in all_names if len(name)>3]
        dataset_list = []
        save_path_list = []
        # optimize latent code for each test subject
        embedding_list = []
        trans_list = []
        verts_list = []
        for id,file in enumerate(all_names):
            save_path = os.path.join(root_path,file)
            embedding_file_name = os.path.join(save_path,"embedding.txt")
            trans_file_name = os.path.join(save_path,"trans.txt")
            embedding_list.append(np.loadtxt(embedding_file_name))
            trans_list.append(np.loadtxt(trans_file_name))
            # mesh_path = os.path.join(os.path.dirname(meta_params['point_cloud_path']),"scaled_mesh",file+'.obj')
            mesh_path = os.path.join(meta_params['mesh_path'],file+'.obj')
            if os.path.exists(mesh_path):
                mesh=trimesh.load(mesh_path,process=False)
            else:
                mesh=trimesh.load(mesh_path.replace('.obj','.ply'))
            verts_list.append(mesh.vertices)

        embedding_list = torch.tensor(embedding_list,dtype=torch.float32).cuda()
        trans_list = torch.tensor(trans_list,dtype=torch.float32).cuda()
        verts_list = torch.tensor(verts_list,dtype=torch.float32).cuda()

        pair_idx, total_coord_idx = model.eval_correspondence(embedding_list,verts_list+trans_list.unsqueeze(1),pair_num_max=2500)
        single_shape_error = []
        for shape_id, file in enumerate(tqdm(all_names)):
            distance_file_name = os.path.join(meta_params["geodesic_distance_path"],file+".mat")
            geodesic_matrix = torch.tensor(loadmat(distance_file_name)["D"]).cuda()
            geodesic_max = geodesic_matrix.max()
            coord_idx = total_coord_idx[pair_idx[:,1]==shape_id,:]
            error = geodesic_matrix[torch.arange(geodesic_matrix.shape[0],device=geodesic_matrix.device)[None,:].expand_as(coord_idx),coord_idx]/geodesic_max
            single_shape_error.append(error.mean(dim=-1).detach().cpu().numpy().tolist())
        print(np.mean(single_shape_error))
        total_error_list.extend(single_shape_error)
    print(np.mean(total_error_list))
    

