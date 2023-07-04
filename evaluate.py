# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''Evaluation script for DIF-Net.
'''

import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import yaml
import numpy as np
import dataset, utils

import torch
import configargparse
from torch import embedding
from flexfield import FlexField
import sdf_meshing


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

p = configargparse.ArgumentParser()
p.add_argument('--config', required=True, help='Evaluation configuration')
p.add_argument('--type', required=True, help='fit or generate')

# load configs
opt = p.parse_args()
with open(os.path.join(opt.config),'r') as stream:
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
with open(meta_params['eval_split'],'r') as file:
    all_names = file.read().split('\n')
all_names = [name for name in all_names if len(name)>3]
dataset_list = []
save_path_list = []
# optimize latent code for each test subject
for id,file in enumerate(all_names):
    print(file)
    save_path = os.path.join(root_path,file)
    save_path_list.append(save_path)
    if opt.type == 'fit':
        dataset_list.append(dataset.PointCloud_wo_FreePoints(instance_idx=id,
                pointcloud_path=os.path.join(meta_params['point_cloud_path'],file+'.mat'),
                on_surface_points=meta_params['on_surface_points'],expand=meta_params['expand'],max_points=meta_params['max_points']))
        
if opt.type == 'fit':
    batchsize = 16
    total_shape = len(dataset_list)
    lr = meta_params['lr']
    from tqdm import tqdm
    fitted_embedding = np.zeros([total_shape,meta_params['latent_dim']])
    fitted_trans = np.zeros([total_shape,3])
    iter_idx = torch.arange(start=0,end=total_shape,step=batchsize).tolist()+[int(total_shape)]
    for start_id, end_id in zip(iter_idx[:-1],iter_idx[1:]):
        datasubset_list = dataset_list[start_id:end_id]
        cur_batchsize = len(datasubset_list)
        datasubset = dataset.PointCloudMulti(datasubset_list)
        embedding = model.template_code.clone().detach().unsqueeze(0).repeat(cur_batchsize,1) # initialization for evaluation stage
        embedding.requires_grad = True
        trans=torch.nn.Parameter(torch.zeros([cur_batchsize,3],dtype=embedding.dtype).cuda(),requires_grad=True)
        optim = torch.optim.Adam(lr=lr, params=[embedding,trans])
        with tqdm(total=len(datasubset)) as pbar:
            for i in range(len(datasubset)):
                print(i)
                model_input,gt = datasubset[i]
                model_input = {k:v.cuda() for k,v in model_input.items()}
                gt = {k:v.cuda() for k,v in gt.items()}
                model_input['coords']=model_input['coords']+trans.unsqueeze(1)
                losses = model.embedding(embedding, model_input,gt)
                train_loss = sum(list(losses.values()))
                optim.zero_grad()
                train_loss.backward()
                optim.step()
                pbar.update(1)
                if i%10==0:
                    tqdm.write("loss:%f"%train_loss)
        fitted_embedding[start_id:end_id] = embedding.detach().cpu().numpy()
        fitted_trans[start_id:end_id] = trans.detach().cpu().numpy()

    print('start saving parameters ......')
    for save_folder, embedding, trans in zip(save_path_list,fitted_embedding,fitted_trans):
        os.makedirs(save_folder,exist_ok=True)
        embedding_file_name = os.path.join(save_folder,"embedding.txt")
        trans_file_name = os.path.join(save_folder,"trans.txt")
        np.savetxt(embedding_file_name,embedding)
        np.savetxt(trans_file_name,trans)
    print('start saving mesh ......')
    with torch.no_grad():
        for save_folder, embedding, trans in zip(save_path_list,fitted_embedding,fitted_trans):
            sdf_meshing.create_mesh(model,os.path.join(save_folder,"mesh"),embedding=torch.from_numpy(embedding.astype(np.float32)).cuda(),N=256,get_color=True,offset=trans)

elif opt.type == 'generate':
    print('start saving parameters ......')
    for save_folder, embedding in zip(save_path_list,model.latent_codes.weight):
        os.makedirs(save_folder,exist_ok=True)
        embedding_file_name = os.path.join(save_folder,"embedding.txt")
        trans_file_name = os.path.join(save_folder,"trans.txt")
        # temp = []
        # for i in range(128):
        #     temp.append(embedding[i].detach().numpy())
        # np.savetxt(embedding_file_name, temp)
        np.savetxt(embedding_file_name,embedding.cpu().detach().numpy())
        np.savetxt(trans_file_name,np.zeros(3))
    print('start saving mesh ......')
    with torch.no_grad():
        for idx, save_folder in enumerate(save_path_list):
            os.makedirs(save_folder,exist_ok=True)
            sdf_meshing.create_mesh(model, os.path.join(save_folder,"mesh"),subject_idx=idx, N=256,get_color=False)

else:
    assert 0, "type must be 'fit' or 'generate'"
