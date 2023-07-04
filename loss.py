# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''Training losses for DIF-Net.
'''


from tkinter import W
import torch
import torch.nn.functional as F
import numpy as np
import utils
def smooth_loss(x,beta=0.5):
    xabs=torch.abs(x)
    loss=torch.where(xabs>beta,torch.square(xabs+beta)/(4*beta),xabs)
    return loss
def smooth_loss(x,y,beta=0.5):
    xabs=(x-y).norm(dim=-1)
    squarex=(torch.sum((x-y)**2,dim=-1)+2*beta*xabs)/(4*beta)+beta/4
    loss=torch.where(xabs>beta,squarex,xabs)
    return loss

def gaussian_KL(mean1,var1,mean2,var2,eps=1e-7):
    var1=var1.clamp(eps)
    var2=var2.clamp(eps)
    KL_div=0.5*(torch.log(var2/var1)-1+var1/var2+(mean1-mean2)**2/var2)
    return KL_div

def Welsh(x,v):
    return 1-torch.exp(-x**2/(2*v**2))

def compute_elastic_loss(jacobian=None, svals=None, eps=1e-6):
    if svals is None:
        svals = torch.linalg.svdvals(jacobian)
    log_svals = torch.log(svals.clamp(eps))
    sq_residual = torch.sum(log_svals**2, dim=-1)
    return sq_residual

def general_loss_with_squared_residual(squared_x, scale=0.03):
    squared_scaled_x = squared_x / (scale ** 2)
    loss = 2*squared_scaled_x/(squared_scaled_x + 4)
    return loss

def deform_implicit_loss(model_output, gt):

    gt_sdf = gt['sdf']
    gt_normals = gt['normals']
    net_input = model_output['model_in']


    embeddings = model_output['latent_vec']
    alignment_loss = model_output['alignment_loss']

    sdf_stage1=model_output['sdf_stage1']
    grad_sdf_stage1=model_output['grad_sdf_stage1']
    template_sdf_stage1=model_output['template_sdf_stage1']
    grad_sdf_template=model_output['grad_sdf_template']
    template_embedding=model_output['template_code']
    gradient_deform = model_output['grad_deform']

    sdf_constraint_stage1 = torch.where(gt_sdf != -1, torch.clamp(sdf_stage1,-0.5,0.5)-torch.clamp(gt_sdf,-0.5,0.5), torch.zeros_like(sdf_stage1))
    cos_sim, _,_, grad_norm,_ = utils.Safe_Cosine_Similarity.apply(grad_sdf_stage1, gt_normals,-1, True, 1e-8,1e-8)
    normal_constraint = torch.where(gt_sdf == 0, 1 - cos_sim, torch.zeros_like(grad_sdf_stage1[..., :1]))

    inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(sdf_stage1), torch.exp(-1e2 * torch.abs(sdf_stage1)))
    template_inter_constraint = torch.where(gt_sdf[:1] != -1, torch.zeros_like(template_sdf_stage1), 
                                            torch.exp(-1e2 * torch.abs(template_sdf_stage1)))
    total_inter_constraint = torch.cat([template_inter_constraint,inter_constraint])

    grad_constraint = torch.abs(grad_norm.squeeze(-1) - 1)
    template_grad_constraint = torch.abs(grad_sdf_template.norm(dim=-1) - 1)
    total_grad_constraint = torch.cat([grad_constraint,template_grad_constraint])

    det_grad_deform=model_output['differential_det']

    deform_flip_constraint = F.margin_ranking_loss(det_grad_deform,torch.zeros_like(det_grad_deform),
                                                        torch.ones_like(det_grad_deform,dtype=torch.int32),
                                                        margin = 1e-3,
                                                        reduction='none')

    rigid_measurement = model_output['local_rigid_measurement']
    grad_deform_constraint = torch.where((gt_sdf<=0)*(gt_sdf!=-1),
                                            rigid_measurement.unsqueeze(-1),
                                            torch.zeros_like(gt_sdf))

    reconstructed_coords=model_output['reconstructed_coords']
    reconstructed_loss = reconstructed_coords-net_input[...,:3]
    reconstructed_loss = reconstructed_loss ** 2

    target2template_loc_norm=model_output['target2template_loc_norm']
    norm_forward=(gradient_deform @ gt_normals.unsqueeze(-1)).squeeze(-1)
    grad_sdf_stage2=(target2template_loc_norm.unsqueeze(-2) @ gradient_deform).squeeze(-2)

    forward_normal_constraint = torch.where(gt_sdf == 0,
                                                1 - utils.safe_cosine_similarity(norm_forward, target2template_loc_norm, dim=-1,eps_backward=5e-1,keepdim=True),
                                                torch.zeros_like(target2template_loc_norm[...,:1]))

    normal_constraint_stage_pull_back = torch.where(gt_sdf == 0,
                                                1 - utils.safe_cosine_similarity(grad_sdf_stage2, gt_normals, dim=-1,eps_backward=5e-1,keepdim=True),
                                                torch.zeros_like(grad_sdf_stage2[..., :1]))

    
    # embeddings_constraint = torch.mean(torch.cat([template_embedding.unsqueeze(0),embeddings]) ** 2)
    embeddings_constraint = embeddings ** 2

    target_pb_template_sdf = model_output['target_pb_template_sdf']

    sdf_constraint_target_pb_template=torch.where(((gt_sdf*target_pb_template_sdf)<=0)*\
                                (gt_sdf!=-1),\
                                target_pb_template_sdf,torch.zeros_like(target_pb_template_sdf))

    discriptor_similarity = model_output['discriptor_similarity']**2
    similar_latent_embedding = model_output['similar_latent_code']
    template_reg = (similar_latent_embedding - template_embedding)**2

    return {
            'surface_target_pb_template':torch.abs(sdf_constraint_target_pb_template).mean() * 3e2,

            'reconstructed_loss':reconstructed_loss.mean() * 5e3,
            'alignment_loss':alignment_loss.mean() * 3e3, #important
            'sdf_constraint_stage1':torch.abs(sdf_constraint_stage1).mean() * 3e2,

            'embeddings_constraint': embeddings_constraint.mean() * 1e5,
            'inter': total_inter_constraint.mean() * 5e1,

            'normal_constraint': normal_constraint.mean() * 5e1,
            'normal_constraint_stage_pull_back': normal_constraint_stage_pull_back.mean() * 5e1,
            'forward_normal_constraint': forward_normal_constraint.mean() * 5e1,

            'grad_constraint': total_grad_constraint.mean() * 5,
            'template_reg':template_reg.mean() * 1e5,
            'deform_flip_constraint':deform_flip_constraint.mean() * 1e3,
            'grad_deform_constraint':grad_deform_constraint.mean() * 1e1, #important
            'discriptor_similarity':discriptor_similarity.mean() * 5e4, #important
            }

    # return {
    #         'surface_target_pb_template':torch.abs(sdf_constraint_target_pb_template).mean() * 3e3,

    #         'reconstructed_loss':reconstructed_loss.mean() * 5e4,
    #         'alignment_loss':alignment_loss.mean() * 3e4, #important
    #         'sdf_constraint_stage1':torch.abs(sdf_constraint_stage1).mean() * 3e3,

    #         'embeddings_constraint': embeddings_constraint.mean() * 1e6,
    #         'inter': total_inter_constraint.mean() * 5e2,

    #         'normal_constraint': normal_constraint.mean() * 5e2,
    #         'normal_constraint_stage_pull_back': normal_constraint_stage_pull_back.mean() * 5e2,
    #         'forward_normal_constraint': forward_normal_constraint.mean() * 5e2,

    #         'grad_constraint': total_grad_constraint.mean() * 5e1,
    #         'template_reg':template_reg.mean() * 1e6,
    #         'deform_flip_constraint':deform_flip_constraint.mean() * 1e4,
    #         'grad_deform_constraint':grad_deform_constraint.mean() * 1e2, #important
    #         'discriptor_similarity':discriptor_similarity.mean()*5e5, #important
    #         }

def implicit_loss(model_output, gt):

    gt_sdf = gt['sdf']
    gt_normals = gt['normals']

    embeddings = model_output['latent_vec']

    sdf_stage1=model_output['sdf_stage1']
    grad_sdf_stage1=model_output['grad_sdf_stage1']

    sdf_constraint_stage1 = torch.where(gt_sdf != -1, torch.clamp(sdf_stage1,-0.5,0.5)-torch.clamp(gt_sdf,-0.5,0.5), torch.zeros_like(sdf_stage1))
    inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(sdf_stage1), torch.exp(-1e2 * torch.abs(sdf_stage1)))
    normal_constraint = torch.where(gt_sdf == 0, 1 - F.cosine_similarity(grad_sdf_stage1, gt_normals, dim=-1)[..., None],
                                    torch.zeros_like(grad_sdf_stage1[..., :1]))
    embeddings_constraint=embeddings**2
    grad_constraint = torch.abs(grad_sdf_stage1.norm(dim=-1) - 1)

    model_in = model_output['model_in']
    model_out = model_output['model_out']
    reconstructed_loss = (model_out-model_in)**2

    return {
            'sdf_constraint_stage1':torch.abs(sdf_constraint_stage1).mean() * 3e3,
            'embeddings_constraint': embeddings_constraint.mean() * 1e6,
            'inter': inter_constraint.mean() * 5e2,
            'normal_constraint': normal_constraint.mean() * 5e2,
            'grad_constraint': grad_constraint.mean() * 5e1,
            'reconstructed_loss':reconstructed_loss.mean() * 5e4,
            }

def embedding_loss(model_output, gt):

    gt_sdf = gt['sdf']
    gt_normals = gt['normals']
    embeddings = model_output['latent_vec']
    sdf_stage1=model_output['sdf_stage1']
    grad_sdf_stage1=model_output['grad_sdf_stage1']
    gradient_deform = model_output['grad_deform']
    target_pb_template_sdf = model_output['target_pb_template_sdf']

    sdf_constraint_stage1 = torch.where(gt_sdf != -1, torch.clamp(sdf_stage1,-0.5,0.5)-torch.clamp(gt_sdf,-0.5,0.5), torch.zeros_like(sdf_stage1))
    normal_constraint = torch.where(gt_sdf == 0, 1 - F.cosine_similarity(grad_sdf_stage1, gt_normals, dim=-1)[..., None],
                                    torch.zeros_like(grad_sdf_stage1[..., :1]))
    grad_constraint = torch.abs(grad_sdf_stage1.norm(dim=-1) - 1)

    embeddings_constraint = torch.mean(embeddings ** 2)

    sdf_constraint_target_pb_template=torch.where(((gt_sdf*target_pb_template_sdf)<=0)*\
                                (gt_sdf!=-1),\
                                target_pb_template_sdf,torch.zeros_like(target_pb_template_sdf))
    target2template_loc_norm=model_output['target2template_loc_norm']
    norm_forward=(gradient_deform @ gt_normals.unsqueeze(-1)).squeeze(-1)
    grad_sdf_stage2=(target2template_loc_norm.unsqueeze(-2) @ gradient_deform).squeeze(-2)
    forward_normal_constraint = torch.where(gt_sdf == 0,
                                                1 - utils.safe_cosine_similarity(norm_forward, target2template_loc_norm, dim=-1,eps_backward=1e-3,keepdim=True),
                                                torch.zeros_like(target2template_loc_norm[...,:1]))

    normal_constraint_stage_pull_back = torch.where(gt_sdf == 0,
                                                1 - utils.safe_cosine_similarity(grad_sdf_stage2, gt_normals, dim=-1,eps_backward=1e-3,keepdim=True),
                                                torch.zeros_like(grad_sdf_stage2[..., :1]))


    return {
            'sdf_constraint_target_pb_template': torch.abs(sdf_constraint_target_pb_template).mean() * 3e3,
            'grad_constraint':grad_constraint.mean()*5e1,
            'sdf_constraint_stage1':torch.abs(sdf_constraint_stage1).mean() * 3e3,
            'embeddings_constraint': embeddings_constraint.mean() * 1e6,
            'normal_constraint': normal_constraint.mean() * 5e2,
            'normal_constraint_stage_pull_back': normal_constraint_stage_pull_back.mean() * 5e2,
            'forward_normal_constraint': forward_normal_constraint.mean() * 5e2,
            }

def editing_loss(model_output):
    template_source_points = model_output['template_source_points']
    template_target_points = model_output['template_target_points']
    target_sdf_stage1 = model_output['target_sdf_stage1']
    target_sdf_stage2 = model_output['target_sdf_stage2']
    target_embedding = model_output['target_embedding']
    source_embedding = model_output['source_embedding']

    corr_loss = (template_target_points - template_source_points)**2
    embedding_loss = (target_embedding - source_embedding)**2
    surface_loss1 = torch.abs(target_sdf_stage1)
    surface_loss2 = torch.abs(target_sdf_stage2)

    return {
        'sdf1':surface_loss1.mean() * 3e3,
        'sdf2':surface_loss2.mean() * 3e3,
        'corr_loss':corr_loss.mean() * 1e6,
        'embedding_loss':embedding_loss.mean() * 1e6,
    }
