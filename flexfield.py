# Adapted from DIF https://github.com/microsoft/DIF-Net.git

'''Define Flexfield-Net
'''

from re import template
from unicodedata import category
from numpy import mat
import torch
from torch import nn
from torch._C import dtype
import torch.nn.functional as F
import modules
from meta_modules import HyperNetwork
from loss import *
import utils
import numpy as np
from torchmcubes import  marching_cubes


class FlexField(nn.Module):
    def __init__(self, num_instances,part_num=20, latent_dim=128, hidden_num=128, hyper_hidden_layers=1,hyper_hidden_features=256, **kwargs):
        super().__init__()
        # latent code embedding for training subjects
        self.latent_dim = latent_dim
        self.immediate_dim=8
        self.num_instances = num_instances
        self.latent_codes = nn.Embedding(num_instances, self.latent_dim)
        self.template_code = nn.Parameter(torch.zeros(self.latent_dim),requires_grad=True)
        nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)

        self.template_part_field = modules.PartNet(hidden_features=hidden_num,out_features=part_num,in_features=3,num_hidden_layers=1,initial_first=True)
        self.subject_part_field = modules.PartNet(hidden_features=hidden_num,out_features=part_num,in_features=self.immediate_dim,num_hidden_layers=2)
        # Encoder
        self.deform_encoder=modules.SingleBVPNet(type='sine30',mode='mlp', hidden_features=hidden_num, num_hidden_layers=3, in_features=3+1,out_features=self.immediate_dim,
                                                     outermost_linear=False,last_initial=False)
        self.encoder_hyper_net = HyperNetwork(hyper_in_features=self.latent_dim, hyper_hidden_layers=hyper_hidden_layers, hyper_hidden_features=hyper_hidden_features,
                                      hypo_module=self.deform_encoder)
        # Decoder
        self.deform_decoder=modules.SingleBVPNet(type='sine30',mode='mlp', hidden_features=hidden_num, num_hidden_layers=2, in_features=self.immediate_dim,out_features=3,
                                                    first_initial=False)
        self.decoder_hyper_net = HyperNetwork(hyper_in_features=self.latent_dim, hyper_hidden_layers=hyper_hidden_layers, hyper_hidden_features=hyper_hidden_features,
                                      hypo_module=self.deform_decoder)
        
        # SDF module
        self.sdf_net=modules.SDFBVPNet(type='sine30',mode='mlp', hidden_features=hidden_num, num_hidden_layers=3, in_features=3,out_features=1)
        self.sdf_hyper_net = HyperNetwork(hyper_in_features=self.latent_dim, hyper_hidden_layers=3, hyper_hidden_features=hyper_hidden_features,
                                      hypo_module=self.sdf_net)

        self.template_points_num=50000
        self.register_buffer('template_surface_points',torch.randn([self.template_points_num,3],dtype=torch.float32)/3)
        self.nearest_embedding_id = None
        self.grid_res = 128
        self.voxel_size  = 2.0/float(self.grid_res)

        
    def detach_sdf(self):
        if not hasattr(self, 'detach_sdf_hyper_net'):
            self.detach_sdf_hyper_net=utils.detach_new_module(self.sdf_hyper_net)
        else:
            print('already')


    def get_template_coords(self,coords,embedding):
        with torch.no_grad():
            if self.template_code.allclose(embedding[0]):
                return coords
            sdf_hypo_params=self.sdf_hyper_net(embedding)
            sdf_stage1=self.sdf_net({'coords':coords},params=sdf_hypo_params)['model_out']

            additional_embedding=self.template_code[None,:]

            encoder_hypo_params=self.encoder_hyper_net(embedding)
            model_in=torch.cat([coords,sdf_stage1],dim=-1)
            latent_mean=self.deform_encoder({'coords':model_in},params=encoder_hypo_params)['model_out']#B,N,D
            latent_feature=latent_mean
            decoder_hypo_params=self.decoder_hyper_net(additional_embedding)
            
            output=self.deform_decoder({'coords':latent_feature},params=decoder_hypo_params)['model_out']#B,N,4
            new_coords=output[...,:3]
        
        return new_coords



    def get_part_id(self,coords,embedding):
        with torch.no_grad():
            # if self.template_code.allclose(embedding[0]):
            #     part_prob=self.part_field(coords)
            #     part_prob=torch.softmax(part_prob,dim=-1)
            #     val,part_id=part_prob.max(dim=-1)
            #     print(val.min(),val.max())
            #     return part_id
            # embedding=self.template_code.unsqueeze(0)
            if self.template_code.allclose(embedding[0]):
                part_prob=self.template_part_field(coords)
                part_prob=torch.softmax(part_prob,dim=-1)
                val,part_id=part_prob.max(dim=-1)
                return part_id
            sdf_hypo_params=self.sdf_hyper_net(embedding)
            sdf_stage1=self.sdf_net({'coords':coords},params=sdf_hypo_params)['model_out']
            # sample_instance_idx=torch.tensor([135],dtype=int,device=coords.device)

            # additional_embedding=self.latent_codes(sample_instance_idx)

            encoder_hypo_params=self.encoder_hyper_net(embedding)
            model_in=torch.cat([coords,sdf_stage1],dim=-1)
            latent_feature=self.deform_encoder({'coords':model_in},params=encoder_hypo_params)['model_out']#B,N,D
            # part_prob=self.subject_part_field(latent_feature)
            decoder_hypo_params=self.decoder_hyper_net(self.template_code.unsqueeze(0))
            output=self.deform_decoder({'coords':latent_feature},params=decoder_hypo_params)['model_out']#B,N,4
            new_coords=output[...,:3]
            # new_coords=output
            part_prob=self.template_part_field(new_coords)
            part_prob=torch.softmax(part_prob,dim=-1)
            val,part_id=part_prob.max(dim=-1)
            print(val.min(),val.max())
            
            return part_id



    def get_latent_code(self,instance_idx):

        embedding = self.latent_codes(instance_idx)

        return embedding


    def warp_shape(self,coords,instance_idx,instance_target):
        with torch.no_grad():
            batchsize=coords.shape[0]
            encoder_embedding=self.latent_codes(instance_idx)
            decoder_embedding=self.latent_codes(instance_target)
            encoder_hypo_params=self.encoder_hyper_net(encoder_embedding)
            model_in=torch.cat([coords,torch.zeros_like(coords[...,:1])],dim=-1)
            latent_mean,latent_var=self.deform_encoder({'coords':model_in},params=encoder_hypo_params)['model_out'].split([self.immediate_dim,self.immediate_dim],dim=-1)#B,N,D
            latent_feature=latent_mean+torch.randn_like(latent_var)*latent_var
            decoder_hypo_params=self.decoder_hyper_net(decoder_embedding)
            output=self.deform_decoder({'coords':latent_feature},params=decoder_hypo_params)['model_out']
            new_coords=output[...,:-1]
            return new_coords


    def inference(self,coords,embedding):
        with torch.no_grad():
            additional_embedding=self.template_code[None,:]
            # embedding, additional_embedding = additional_embedding, embedding
            sdf_hypo_params=self.sdf_hyper_net(embedding)
            sdf_stage1=self.sdf_net({'coords':coords},params=sdf_hypo_params)['model_out']
            # return sdf_stage1
            if self.template_code.allclose(embedding[0]):
                return sdf_stage1
            encoder_hypo_params=self.encoder_hyper_net(embedding)
            model_in=torch.cat([coords,sdf_stage1],dim=-1)
            latent_feature=self.deform_encoder({'coords':model_in},params=encoder_hypo_params)['model_out']#B,N,D
            decoder_hypo_params=self.decoder_hyper_net(additional_embedding)
            
            output=self.deform_decoder({'coords':latent_feature},params=decoder_hypo_params)['model_out']#B,N,3
            new_coords=output[...,:3]
            sdf_hypo_params=self.sdf_hyper_net(additional_embedding)
            sdf_final=self.sdf_net({'coords':new_coords},
                                        params=sdf_hypo_params)['model_out']
        return sdf_final
    

    def warp_to_multi_shapes(self,coords,start_embedding,end_embeddings):
        with torch.no_grad():
            sdf_hypo_params=self.sdf_hyper_net(start_embedding.unsqueeze(0))
            sdf_stage1=self.sdf_net({'coords':coords.unsqueeze(0)},params=sdf_hypo_params)['model_out']

            encoder_hypo_params=self.encoder_hyper_net(start_embedding.unsqueeze(0))
            model_in=torch.cat([coords.unsqueeze(0),sdf_stage1],dim=-1)
            latent_mean=self.deform_encoder({'coords':model_in},params=encoder_hypo_params)['model_out']#B,N,D
            latent_feature=latent_mean
            decoder_hypo_params=self.decoder_hyper_net(end_embeddings)
            
            output=self.deform_decoder({'coords':latent_feature.repeat(end_embeddings.shape[0],1,1)},params=decoder_hypo_params)['model_out']#B,N,4
            new_coords=output[...,:-1]
        return new_coords

    def warp_to_multi_shapes_w_template(self,start_coords,end_coords,start_embedding,end_embeddings):
        import pytorch3d.ops
        with torch.no_grad():
            total_embedding=torch.cat([start_embedding.unsqueeze(0),end_embeddings],dim=0)
            total_coords=torch.cat([start_coords.unsqueeze(0),end_coords],dim=0)
            sdf_hypo_params=self.sdf_hyper_net(total_embedding)
            sdf_stage1=self.sdf_net({'coords':total_coords},params=sdf_hypo_params)['model_out']

            encoder_hypo_params=self.encoder_hyper_net(total_embedding)
            model_in=torch.cat([total_coords,sdf_stage1],dim=-1)
            latent_mean=self.deform_encoder({'coords':model_in},params=encoder_hypo_params)['model_out']#B,N,D
            latent_feature=latent_mean
            decoder_hypo_params=self.decoder_hyper_net(self.template_code.unsqueeze(0))
            
            output=self.deform_decoder({'coords':latent_feature.reshape(1,-1,latent_feature.shape[-1])},params=decoder_hypo_params)['model_out']#B,N,4
            output=output.reshape(total_coords.shape[0],total_coords.shape[1],output.shape[-1])
            new_coords=output[...,:-1]
            _,idx,_=pytorch3d.ops.knn_points(new_coords[:1].repeat(new_coords.shape[0]-1,1,1),new_coords[1:],K=1)
            nearest_points=torch.gather(end_coords,dim=1,index=idx.repeat(1,1,3))
        return nearest_points

    def forward(self, model_input,gt,**kwargs):

        # sdf_hypo_params=self.sdf_hyper_net(self.template_code.unsqueeze(0))
        # sdf_hypo_params_detach=self.detach_sdf_hyper_net(self.template_code.unsqueeze(0))
        # for k in sdf_hypo_params:
        #     print(torch.all(sdf_hypo_params[k]==sdf_hypo_params_detach[k]))
        ## totally same!
        
        instance_idx = model_input['instance_idx']
        embedding=self.latent_codes(instance_idx)
        coords = model_input['coords'] # 3 dimensional input coordinates
        coords.requires_grad_()

        batchsize=coords.shape[0]
        points_num=coords.shape[1]

        template_surface_points_num=points_num//2
        surface_rand_idcs = np.random.choice(self.template_points_num,points_num//2)
        free_rand_idcs=np.random.choice(self.template_points_num,points_num//4)
        selected_surface_points=self.template_surface_points[surface_rand_idcs]
        selected_surface_points.requires_grad_()
        with torch.no_grad():            
            off_surface_points = torch.rand([points_num//4,3],device=coords.device) * 2 - 1
            template_coords = torch.cat([selected_surface_points,
                                        off_surface_points,
                                        self.template_surface_points[free_rand_idcs]+torch.randn([points_num//4,3],device=coords.device)*0.0025,
                                        ],dim=0).unsqueeze(0)
        template_coords.requires_grad_()

        total_embedding=torch.cat([self.template_code.unsqueeze(0),embedding],dim=0)

        sdf_hypo_params=self.sdf_hyper_net(embedding)
        template_sdf_hypo_params=self.detach_sdf_hyper_net(self.template_code.unsqueeze(0))

        target_sdf_stage1 = self.sdf_net({'coords':coords},params=sdf_hypo_params)['model_out']
        template_sdf_stage1 = self.sdf_net({'coords':template_coords},params=template_sdf_hypo_params)['model_out']
        target_grad_sdf_stage1,template_grad_sdf_stage1 = \
                                torch.autograd.grad([target_sdf_stage1,template_sdf_stage1], [coords,template_coords], 
                                                    grad_outputs=[torch.ones_like(target_sdf_stage1),torch.ones_like(template_sdf_stage1)], create_graph=True)

        target_model_in = torch.cat([coords,target_sdf_stage1.detach()],dim=-1)
        template_model_in = torch.cat([template_coords,template_sdf_stage1.detach()],dim=-1)
        

        ################## first forward ##################
        encoder_hypo_params=self.encoder_hyper_net(total_embedding)
        target_encoder_hypo_params={}
        template_encoder_hypo_params={}
        for k in encoder_hypo_params:
            template_encoder_hypo_params[k] = encoder_hypo_params[k][:1]
            target_encoder_hypo_params[k] = encoder_hypo_params[k][1:]

        template_latent=self.deform_encoder({'coords':template_model_in},params=template_encoder_hypo_params)['model_out']#1,N,D
        target_latent=self.deform_encoder({'coords':target_model_in},params=target_encoder_hypo_params)['model_out']#B,N,D
        

        template_batch_decoder_hypo_params=self.decoder_hyper_net(total_embedding)
        template_decoder_hypo_params:dict={}
        decoder_hypo_params:dict={}
        for k,v in template_batch_decoder_hypo_params.items():
            template_decoder_hypo_params[k]=v[:1]
            decoder_hypo_params[k]=v[1:batchsize+1]

        template_output=self.deform_decoder({'coords':target_latent.view(1,-1,self.immediate_dim)},
                                                        params=template_decoder_hypo_params)['model_out']
        target2template_loc=template_output.reshape(batchsize,points_num,3)#B,N,3
        template_recon = self.deform_decoder({'coords':template_latent},
                                                        params=template_decoder_hypo_params)['model_out']#1,N,3
        
        target2target_output = self.deform_decoder({'coords':target_latent},
                                                    params=decoder_hypo_params)['model_out']#B,N,3

        u = target2template_loc[:,:,0]
        v = target2template_loc[:,:,1]
        w = target2template_loc[:,:,2]
        grad_outputs = torch.ones_like(u)
        grad_u = torch.autograd.grad(u, [coords], grad_outputs=grad_outputs, create_graph=True)[0]
        grad_v = torch.autograd.grad(v, [coords], grad_outputs=grad_outputs, create_graph=True)[0]
        grad_w = torch.autograd.grad(w, [coords], grad_outputs=grad_outputs, create_graph=True)[0]
        grad_deform = torch.stack([grad_u,grad_v,grad_w],dim=2)

######################## get sdf ########################
        target_pb_template_sdf=self.sdf_net({'coords':template_output},
                                    params=template_sdf_hypo_params)['model_out'].reshape(batchsize,points_num,1)

        sdf_hypo_params = {k:v.detach() for k,v in sdf_hypo_params.items()}

        target2template_loc_norm = torch.autograd.grad([target_pb_template_sdf],
                                                    [template_output],
                                                    grad_outputs=torch.ones_like(target_pb_template_sdf), 
                                                    create_graph=True)[0].reshape(batchsize,points_num,3)

############################################################
        G=8
        sigma=5e-2
        with torch.no_grad():
            batch_sample_noise = torch.randn([batchsize,points_num,G,3],device=coords.device)*sigma

        local_rigid_measurement, direction_transformer, differential_det, differential_singular = utils.Differential_Relevant_Terms.apply(grad_deform,1,1e-1)
        template_sample_noise = batch_sample_noise @ direction_transformer.transpose(-1,-2)
        batch_sample_sdf=self.sdf_net({'coords':(coords.unsqueeze(-2)+batch_sample_noise).view(batchsize,points_num*G,3)},
                                            params=sdf_hypo_params)['model_out'].reshape(batchsize,points_num,G,1)
        template_sample_sdf=self.sdf_net({'coords':(target2template_loc[:,:,None,:3]+template_sample_noise).view(1,batchsize*points_num*G,3)},
                                            params=template_sdf_hypo_params)['model_out'].reshape(batchsize,points_num,G,1)
        
        discriptor_similarity=torch.where(gt['sdf'].unsqueeze(-2)==0,batch_sample_sdf-template_sample_sdf,torch.zeros_like(batch_sample_sdf))


        ####################################

        target_subject_part_prob=torch.softmax(self.subject_part_field(target_latent),dim=-1)\
                                *((gt['sdf']<sigma)*(gt['sdf']!=-1)).type(target_latent.dtype)

        target_template_part_prob=torch.softmax(self.template_part_field(target2template_loc),dim=-1)\
                                *((gt['sdf']<sigma)*(gt['sdf']!=-1)).type(target2template_loc.dtype)
        
        target_foreground_prob = torch.cat([target_subject_part_prob, target_template_part_prob],dim = -1)

        alignment_loss = utils.corresponding_points_alignment_loss(target2template_loc,
                                                                    coords,
                                                                    target_foreground_prob.transpose(-1,-2),
                                                                    eps_backward = 1e-7).sum(dim=-1)/points_num

        ####################################
        similar_latent_code = torch.zeros_like(self.template_code)
        # if self.nearest_embedding_id is None:
        #     similar_latent_code = torch.zeros_like(self.template_code)
        # else:
        #     similar_latent_code = self.latent_codes(torch.tensor(self.nearest_embedding_id,dtype=torch.int32,device=self.template_code.device)).detach()

        model_out = {   
                        'model_in':torch.cat([template_coords,coords],dim=0),
                        'reconstructed_coords':torch.cat([template_recon,target2target_output],dim=0),
                        
                        'sdf_stage1':target_sdf_stage1,
                        'grad_sdf_stage1':target_grad_sdf_stage1,
                        'latent_vec':embedding,

                        'template_sdf_stage1':template_sdf_stage1,
                        'grad_sdf_template':template_grad_sdf_stage1,
                        'template_code':self.template_code,
                        'similar_latent_code':similar_latent_code,

                        'target2template_loc_norm':target2template_loc_norm,

                        'target_pb_template_sdf':target_pb_template_sdf,
                        'template_surface_points_num':template_surface_points_num,
                        
                        'local_rigid_measurement':local_rigid_measurement,
                        'differential_det':differential_det,
                        'alignment_loss':alignment_loss,
                        'grad_deform':grad_deform,
                        'discriptor_similarity':discriptor_similarity,
                        }
        losses = deform_implicit_loss(model_out, gt)

        return losses

    # for evaluation
    def embedding(self, embed, model_input,gt):
        coords = model_input['coords'] # 3 dimensional input coordinates
        coords.requires_grad_()

        batchsize=coords.shape[0]
        points_num=coords.shape[1]

        sdf_hypo_params=self.sdf_hyper_net(embed)
        template_sdf_hypo_params=self.sdf_hyper_net(self.template_code.unsqueeze(0))

        target_sdf_stage1 = self.sdf_net({'coords':coords},params=sdf_hypo_params)['model_out']
        target_grad_sdf_stage1 = torch.autograd.grad([target_sdf_stage1], [coords], 
                                                    grad_outputs=torch.ones_like(target_sdf_stage1), create_graph=True)[0]

        target_model_in = torch.cat([coords,torch.where(gt['sdf']!=-1,gt['sdf'],target_sdf_stage1).detach()],dim=-1)
        

        ################## first forward ##################
        encoder_hypo_params=self.encoder_hyper_net(embed)

        target_latent=self.deform_encoder({'coords':target_model_in},params=encoder_hypo_params)['model_out']#B,N,D
        

        template_decoder_hypo_params=self.decoder_hyper_net(self.template_code.unsqueeze(0))

        template_output=self.deform_decoder({'coords':target_latent.view(1,-1,self.immediate_dim)},
                                                        params=template_decoder_hypo_params)['model_out']

        target2template_loc=template_output.reshape(batchsize,points_num,3)#B,N,3
        u = target2template_loc[:,:,0]
        v = target2template_loc[:,:,1]
        w = target2template_loc[:,:,2]
        grad_outputs = torch.ones_like(u)
        grad_u = torch.autograd.grad(u, [coords], grad_outputs=grad_outputs, create_graph=True)[0]
        grad_v = torch.autograd.grad(v, [coords], grad_outputs=grad_outputs, create_graph=True)[0]
        grad_w = torch.autograd.grad(w, [coords], grad_outputs=grad_outputs, create_graph=True)[0]
        grad_deform = torch.stack([grad_u,grad_v,grad_w],dim=2)

        target_pb_template_sdf=self.sdf_net({'coords':template_output},
                                    params=template_sdf_hypo_params)['model_out'].reshape(batchsize,points_num,1)

        sdf_hypo_params = {k:v.detach() for k,v in sdf_hypo_params.items()}

        target2template_loc_norm = torch.autograd.grad([target_pb_template_sdf],
                                                    [template_output],
                                                    grad_outputs=torch.ones_like(target_pb_template_sdf), 
                                                    create_graph=True)[0].reshape(batchsize,points_num,3)

############################################################

        model_out = {   
                        'model_in':coords,
                        'sdf_stage1':target_sdf_stage1,
                        'grad_sdf_stage1':target_grad_sdf_stage1,
                        'latent_vec':embed,
                        'grad_deform':grad_deform,
                        'target_pb_template_sdf':target_pb_template_sdf,
                        'target2template_loc_norm':target2template_loc_norm,
                        }
        losses = embedding_loss(model_out, gt)

        return losses

    def editing(self, source_points:torch.Tensor, target_points:torch.Tensor,source_embedding:torch.Tensor,target_embedding:torch.Tensor):
        source_points=source_points.unsqueeze(0)
        target_points=target_points.unsqueeze(0)
        target_embedding=target_embedding.unsqueeze(0)
        source_embedding=source_embedding.unsqueeze(0)
        with torch.no_grad():
            if self.template_code.allclose(source_embedding[0]):
                template_source_points = source_points
            else:
                sdf_hypo_params=self.sdf_hyper_net(source_embedding)
                source_sdf_stage1 = self.sdf_net({'coords':source_points},params=sdf_hypo_params)['model_out']
                source_model_in = torch.cat([source_points,source_sdf_stage1],dim=-1)
                encoder_hypo_params=self.encoder_hyper_net(source_embedding)
                source_latent=self.deform_encoder({'coords':source_model_in},params=encoder_hypo_params)['model_out']#B,N,D
                decoder_hypo_params=self.decoder_hyper_net(self.template_code.squeeze(0))
                template_source_points = self.deform_decoder({'coords':source_latent},
                                                        params=decoder_hypo_params)['model_out']#B,N,3
        sdf_hypo_params=self.sdf_hyper_net(target_embedding)
        target_sdf_stage1 = self.sdf_net({'coords':target_points},params=sdf_hypo_params)['model_out']
        target_model_in = torch.cat([target_points,target_sdf_stage1],dim=-1)
        encoder_hypo_params=self.encoder_hyper_net(target_embedding)
        target_latent=self.deform_encoder({'coords':target_model_in},params=encoder_hypo_params)['model_out']#B,N,D
        decoder_hypo_params=self.decoder_hyper_net(self.template_code.squeeze(0))
        template_target_points = self.deform_decoder({'coords':target_latent},
                                                params=decoder_hypo_params)['model_out']#B,N,3
        sdf_hypo_params=self.sdf_hyper_net(self.template_code.squeeze(0))
        target_sdf_stage2 = self.sdf_net({'coords':template_target_points},params=sdf_hypo_params)['model_out']
        model_out = {
            'template_source_points':template_source_points,
            'template_target_points':template_target_points,
            'target_sdf_stage1':target_sdf_stage1,
            'target_sdf_stage2':target_sdf_stage2,
            'target_embedding':target_embedding,
            'source_embedding':source_embedding,
        }
        losses = editing_loss(model_out)
        return losses

    def sample_template_points(self):
        with torch.no_grad():
            N=self.grid_res
            voxel_origin = [-1, -1, -1]
            voxel_size = 2.0 / (N - 1)

            overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor()).cuda()
            samples = torch.zeros(N ** 3, 3).cuda()

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

            template_sdf_hypo_params=self.sdf_hyper_net(self.template_code.unsqueeze(0))
            template_sdf=self.sdf_net({'coords':samples.unsqueeze(0)},
                                    params=template_sdf_hypo_params)['model_out'][0]
            sdf_values = template_sdf.reshape(N, N, N)
            
            verts, faces = marching_cubes(sdf_values.transpose(0,2).contiguous(), 0)
            verts*=voxel_size
            verts+=torch.tensor(voxel_origin,dtype=verts.dtype,device=verts.device)

            alpha_gen = torch.tensor(np.random.dirichlet((1,)*3, self.template_points_num)).cuda()
            face_idx = torch.randperm(self.template_points_num).cuda()%faces.shape[0]
            selected_verts=(alpha_gen[:, :, None] * verts[faces.long()[face_idx]]).sum(dim=1)
            self.template_surface_points[:]=selected_verts

            # dist=torch.sum((self.latent_codes.weight-self.template_code.unsqueeze(0))**2,dim=-1)
            # self.nearest_embedding_id = int(dist.argmin())
            
    def eval_correspondence(self,embeddings:torch.Tensor,coords:torch.Tensor,batchsize:int=100,pair_num_max:int=500,id_list = None):
            if id_list is not None:
                embeddings = self.latent_codes(id_list)
            import pytorch3d.ops
            from tqdm import tqdm
            with torch.no_grad():
                iter_idx = torch.arange(start=0,end=coords.shape[0],step=batchsize).tolist()+[int(coords.shape[0])]
                template_coords = torch.zeros_like(coords)
                print("Mapping points to template space")
                for start_id, end_id in zip(tqdm(iter_idx[:-1]),iter_idx[1:]):
                    cur_embedding = embeddings[start_id:end_id]
                    cur_coords = coords[start_id:end_id]
                    sdf_hypo_params=self.sdf_hyper_net(cur_embedding)
                    target_sdf_stage1 = self.sdf_net({'coords':cur_coords},params=sdf_hypo_params)['model_out']
                    target_model_in = torch.cat([cur_coords,target_sdf_stage1],dim=-1)
                    target_encoder_hypo_params=self.encoder_hyper_net(cur_embedding)
                    target_latent=self.deform_encoder({'coords':target_model_in},params=target_encoder_hypo_params)['model_out']#B,N,D
                    target_decoder_hypo_params=self.decoder_hyper_net(self.template_code.unsqueeze(0))
                    template_coords[start_id:end_id] = self.deform_decoder({'coords':target_latent.reshape(1,-1,self.immediate_dim)},
                                                            params=target_decoder_hypo_params)['model_out'].reshape(cur_coords.shape)
                comb = torch.combinations(torch.arange(coords.shape[0],device=coords.device))
                pair_idx = torch.cat([comb,torch.stack([comb[:,1],comb[:,0]],dim=1)],dim=0)
                iter_idx = torch.arange(start=0,end=pair_idx.shape[0],step=pair_num_max).tolist()+[int(pair_idx.shape[0])]
                total_coord_idx = torch.zeros([pair_idx.shape[0],coords.shape[1]],dtype=pair_idx.dtype,device=pair_idx.device)
                print("Searching correspondence")
                for start_id, end_id in zip(tqdm(iter_idx[:-1]),iter_idx[1:]):
                    cur_pair_idx = pair_idx[start_id:end_id]
                    coords_from = template_coords[cur_pair_idx[:,0]]
                    coords_to = template_coords[cur_pair_idx[:,1]]
                    _,corr_idx,_=pytorch3d.ops.knn_points(coords_from,coords_to,K=1)
                    total_coord_idx[start_id:end_id]=corr_idx.squeeze(-1)
                return pair_idx, total_coord_idx