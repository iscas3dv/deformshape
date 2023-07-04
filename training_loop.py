# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''Main training loop for DIF-Net.
'''
import torch
from torch.nn import parameter
import utils
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import sdf_meshing

def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_schedules=None, is_train=True, optim='Adam', **kwargs):

    print('Training Info:')
    print('num_instances:\t\t',kwargs['num_instances'])
    print('batch_size:\t\t',kwargs['batch_size'])
    print('epochs:\t\t\t',epochs)
    print('learning rate:\t\t',lr)
    for key in kwargs:
        if 'loss' in key:
            print(key+':\t',kwargs[key])
    if is_train:
        if optim=='Adam':
            for name, param in model.module.named_parameters():
                if not name.find('detach')==-1:
                    print(name)
            optim = torch.optim.Adam(lr=lr,
                params=[param for name, param in model.module.named_parameters() if name.find('detach')==-1])
            if 'checkpoint_path' in kwargs and len(kwargs['checkpoint_path'])>0:
                state_dict=torch.load(kwargs['checkpoint_path'].replace('model','optim'))
                optim.load_state_dict(state_dict)
        else:
            optim = torch.optim.SGD(lr=lr,momentum=0.9,weight_decay=0.0001,#weight_decay=0.0005,
                params=[param for name, param in model.module.named_parameters() if name.find('detach')==-1])
        model.module.detach_sdf()
    else:
        embedding = model.latent_codes(torch.zeros(1).long().cuda()).clone().detach() # initialization for evaluation stage
        # embedding = model.template_code.unsqueeze(0).clone().detach() # initialization for evaluation stage
        embedding.requires_grad = True
        trans=torch.nn.Parameter(torch.zeros([1,3],device=embedding.device,dtype=embedding.dtype),requires_grad=True)
        optim = torch.optim.Adam(lr=lr, params=[embedding,trans])

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            if not epoch % epochs_til_checkpoint:
                if is_train:
                    torch.save(model.module.state_dict(),
                               os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                    torch.save(optim.state_dict(),
                                   os.path.join(checkpoints_dir, 'optim_epoch_%04d.pth' % epoch))
                else:
                    embed_save = embedding.detach().squeeze().cpu().numpy()
                    np.savetxt(os.path.join(checkpoints_dir, 'embedding_epoch_%04d.txt' % epoch),
                               embed_save)
                    trans_save = trans[0].detach().squeeze().cpu().numpy()
                    np.savetxt(os.path.join(checkpoints_dir, 'trans_epoch_%04d.txt' % epoch),
                            trans_save)

                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                           np.array(train_losses))
            # train_dataloader.dataset.shuffle_instance_points()
            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()
            
                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}

                if is_train:
                    losses = model(model_input,gt)
                else:
                    model_input['coords']=model_input['coords']+trans.unsqueeze(1)
                    losses = model.embedding(embedding, model_input,gt)

                train_loss = 0.
                has_nan=False
                output_string=""
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()
                    if torch.any(torch.isnan(single_loss)):
                        print(loss_name,'nan!')
                        has_nan=True
                    if loss_schedules is not None and loss_name in loss_schedules:
                        writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                        single_loss *= loss_schedules[loss_name](total_steps)

                    writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss
                    output_string+="%s %.3f "%(loss_name,float(single_loss))
                if not total_steps%1:
                    tqdm.write(output_string)
                assert(not(has_nan))

                train_losses.append(train_loss.item())
                writer.add_scalar("total_train_loss", train_loss, total_steps)

                if not total_steps % steps_til_summary:
                    if is_train:
                        torch.save(model.module.state_dict(),
                                   os.path.join(checkpoints_dir, 'model_current.pth'))
                        

                optim.zero_grad()
                train_loss.backward()
                # torch.nn.utils.clip_grad_value_(model.module.parameters(), clip_value = 1)
                optim.step()

                pbar.update(1)

                if not total_steps % steps_til_summary:
                    tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

                total_steps += 1
                if is_train and total_steps%5==0:
                    model.module.sample_template_points()


        if is_train:
            torch.save(model.module.cpu().state_dict(),
                       os.path.join(checkpoints_dir, 'model_final.pth'))
            torch.save(optim.state_dict(),
                        os.path.join(checkpoints_dir, 'optim_final.pth'))
        else:
            embed_save = embedding.detach().squeeze().cpu().numpy()
            np.savetxt(os.path.join(checkpoints_dir, 'embedding_epoch_%04d.txt' % epoch),
                       embed_save)
            trans_save = trans[0].detach().squeeze().cpu().numpy()
            np.savetxt(os.path.join(checkpoints_dir, 'trans_epoch_%04d.txt' % epoch),
                       trans_save)
            
            sdf_meshing.create_mesh(model, os.path.join(checkpoints_dir,'test'), embedding=embedding, N=256, level=0, get_color=False,offset=trans[0].detach().cpu().numpy())

        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))
