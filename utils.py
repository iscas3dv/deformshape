# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from tkinter import W
from turtle import forward
from numpy import float64
import torch
import torch.nn.functional as F

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

#fork https://github.com/learnables/learn2learn/blob/752200384c3ca8caeb8487b5dd1afd6568e8ec01/learn2learn/utils/__init__.py

def clone_module(module, memo=None):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)
    **Description**
    Creates a copy of a module, whose parameters/buffers/submodules
    are created using PyTorch's torch.clone().
    This implies that the computational graph is kept, and you can compute
    the derivatives of the new modules' parameters w.r.t the original
    parameters.
    **Arguments**
    * **module** (Module) - Module to be cloned.
    **Return**
    * (Module) - The cloned module.
    **Example**
    ~~~python
    net = nn.Sequential(Linear(20, 10), nn.ReLU(), nn.Linear(10, 2))
    clone = clone_module(net)
    error = loss(clone(X), y)
    error.backward()  # Gradients are back-propagate all the way to net.
    ~~~
    """
    # NOTE: This function might break in future versions of PyTorch.

    # TODO: This function might require that module.forward()
    #       was called in order to work properly, if forward() instanciates
    #       new variables.
    # TODO: We can probably get away with a shallowcopy.
    #       However, since shallow copy does not recurse, we need to write a
    #       recursive version of shallow copy.
    # NOTE: This can probably be implemented more cleanly with
    #       clone = recursive_shallow_copy(model)
    #       clone._apply(lambda t: t.clone())

    if memo is None:
        # Maps original data_ptr to the cloned tensor.
        # Useful when a Module uses parameters from another Module; see:
        # https://github.com/learnables/learn2learn/issues/174
        memo = {}

    # First, create a copy of the module.
    # Adapted from:
    # https://github.com/pytorch/pytorch/blob/65bad41cbec096aa767b3752843eddebf845726f/torch/nn/modules/module.py#L1171
    if not isinstance(module, torch.nn.Module):
        return module
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()

    # Second, re-write all parameters
    if hasattr(clone, '_parameters'):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                param = module._parameters[param_key]
                param_ptr = param.data_ptr
                if param_ptr in memo:
                    clone._parameters[param_key] = memo[param_ptr]
                else:
                    cloned = param.clone()
                    clone._parameters[param_key] = cloned
                    memo[param_ptr] = cloned

    # Third, handle the buffers if necessary
    if hasattr(clone, '_buffers'):
        for buffer_key in module._buffers:
            if clone._buffers[buffer_key] is not None and \
                    clone._buffers[buffer_key].requires_grad:
                buff = module._buffers[buffer_key]
                buff_ptr = buff.data_ptr
                if buff_ptr in memo:
                    clone._buffers[buffer_key] = memo[buff_ptr]
                else:
                    cloned = buff.clone()
                    clone._buffers[buffer_key] = cloned
                    memo[param_ptr] = cloned

    # Then, recurse for each submodule
    if hasattr(clone, '_modules'):
        for module_key in clone._modules:
            clone._modules[module_key] = clone_module(
                module._modules[module_key],
                memo=memo,
            )

    # Finally, rebuild the flattened parameters for RNNs
    # See this issue for more details:
    # https://github.com/learnables/learn2learn/issues/139
    if hasattr(clone, 'flatten_parameters'):
        clone = clone._apply(lambda x: x)
    return clone


def detach_module(module):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)
    **Description**
    Detaches all parameters/buffers of a previously cloned module from its computational graph.
    Note: detach works in-place, so it does not return a copy.
    **Arguments**
    * **module** (Module) - Module to be detached.
    **Example**
    ~~~python
    net = nn.Sequential(Linear(20, 10), nn.ReLU(), nn.Linear(10, 2))
    clone = clone_module(net)
    detach_module(clone)
    error = loss(clone(X), y)
    error.backward()  # Gradients are back-propagate on clone, not net.
    ~~~
    """
    if not isinstance(module, torch.nn.Module):
        return
    # First, re-write all parameters
    for param_key in module._parameters:
        if module._parameters[param_key] is not None:
            detached = module._parameters[param_key].detach_()

    # Second, handle the buffers if necessary
    for buffer_key in module._buffers:
        if module._buffers[buffer_key] is not None and \
                module._buffers[buffer_key].requires_grad:
            module._buffers[buffer_key] = module._buffers[buffer_key].detach_()

    # Then, recurse for each submodule
    for module_key in module._modules:
        detach_module(module._modules[module_key])

def detach_new_module(module):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)
    **Description**
    Creates a copy of a module, whose parameters/buffers/submodules
    are created using PyTorch's torch.clone().
    This implies that the computational graph is kept, and you can compute
    the derivatives of the new modules' parameters w.r.t the original
    parameters.
    **Arguments**
    * **module** (Module) - Module to be cloned.
    **Return**
    * (Module) - The cloned module.
    **Example**
    ~~~python
    net = nn.Sequential(Linear(20, 10), nn.ReLU(), nn.Linear(10, 2))
    clone = clone_module(net)
    error = loss(clone(X), y)
    error.backward()  # Gradients are back-propagate all the way to net.
    ~~~
    """
    # NOTE: This function might break in future versions of PyTorch.

    # TODO: This function might require that module.forward()
    #       was called in order to work properly, if forward() instanciates
    #       new variables.
    # TODO: We can probably get away with a shallowcopy.
    #       However, since shallow copy does not recurse, we need to write a
    #       recursive version of shallow copy.
    # NOTE: This can probably be implemented more cleanly with
    #       clone = recursive_shallow_copy(model)
    #       clone._apply(lambda t: t.clone())

    
    # First, create a copy of the module.
    # Adapted from:
    # https://github.com/pytorch/pytorch/blob/65bad41cbec096aa767b3752843eddebf845726f/torch/nn/modules/module.py#L1171
    if not isinstance(module, torch.nn.Module):
        return module
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()

    # Second, re-write all parameters
    if hasattr(clone, '_parameters'):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                param = module._parameters[param_key]
                cloned = param.detach()
                clone._parameters[param_key] = cloned


    # Third, handle the buffers if necessary
    if hasattr(clone, '_buffers'):
        for buffer_key in module._buffers:
            if clone._buffers[buffer_key] is not None and \
                    clone._buffers[buffer_key].requires_grad:
                buff = module._buffers[buffer_key]
                
                cloned = buff.detach()
                clone._buffers[buffer_key] = cloned


    # Then, recurse for each submodule
    if hasattr(clone, '_modules'):
        for module_key in clone._modules:
            clone._modules[module_key] = detach_new_module(
                module._modules[module_key],
            )

    # Finally, rebuild the flattened parameters for RNNs
    # See this issue for more details:
    # https://github.com/learnables/learn2learn/issues/139
    if hasattr(clone, 'flatten_parameters'):
        clone = clone._apply(lambda x: x)
    return clone

def det3(M):
    det=M[...,0,0]*M[...,1,1]*M[...,2,2]\
        +M[...,0,1]*M[...,1,2]*M[...,2,0]\
        +M[...,0,2]*M[...,1,0]*M[...,2,1]\
        -M[...,0,2]*M[...,1,1]*M[...,2,0]\
        -M[...,0,1]*M[...,1,0]*M[...,2,2]\
        -M[...,0,0]*M[...,1,2]*M[...,2,1]
    return det

def wmean(
        x: torch.Tensor,
        weight = None,
        dim = -2,
        keepdim: bool = True,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        """
        copy from pytorch3d
        Finds the mean of the input tensor across the specified dimension.
        If the `weight` argument is provided, computes weighted mean.
        Args:
            x: tensor of shape `(*, D)`, where D is assumed to be spatial;
            weights: if given, non-negative tensor of shape `(*,)`. It must be
                broadcastable to `x.shape[:-1]`. Note that the weights for
                the last (spatial) dimension are assumed same;
            dim: dimension(s) in `x` to average over;
            keepdim: tells whether to keep the resulting singleton dimension.
            eps: minimum clamping value in the denominator.
        Returns:
            the mean tensor:
            * if `weights` is None => `mean(x, dim)`,
            * otherwise => `sum(x*w, dim) / max{sum(w, dim), eps}`.
        """
        args = {"dim": dim, "keepdim": keepdim}

        if weight is None:
            return x.mean(**args)

        if any(
            xd != wd and xd != 1 and wd != 1
            for xd, wd in zip(x.shape[-2::-1], weight.shape[::-1])
        ):
            raise ValueError("wmean: weights are not compatible with the tensor")

        return (x * weight[..., None]).sum(**args) / weight[..., None].sum(**args).clamp(
            eps
        )



class Differential_Relevant_Terms(torch.autograd.Function):
    # high oreder gradients are NOT considered!
    @staticmethod
    def forward(ctx, m, beta:float = None, eps:float=1e-7):
        U, S, VT = torch.linalg.svd(m)
        f1 = torch.ones_like(S)
        f2 = torch.ones_like(S)
        f2[...,-1]=-1
        differential_det = det3(m)
        f = torch.where((differential_det>0).unsqueeze(-1),f1,f2)
        F = torch.diag_embed(f)
        U, S, VT = U, S*f, F@VT
        ctx.U, ctx.S, ctx.VT = U, S, VT
        ctx.eps, ctx.beta = eps, beta
        if beta is None:
            local_rigid_measurement = torch.sum((S-1)**2,dim=-1)
        else:
            error_abs = torch.abs(S-1)
            local_rigid_measurement = torch.where(error_abs<beta, (S-1)**2, 2*beta*error_abs-beta**2)
            local_rigid_measurement = local_rigid_measurement.sum(dim=-1)
        R = U@VT
        return local_rigid_measurement, R, differential_det, S
    
    @staticmethod
    def backward(ctx, grad_out_r,grad_out_R,grad_out_d,grad_out_S):
        # high oreder gradients are NOT considered!
        with torch.no_grad():
            U, S, VT = ctx.U, ctx.S, ctx.VT
            eps = ctx.eps
            # reciprocal_eps = 1/eps
            beta = ctx.beta
            grad_sigma = grad_out_r.unsqueeze(-1) * 2 * (S - 1 if beta is None else (S - 1).clamp(max=beta,min=-beta))\
                        + grad_out_d.unsqueeze(-1) * torch.stack([S[...,1]*S[...,2],S[...,0]*S[...,2],S[...,0]*S[...,1]],dim=-1)\
                        + grad_out_S
            x = VT @ grad_out_R.transpose(-1,-2) @ U
            X = x.transpose(-1,-2) - x
            B = S[...,:,None]+S[...,None,:]
            rB = (1/B).clamp(min=-2, max=2)
            Z = X * rB
            M = Z+torch.diag_embed(grad_sigma)
            norm_M = torch.sum(M ** 2, [-1,-2],keepdim=True).sqrt()
            print(norm_M.mean())
            scale = torch.where(norm_M>eps, eps/norm_M,torch.ones_like(norm_M))
            grad_input = U @ (M * scale) @ VT
        return grad_input, None, None

class Safe_Cosine_Similarity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v1, v2, dim=-1, keepdim = False, eps_forward:float=1e-12, eps_backward:float=1e-12):
        v1_norm = v1.norm(dim=dim, keepdim=True)
        v2_norm = v2.norm(dim=dim, keepdim=True)
        v1_normalized = torch.div(v1, v1_norm.clamp_min(eps_forward))
        v2_normalized = torch.div(v2, v2_norm.clamp_min(eps_forward))
        cos_sim = torch.sum(v1_normalized * v2_normalized, dim = dim, keepdim=keepdim)
        ctx.save_for_backward(v1_normalized, v2_normalized, cos_sim, v1_norm, v2_norm)
        ctx.keepdim, ctx.dim, ctx.eps_backward = keepdim, dim, eps_backward
        return cos_sim, v1_normalized, v2_normalized, v1_norm, v2_norm
    
    @staticmethod
    def backward(ctx, grad_cos_sim, grad_v1_normalized, grad_v2_normalized, grad_v1_norm, grad_v2_norm):
        v1_normalized, v2_normalized, cos_sim, v1_norm, v2_norm = ctx.saved_tensors
        eps_backward, keepdim, dim = ctx.eps_backward, ctx.keepdim, ctx.dim
        if not keepdim:
            cos_sim = cos_sim.unsqueeze(dim)
            grad_cos_sim = grad_cos_sim.unsqueeze(dim)
        grad_v1 = torch.div(v2_normalized * grad_cos_sim + grad_v1_normalized - (cos_sim * grad_cos_sim + torch.sum(grad_v1_normalized * v1_normalized, dim=-1, keepdim=True)) * v1_normalized, 
                            v1_norm.clamp_min(eps_backward))
        grad_v2 = torch.div(v1_normalized * grad_cos_sim + grad_v2_normalized - (cos_sim * grad_cos_sim + torch.sum(grad_v2_normalized * v2_normalized, dim=-1, keepdim=True)) * v2_normalized, 
                            v2_norm.clamp_min(eps_backward))
        grad_v1 += v1_normalized * grad_v1_norm
        grad_v2 += v2_normalized * grad_v2_norm
        return grad_v1, grad_v2, None, None, None, None

    # @staticmethod
    # def backward(ctx, grad_cos_sim, grad_v1_normalized, grad_v2_normalized, grad_v1_norm, grad_v2_norm):
    #     v1_normalized, v2_normalized, cos_sim, v1_norm, v2_norm = ctx.saved_tensors
    #     eps_backward, keepdim, dim = ctx.eps_backward, ctx.keepdim, ctx.dim
    #     if not keepdim:
    #         cos_sim = cos_sim.unsqueeze(dim)
    #         grad_cos_sim = grad_cos_sim.unsqueeze(dim)
    #     grad_v1_normalized = grad_v1_normalized-torch.sum(grad_v1_normalized*v1_normalized,dim=-1,keepdim=True) * v1_normalized
    #     grad_v2_normalized = grad_v2_normalized-torch.sum(grad_v2_normalized*v2_normalized,dim=-1,keepdim=True) * v2_normalized
    #     grad_v1 = torch.div((v2_normalized - cos_sim * v1_normalized) * grad_cos_sim + grad_v1_normalized, 
    #                         v1_norm.clamp_min(eps_backward))
    #     grad_v2 = torch.div((v1_normalized - cos_sim * v2_normalized) * grad_cos_sim + grad_v2_normalized, 
    #                         v2_norm.clamp_min(eps_backward))
    #     grad_v1 += v1_normalized * grad_v1_norm
    #     grad_v2 += v2_normalized * grad_v2_norm
    #     return grad_v1, grad_v2, None, None, None, None

class Corresponding_Points_Alignment_Loss(torch.autograd.Function):
    # high oreder gradients are NOT considered!
    @staticmethod
    def forward(ctx, X:torch.Tensor, Y:torch.Tensor, weights:torch.Tensor, eps_forward:float, eps_backward:float):
        """
        X:(*,N,3)
        Y:(*,N,3)
        weight:(*,P,N)
        """
        X_ = X.unsqueeze(-3)
        Y_ = Y.unsqueeze(-3)
        sum_weight = weights[..., None].sum(dim=-2,keepdim=True)
        Xmu = (X_ * weights[..., None]).sum(dim=-2,keepdim=True) / sum_weight.clamp(eps_forward)
        Ymu = (Y_ * weights[..., None]).sum(dim=-2,keepdim=True) / sum_weight.clamp(eps_forward)
        Xc = X_ - Xmu
        Yc = Y_ - Ymu
        XYcov = (Xc*weights.unsqueeze(-1)).transpose(-1, -2)@ Yc

        U, S, VT = torch.linalg.svd(XYcov)
        f1 = torch.ones_like(S)
        f2 = torch.ones_like(S)
        f2[...,-1]=-1
        f = torch.where((det3(XYcov)>0).unsqueeze(-1),f1,f2)
        F = torch.diag_embed(f)
        S, R = S*f, U @ F @VT
        ctx.S, ctx.R, ctx.sum_weight = S, R, sum_weight
        ctx.eps_forward,ctx.eps_backward = eps_forward, eps_backward
        ctx.Xc, ctx.Yc = Xc, Yc
        ctx.save_for_backward(weights)
        term1=torch.sum(torch.sum(Xc**2+Yc**2,dim=-1)*weights,dim=-1)
        term2=torch.sum(S,dim=-1)
        return term1 - 2 * term2

    @staticmethod
    def backward(ctx, grad_outputs):
        with torch.no_grad():
            Xc, Yc = ctx.Xc, ctx.Yc
            weights = ctx.saved_tensors[0]
            grad_XYcov = -2 * ctx.R
            wXc = Xc * weights.unsqueeze(-1)
            wYc = Yc * weights.unsqueeze(-1)
            grad_Xc = grad_outputs[...,None,None] * (2 * wXc + wYc @ grad_XYcov.transpose(-1,-2)) # (*,P,N,3)
            grad_Yc = grad_outputs[...,None,None] * (2 * wYc + wXc @ grad_XYcov) # (*,P,N,3)
            sum_weight = ctx.sum_weight# (*,P,1,1)
            eps_forward, eps_backward = ctx.eps_forward, ctx.eps_backward
            grad_X = grad_Xc - grad_Xc.sum(dim=-2,keepdim=True) * weights.unsqueeze(-1)/sum_weight.clamp(eps_forward)
            grad_Y = grad_Yc - grad_Yc.sum(dim=-2,keepdim=True) * weights.unsqueeze(-1)/sum_weight.clamp(eps_forward)
            grad_input_X=grad_X.sum(dim=-3)
            grad_input_Y=grad_Y.sum(dim=-3)
            grad_input_w = grad_outputs[...,None] * ((Xc @ grad_XYcov) * Yc + Xc**2+Yc**2).sum(dim=-1)\
                        -((grad_Xc.sum(dim=-2,keepdim=True) * Xc + grad_Yc.sum(dim=-2,keepdim=True) * Yc)/sum_weight.clamp(eps_backward)).sum(dim=-1)\
                        
        return grad_input_X, grad_input_Y, grad_input_w, None, None
        

def differential_relative_terms(m):
    return Differential_Relevant_Terms.apply(m,1)[0]

def safe_cosine_similarity(v1, v2, dim=-1, keepdim = False, eps_forward:float=1e-7, eps_backward:float=1e-7):
    return Safe_Cosine_Similarity.apply(v1, v2, dim, keepdim, eps_forward,eps_backward)[0]

def rigid_loss(x,y,w):
    return Corresponding_Points_Alignment_Loss.apply(x,y,w,0,0)

def corresponding_points_alignment_loss(X:torch.Tensor, Y:torch.Tensor, weights:torch.Tensor, eps_forward:float=1e-7, eps_backward:float=1e-7):
    # The backward process suffers from small weights very much
    return Corresponding_Points_Alignment_Loss.apply(X,Y,weights,eps_forward,eps_backward)

if __name__ =='__main__':
    x = torch.randn([5,100,3],dtype=torch.double).cuda()
    y = torch.randn([5,100,3],dtype=torch.double).cuda()
    w = torch.softmax(torch.rand([5,6,100],dtype=torch.double),dim=-2).cuda()
    x.requires_grad=True
    y.requires_grad=True
    w.requires_grad=True
    torch.autograd.gradcheck(safe_cosine_similarity, [x,y])
    torch.autograd.gradgradcheck(safe_cosine_similarity, [x,y])
    