'''
Code for deep evidential emotion regression (DEER)

Author: Wen 2022

See: https://github.com/W-Wu/DEER/blob/dev/deep_evidential_emotion_regression.py
'''

import torch
import torch.nn as nn
import numpy as np

# Should still work with y having NaN padding in the data since these are probably elementwise calculation
def NIG_NLL(y, gamma, v, alpha, beta, reduce=False):
    # Adapted from https://github.com/hxu296/torch-evidental-deep-learning
    twoBlambda = 2*beta*(1+v)
    nll = 0.5*torch.log(np.pi/v)  \
        - alpha*torch.log(twoBlambda)  \
        + (alpha+0.5) * torch.log(v*(y-gamma)**2 + twoBlambda)  \
        + torch.lgamma(alpha)  \
        - torch.lgamma(alpha+0.5)
    return torch.mean(nll) if reduce else nll

class DenseNormalGamma(nn.Module):
    # Adapted from https://github.com/hxu296/torch-evidental-deep-learning
    def __init__(self, units_in, units_out):
        super(DenseNormalGamma, self).__init__()
        self.units_in = int(units_in)
        self.units_out = int(units_out)
        self.linear = nn.Linear(units_in, 4 * units_out)

    def evidence(self, x):
        softplus = nn.Softplus(beta=1)
        return softplus(x)

    def forward(self, x):
        output = self.linear(x)
        mu, logv, logalpha, logbeta = torch.split(output, self.units_out, dim=-1)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return torch.cat(tensors=(mu, v, alpha, beta), dim=-1)

    def compute_output_shape(self):
        return (self.units_in, 4 * self.units_out)
   

# Should still work with y having NaN padding in the data since these are elementwise calculation
def NIG_Reg_phi(y, gamma, v, alpha, beta, reduce=False):
    error = torch.abs(y-gamma)
    evi = 1/(beta * (1 + v) / (v * (alpha-1)))    #pred_std
    reg = error*evi

    # When alpha is exactly 1 we get inf in error (as gamma was calculated using beta/(alpha-1) for aleatoric uncertainty)
    # evi is then 1/(.../0) -> 1/inf -> 0. error*evi is then 0*inf which is undefined and so value becomes NaN
    # So we will set NaN to zero as generally smaller values trend towards zero
    reg[reg.isnan()] = 0

    if reg.isnan().any():
        print(f'NIG_Reg_phi loss check: {y=} {gamma=} {v=} {alpha=} {beta=} {error=} {evi=} {reg=}')

    return torch.mean(reg) if reduce else reg

# Need to calculate variance below ignoring NaN
def nanvar(tensor, dim=None, keepdim=False):
    tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
    output = (tensor - tensor_mean).square().nanmean(dim=dim, keepdim=keepdim)
    return output

# In the original DEER code they do the following before calling DEER_loss:
# If training we get label = batch labels
# the label is then padded to the length of label with the maximum annotators
# Now have label_padded and label_mask
# assign label = label_padded
# now if the output_dim == 1 we select the output_idx value of the last rater in the label
# I'm confused how the label has more than one value per rater? Activation/Valence? 
# The last rater is the mean rater, so label_ref = self.check_nan(mean rating)
# and then the last rater is removed from the label, so, label = raters without mean rater
# check nan goes through transpoe of label_ref, and if all values are equal it adds small random permutation to prevent NaN calculation when computing PCC 
# for our code, label=individual_act_labels, label_ref=act
# not sure how this works across activation and valence, they may need to be stacked on top of each other 
def DEER_loss(label, label_ref, evidential_output, avg_rater=True, coeff_reg=1.0,ref_only=False,coeff_ref=0.0):
    label_var = nanvar(label,dim=1)
    gamma, v, alpha, beta = evidential_output  #gamma.shape: [batch_size,output_dim]
    aleatoric = beta /  (alpha - 1)
    # Label_ref is just the mean rater performance so operations are unchanged for label_ref
    loss_reg = NIG_Reg_phi(label_ref, gamma, v, alpha, beta) + NIG_Reg_phi(label_var, aleatoric, v, alpha, beta) 

    loss_nll_ref = NIG_NLL(label_ref, gamma, v, alpha, beta)    

    if ref_only:
        return loss_nll_ref + coeff_reg * loss_reg

    # Now we want to do NLL on each individual rater, in original code this is done via loop but 
    # we will instead use torch broadcasting
    label_mask = torch.isnan(label.sum(dim=-1)) # True when annotator is missing
    label[label_mask] = 0 # Set to 0 to avoid NaN ending up in the model weights, impact removed by mask below

    loss_nll_all = NIG_NLL(label, gamma[:,None,:], v[:,None,:], alpha[:,None,:], beta[:,None,:])
    loss_nll_all[label_mask] = 0
    loss_nll_all = torch.sum(loss_nll_all, dim=1)

    if avg_rater:
        # Invert mask prior to sum to get number of present annotators in each utterance
        loss_nll_all /= torch.sum(~label_mask,-1,keepdim=True)

    full_loss = loss_nll_all + coeff_reg * loss_reg + coeff_ref * loss_nll_ref
    if full_loss.isnan().any():
        print(f'DEER Loss NaN detected: {loss_nll_all.isnan().any()=} {loss_reg.isnan().any()=} {loss_nll_ref.isnan().any()=}')

    return full_loss