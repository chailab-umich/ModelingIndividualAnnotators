import torch
from tqdm import tqdm
import numpy as np

def find_mode(ordinal_votes):
    """Calculate the mode (most common value) for each sample in ordinal votes."""
    mv = []
    for sample in tqdm(ordinal_votes, desc='Calculating majority vote'):
        mv.append(sample[~sample.isnan()].mode()[0])
    return torch.stack(mv)

def calculate_inter_rater_kappa(ordinal_values, annotator_map):
    """Calculate inter-rater kappa between all pairs of annotators."""
    kappa_matrix = torch.zeros((len(annotator_map), len(annotator_map)), device='cpu')
    original_device = ordinal_values.device
    curr_device = original_device
    possible_ordinal_values = torch.as_tensor(list(range(1,8)), device=original_device)
    annotator_idxs = range(len(annotator_map))
    
    for ann1 in tqdm(annotator_idxs):
        relevant_utterances = ~ordinal_values[:,ann1].isnan()
        masker = ~ordinal_values[relevant_utterances].isnan()
        device = 'cpu' if relevant_utterances.sum() > 1000 else original_device
        
        if curr_device != device:
            curr_device = device
            ordinal_values = ordinal_values.to(device, non_blocking=True)
            possible_ordinal_values = possible_ordinal_values.to(device, non_blocking=True)
            relevant_utterances = relevant_utterances.to(device, non_blocking=True)
            masker = masker.to(device, non_blocking=False)
            
        agreement_with_others = ((ordinal_values[relevant_utterances,:] == ordinal_values[relevant_utterances,ann1,None]).sum(dim=0))
        ordinal_count_other = (ordinal_values[relevant_utterances,:,None] == possible_ordinal_values).sum(dim=0)
        N = (~ordinal_values[relevant_utterances,:].isnan()).sum(dim=0)
        ordinal_count_ann1 = (ordinal_values[relevant_utterances,ann1,None] == possible_ordinal_values)
        ordinal_count_ann1 = (masker[:,:,None]*ordinal_count_ann1[:,None,:]).sum(dim=0)
        sum_ef = ((ordinal_count_ann1*ordinal_count_other).sum(dim=-1))/N
        kappa = (agreement_with_others-sum_ef)/(N-sum_ef)
        kappa[N<2] = torch.nan
        kappa = kappa.to('cpu', non_blocking=True)
        kappa_matrix[ann1,:] = kappa
        kappa_matrix[:,ann1] = kappa
        
    return kappa_matrix

def calculate_majority_vote_kappa(ordinal_values, ordinal_mv, annotator_map):
    """Calculate kappa between each annotator and majority vote."""
    kappa_matrix = torch.zeros((len(annotator_map)), device='cpu')
    possible_ordinal_values = torch.as_tensor(list(range(1,8)), device=ordinal_values.device)
    annotator_idxs = range(len(annotator_map))
    
    for ann1 in tqdm(annotator_idxs):
        relevant_utterances = ~ordinal_values[:,ann1].isnan()
        agreement_with_mv = ((ordinal_mv[relevant_utterances] == ordinal_values[relevant_utterances,ann1]).sum(dim=0))
        ordinal_count_mv = (ordinal_mv[relevant_utterances,None] == possible_ordinal_values).sum(dim=0)
        N = relevant_utterances.sum()
        ordinal_count_ann1 = (ordinal_values[relevant_utterances,ann1,None] == possible_ordinal_values).sum(dim=0)
        sum_ef = ((ordinal_count_ann1*ordinal_count_mv).sum(dim=-1))/N
        
        if N == sum_ef and N > 1:
            kappa = torch.nan
        else:
            kappa = (agreement_with_mv-sum_ef)/(N-sum_ef)
            kappa[N<2] = torch.nan
            kappa = kappa.to('cpu', non_blocking=True)
        kappa_matrix[ann1] = kappa
        
    return kappa_matrix

def kappas_to_minp_majp(act_ir_kappa, act_mv_kappa, val_ir_kappa, val_mv_kappa):
    """Convert kappa matrices to minority and majority annotator groups."""
    idxs_to_ignore = np.diag_indices(act_ir_kappa.shape[0])
    act_ir_kappa[idxs_to_ignore] = torch.nan
    val_ir_kappa[idxs_to_ignore] = torch.nan
    all_kappa = torch.cat((act_ir_kappa, act_mv_kappa[:,None], val_ir_kappa, val_mv_kappa[:,None]), dim=1)
    avg_kappa = all_kappa.nanmean(dim=-1)
    kappa_values = avg_kappa[~avg_kappa.isnan()].cpu()
    indices = torch.arange(len(avg_kappa))[~avg_kappa.isnan()]
    count, edges = np.histogram(kappa_values, bins=20)
    print('kappa histogram:', count, edges)
    min_p_annotators = set(indices[kappa_values < edges[int((len(edges)+1)/2)]].tolist())
    maj_p_annotators = set(indices[kappa_values >= edges[int((len(edges)+1)/2)]].tolist())
    return min_p_annotators, maj_p_annotators 