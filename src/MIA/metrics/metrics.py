import torch

def ccc(prediction: torch.Tensor, ground_truth: torch.Tensor, multiple_annotators: bool = False) -> torch.Tensor:
    if len(prediction.shape) == 1:
        prediction = prediction.view(prediction.shape[0], 1)
    if len(ground_truth.shape) == 1:
        ground_truth = ground_truth.view(ground_truth.shape[0], 1)
    if not multiple_annotators:
        assert prediction.shape[-1] == 1, f'Final dimension of prediction should be of shape 1 for CCC, but found {prediction.shape=}'
        assert prediction.shape == ground_truth.shape and len(prediction.shape) == 2, f'CCC requires 2D inputs of the same shape, but found {prediction.shape=} {ground_truth.shape=}'
    else:
        assert len(prediction.shape) == 2, f'When multiple annotators input should be (batch size, num annotators) for CCC calculation'
    mean_gt = torch.mean(ground_truth, 0)
    mean_pred = torch.mean(prediction, 0)
    var_gt = torch.var(ground_truth, 0)
    var_pred = torch.var(prediction, 0)
    v_pred = prediction - mean_pred
    v_gt = ground_truth - mean_gt
    if not torch.count_nonzero(v_pred): # Add small epsilon to prevent NaN error when predictions or gt are all the same value
        v_pred = v_pred + 1e-8
    if not torch.count_nonzero(v_gt):
        v_gt = v_gt + 1e-8
    v_pred[v_pred == 0] = 1e-8
    v_gt[v_gt == 0] = 1e-8
    cor = torch.sum(v_pred * v_gt, dim=0) / (torch.sqrt(torch.sum(v_pred ** 2, dim=0)) * torch.sqrt(torch.sum(v_gt ** 2, dim=0)))
    sd_gt = torch.std(ground_truth)
    sd_pred = torch.std(prediction, dim=0)
    numerator=2*cor*sd_gt*sd_pred
    denominator=var_gt+var_pred+(mean_gt-mean_pred)**2
    denominator[denominator == 0] = 1e-8
    ccc = numerator/denominator
    return ccc

def pearson(prediction: torch.Tensor, ground_truth: torch.Tensor, multiple_annotators: bool = False) -> torch.Tensor:
    """
    Pearson correlation r over dim=0 (batch dimension is 0).
    Mirrors the behavior/shape checks of your CCC() and is numerically safe.
    """
    if len(prediction.shape) == 1:
        prediction = prediction.view(prediction.shape[0], 1)
    if len(ground_truth.shape) == 1:
        ground_truth = ground_truth.view(ground_truth.shape[0], 1)

    if not multiple_annotators:
        assert prediction.shape[-1] == 1, f'Final dimension of prediction should be 1, but found {prediction.shape=}'
        assert prediction.shape == ground_truth.shape and len(prediction.shape) == 2, \
            f'Pearson requires 2D inputs of the same shape, but found {prediction.shape=} {ground_truth.shape=}'
    else:
        assert len(prediction.shape) == 2, \
            'When multiple_annotators=True, input should be (batch_size, num_annotators)'

    # means over batch
    mean_pred = prediction.mean(dim=0)
    mean_gt   = ground_truth.mean(dim=0)

    # center
    v_pred = prediction - mean_pred
    v_gt   = ground_truth - mean_gt

    # guard against all-constant columns producing zero std (and NaNs)
    # add tiny noise only where needed (no grad issues; it’s additive)
    # (Uses boolean mask to avoid unnecessary ops.)
    eps = 1e-12
    zero_pred = (v_pred.abs().sum(dim=0) == 0)
    zero_gt   = (v_gt.abs().sum(dim=0) == 0)
    if zero_pred.any():
        v_pred[:, zero_pred] = v_pred[:, zero_pred] + eps
    if zero_gt.any():
        v_gt[:, zero_gt] = v_gt[:, zero_gt] + eps

    # covariance and std (use unbiased=False for stability/compat with MSE-style losses)
    cov = (v_pred * v_gt).mean(dim=0)  # E[(X-EX)(Y-EY)]
    std_pred = torch.sqrt((v_pred ** 2).mean(dim=0) + eps)
    std_gt   = torch.sqrt((v_gt   ** 2).mean(dim=0) + eps)

    r = cov / (std_pred * std_gt)
    return r

def weighted_ccc(prediction: torch.Tensor,
                 ground_truth: torch.Tensor,
                 multiple_annotators: bool = False,
                 lam: float = 0.1) -> torch.Tensor:
    # --- keep your existing shape checks etc. ---
    if len(prediction.shape) == 1:
        prediction = prediction.view(prediction.shape[0], 1)
    if len(ground_truth.shape) == 1:
        ground_truth = ground_truth.view(ground_truth.shape[0], 1)
    if not multiple_annotators:
        assert prediction.shape[-1] == 1, \
            f'Final dimension of prediction should be of shape 1 for CCC, but found {prediction.shape=}'
        assert prediction.shape == ground_truth.shape and len(prediction.shape) == 2, \
            f'CCC requires 2D inputs of the same shape, but found {prediction.shape=} {ground_truth.shape=}'
    else:
        assert len(prediction.shape) == 2, \
            f'When multiple annotators input should be (batch size, num annotators) for CCC calculation'

    mean_gt = torch.mean(ground_truth, 0)
    mean_pred = torch.mean(prediction, 0)
    var_gt = torch.var(ground_truth, 0)
    var_pred = torch.var(prediction, 0)

    v_pred = prediction - mean_pred
    v_gt = ground_truth - mean_gt
    if not torch.count_nonzero(v_pred):
        v_pred = v_pred + 1e-8
    if not torch.count_nonzero(v_gt):
        v_gt = v_gt + 1e-8

    # Pearson correlation
    cor = torch.sum(v_pred * v_gt, dim=0) / (
        torch.sqrt(torch.sum(v_pred ** 2, dim=0)) *
        torch.sqrt(torch.sum(v_gt ** 2, dim=0))
    )

    # Calibration penalties
    sd_gt = torch.std(ground_truth)
    sd_pred = torch.std(prediction, dim=0)
    mean_pen = (mean_gt - mean_pred) ** 2
    var_pen = (sd_gt - sd_pred) ** 2

    # Weighted objective: Pearson - λ * (penalties)
    weighted = cor - lam * (mean_pen + var_pen)

    return weighted

def total_variation(inp: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
    """Calculate the total variation distance between two probability distributions.
    
    Args:
        inp: Input probability distribution
        tar: Target probability distribution
        
    Returns:
        Total variation distance
    """
    return (torch.abs(inp-tar).sum(dim=[-1,-2])/2).mean()

def JSD(inp: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
    """Calculate the Jensen-Shannon divergence between two probability distributions.
    
    Args:
        inp: Input probability distribution
        tar: Target probability distribution
        
    Returns:
        Jensen-Shannon divergence
    """
    # Add small epsilon to prevent log(0)
    inp = torch.where(inp == 0, torch.tensor(1e-8, device=inp.device), inp)
    tar = torch.where(tar == 0, torch.tensor(1e-8, device=tar.device), tar)
    
    M = (inp + tar)/2
    kld_inp_m = inp*(torch.log2(inp) - torch.log2(M))
    kld_tar_m = tar*(torch.log2(tar) - torch.log2(M))
    return ((kld_inp_m.sum(dim=[-1,-2])+kld_tar_m.sum(dim=[-1,-2]))*0.5).mean()

def total_variation_1d(inp: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
    """Calculate the 1D total variation distance between two probability distributions.
    
    Args:
        inp: Input probability distribution
        tar: Target probability distribution
        
    Returns:
        1D total variation distance
    """
    return torch.pow(inp-tar, 2).sum(dim=-1).mean()

def JSD_1d(inp: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
    """Calculate the 1D Jensen-Shannon divergence between two probability distributions.
    
    Args:
        inp: Input probability distribution
        tar: Target probability distribution
        
    Returns:
        1D Jensen-Shannon divergence
    """
    # Add small epsilon to prevent log(0)
    inp = torch.where(inp == 0, torch.tensor(1e-8, device=inp.device), inp)
    tar = torch.where(tar == 0, torch.tensor(1e-8, device=tar.device), tar)
    
    M = (inp + tar)/2
    kld_inp_m = inp*(torch.log2(inp) - torch.log2(M))
    kld_tar_m = tar*(torch.log2(tar) - torch.log2(M))
    return ((kld_inp_m.sum(dim=-1)+kld_tar_m.sum(dim=-1))*0.5).mean()
