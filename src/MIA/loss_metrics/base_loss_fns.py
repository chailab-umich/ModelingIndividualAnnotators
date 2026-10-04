import torch
from MIA.metrics import ccc, weighted_ccc, pearson

def probability_loss(model_output, targets):
    model_output = model_output['probability_logits']
    bs = model_output.shape[0]
    model_output = torch.log_softmax(model_output.view(bs, -1), dim=-1)
    targets = targets.view(bs, -1)
    return torch.nn.functional.cross_entropy(model_output, targets)

def probability_loss_no_softmax(model_output, targets):
    # Not sure what to do for this right now 
    # will do what didi did and replace 0 with 1e-8
    if type(model_output) == tuple:
        model_output = model_output[0]
    if model_output is None:
        return None # This means brent optimisation failed so this batch should be skipped for the probability distribution
    pre_prob_conversion = model_output
    bs = model_output.shape[0]
    model_output_prob = model_output.view(bs,-1)
    model_output_prob = (model_output_prob / model_output_prob.sum(dim=-1).unsqueeze(dim=-1)) + 1e-8
    targets = targets.view(bs, -1)
    loss = cross_entropy_no_softmax(model_output_prob, targets)

    if loss.isnan():
        print(pre_prob_conversion, model_output_prob, targets)
        raise ValueError('NaN in cross entropy loss')

    return loss

def ccc_loss(preds, true):
    return torch.tensor(1) - ccc(preds, true)
    # preds = preds.squeeze() + 1
    # true = true.squeeze() + 1
    # return (true - preds).pow(2).sum()/(true*preds).sum()

def pearson_loss(preds, true):
    return torch.tensor(1) - pearson(preds, true)

def weighted_ccc_loss(preds, true):
    return torch.tensor(1) - weighted_ccc(preds, true)

def kldiv(target_mean, target_var, mean, log_var):
    # Epsilon is fairly high but due to likert scale labels and some samples having small variance, it can cause gradients to explode with a lower epsilon
    # likely fine anyway (think of quantization of 1-7 labels)
    target_var[target_var<1e-4] = 1e-4 # Add small epsilon when variance in target label is 0 to prevent division by 0 # Assume a variance of 1

    if target_var.device != log_var.device:
        target_var = target_var.to(log_var.device) # Some validation tensors are too large and require moving to CPU during calculation

    # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    # Want to learn KL(Prediction || Target)
    log_frac = torch.log(target_var) - log_var
    log_var_exp = log_var.exp()
    mean_diff_squared = torch.pow(mean-target_mean,2)
    numerator = log_var_exp + mean_diff_squared
    frac = numerator / target_var
    kld_term = (log_frac + frac - 1)*0.5
    return kld_term.mean()

def cross_entropy_no_softmax(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculate cross entropy loss without applying softmax to input.
    
    Args:
        input: Input tensor
        target: Target tensor
        
    Returns:
        Cross entropy loss
    """
    return torch.mean(-torch.sum(target * torch.log(input), 1))
