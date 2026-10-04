import torch
from .base_loss_fns import ccc_loss, pearson_loss, probability_loss_no_softmax, probability_loss, kldiv, weighted_ccc_loss
from .DEER import DEER_loss

class LossFunction: # None means these features are disabled for this task
    def __init__(self, name, required_model_outputs, required_target_labels, calculate_kde=False, sparsity=None, after_warmup=False, validation_only=False):
        self.name = name
        self.required_model_outputs = required_model_outputs
        self.required_target_labels = required_target_labels
        self.sparsity = sparsity
        self.after_warmup = after_warmup
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.calculate_kde = calculate_kde
        self.validation_only = validation_only

def unscale(x, new_min=1, new_max=9):
    minv = -1
    maxv = 1
    return new_min + (x - minv)*(new_max - new_min)/(maxv - minv)

class CRPS_Discrete(LossFunction):
    def __init__(self, *args, **kwargs):
        kwargs['required_model_outputs'] = ['soft_act_preds', 'soft_val_preds']
        kwargs['required_target_labels'] = ['y_padded_individual_annotators_act', 'y_padded_individual_annotators_val']
        self.num_ordinal_bins = kwargs['num_ordinal_bins']
        del kwargs['num_ordinal_bins']
        super().__init__(*args, **kwargs)

    def __call__(self, model_output, targets, masks):
        # coral_processes = targets['CORALProcesses']
        soft_act_preds, soft_val_preds = model_output['soft_act_preds'], model_output['soft_val_preds']
        y_act, y_val = targets['y_padded_individual_annotators_act'], targets['y_padded_individual_annotators_val']
        soft_act_preds = soft_act_preds[~y_act.isnan()].view(-1, self.num_ordinal_bins)
        soft_val_preds = soft_val_preds[~y_val.isnan()].view(-1, self.num_ordinal_bins)
        y_act = y_act[~y_act.isnan()]
        y_val = y_val[~y_val.isnan()]
        # soft_act_preds = coral_processes['act'](soft_act_preds[~soft_act_preds.isnan()])
        # soft_val_preds = coral_processes['val'](soft_val_preds[~soft_val_preds.isnan()])
        y_act = unscale(y_act, new_min=1, new_max=self.num_ordinal_bins).round().to(soft_act_preds.device)
        y_val = unscale(y_val, new_min=1, new_max=self.num_ordinal_bins).round().to(soft_val_preds.device)
        act_crps = crps_discrete(soft_act_preds, y_act.long())
        val_crps = crps_discrete(soft_val_preds, y_val.long())
        return {'CRPS Activation': act_crps, 'CRPS Valence': val_crps}

def crps_discrete(probs, y):
    B,K = probs.shape
    cdf = probs.cumsum(1)
    ks  = torch.arange(K, device=probs.device).view(1,K)
    F   = (ks >= y.view(B,1)).float()
    return ((cdf - F)**2).sum(1).mean()

# Task 1 and Task 1 CCC use almost the exact same code so configure a helper to prevent code repetition
def task1_helper(model_output, targets, masks):
    ind_evals = targets['y_padded_individual_annotators_act']
    ind_val_evals = targets['y_padded_individual_annotators_val']

    soft_act_preds, soft_val_preds = model_output['soft_act_preds'], model_output['soft_val_preds']
    if type(soft_act_preds) == list:
        device = ind_evals.device
        soft_act_preds = torch.nn.utils.rnn.pad_sequence([y for x in soft_act_preds for y in torch.nn.utils.rnn.unpad_sequence(x, batch_first=True, lengths=(~x.isnan()).sum(dim=-1))], batch_first=True, padding_value=torch.nan).squeeze().to(device)
        soft_val_preds = torch.nn.utils.rnn.pad_sequence([y for x in soft_val_preds for y in torch.nn.utils.rnn.unpad_sequence(x, batch_first=True, lengths=(~x.isnan()).sum(dim=-1))], batch_first=True, padding_value=torch.nan).squeeze().to(device)
    # if len(soft_act_preds.shape) > 2:
    #     # Mean and squeeze the middle dimension (in case the model output is more than one observation per annotator)
    #     soft_act_preds = soft_act_preds.nanmean(dim=-1).squeeze()
    #     soft_val_preds = soft_val_preds.nanmean(dim=-1).squeeze()

    if len(soft_act_preds.shape) > 2:
        # Multiple annotators used to predict one annotator - duplicate target labels for each of these annotators
        # print('Before repeating example of first 2 samples')
        # print(soft_act_preds[:2])
        # print(ind_evals[:2])
        # print('----------------------------------------')
        ind_evals = ind_evals.unsqueeze(dim=-1).repeat(1,1,soft_act_preds.shape[-1])
        ind_val_evals = ind_val_evals.unsqueeze(dim=-1).repeat(1,1,soft_act_preds.shape[-1])
        # Now flatten the predictions into two dimensional for CCC 
        # print('Repeated final dimension examples')
        # print(soft_act_preds[:2])
        # print(ind_evals[:2])
        # print('----------------------------------------')
        ind_evals = ind_evals.view(ind_evals.shape[0], -1)
        ind_val_evals = ind_val_evals.view(ind_val_evals.shape[0], -1)
        soft_act_preds = soft_act_preds.view(soft_act_preds.shape[0], -1)
        soft_val_preds = soft_val_preds.view(soft_val_preds.shape[0], -1)
        # print('flattened final dimension examples')
        # print(soft_act_preds[:2])
        # print(ind_evals[:2])
        # print('----------------------------------------')


    # Model output soft labels will already be lined up with the target labels thanks to batch_collator
    # Now we have target and labels of shape batch_size x num_annotators with nan where samples have no annotators
    # so we just want to flatten these lists into the non-nan values
    m1 = ~soft_act_preds.isnan() #, ~soft_val_preds.isnan(), ~soft_act_labels.isnan(), ~soft_val_labels.isnan()
    # print('loss shapes', soft_act_preds.shape, ind_evals.shape)        
    act_preds = soft_act_preds[m1]
    val_preds = soft_val_preds[m1]
    # print('act preds shape in task 1 helper', act_preds.shape, m1.shape, ind_evals.shape)
    if soft_act_preds.shape != ind_evals.shape:
        # If the shapes don't match then the validation set contained unseen annotators
        # should just be ignored 
        raise ValueError(f'Unseen annotators found, this is no longer supported. Model predicted {soft_act_preds.shape} values and target has {ind_evals.shape} values')
        print('Warning -- ignoring unseen annotators in loss calculation')
        print(soft_act_preds.shape, ind_evals.shape)
        new_mask = torch.fill(torch.empty(ind_evals.shape), False)
        new_mask[m1] = True
        m1 = new_mask
    act_targets = ind_evals[m1]
    val_targets = ind_val_evals[m1]
    return act_preds, val_preds, act_targets, val_targets

class BaselineLoss(LossFunction):
    def __init__(self, *args, **kwargs):
        kwargs['required_model_outputs'] = ['probability_logits']
        kwargs['required_target_labels'] = ['y_kde_2d_probability']
        super().__init__(*args, **kwargs)
    def __call__(self, model_output, targets, masks):
        loss = probability_loss(model_output, targets['y_kde_2d_probability'])
        return {'Cross-entropy Loss': loss}

class PreTrainLoss(LossFunction):
    def __init__(self, *args, **kwargs):
        if kwargs['loss_fn'] == 'CCC':
            self.loss_fn = ccc_loss
        elif kwargs['loss_fn'] == 'MSE':
            self.loss_fn = torch.nn.functional.mse_loss
        else:
            raise ValueError(f'Unknown loss type {kwargs["loss_fn"]}')
        self.train_log_var = kwargs['train_log_var']
        self.loss_string = kwargs['loss_fn']
        del kwargs['loss_fn'], kwargs['train_log_var']
        kwargs['required_model_outputs'] = []
        kwargs['required_target_labels'] = []
        super().__init__(*args, **kwargs)

    def __call__(self, model_output, targets, masks, helper_outputs=None):
        act = model_output['act']
        val = model_output['val']
        y_act = targets['y_act']
        y_val = targets['y_val']

        act_loss = self.loss_fn(act, y_act)
        val_loss = self.loss_fn(val, y_val)

        loss_out = {f'Pre-train Activation {self.loss_string} Loss': act_loss, f'Pre-train Valence {self.loss_string} Loss': val_loss}
        if self.train_log_var:
            y_act_var = targets['y_act_variance']
            y_val_var = targets['y_val_variance']
            # y_act_var[y_act_var<1e-6] = 1e-6
            # y_val_var[y_val_var<1e-6] = 1e-6

            act_log_var = model_output['act_log_var']
            val_log_var = model_output['val_log_var']

            act_log_var_loss = self.loss_fn(act_log_var.exp(), y_act_var)
            val_log_var_loss = self.loss_fn(val_log_var.exp(), y_val_var)

            loss_out[f'Pre-train Activation Log Variance {self.loss_string} Loss'] = act_log_var_loss
            loss_out[f'Pre-train Valence Log Variance {self.loss_string} Loss'] = val_log_var_loss
        return loss_out
def soft_round_to_grid(x: torch.Tensor, grid: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """
    Nan-safe, differentiable 'rounding' of x to the discrete grid via softmax over RBF logits.
    x:    [...], float tensor (can contain NaNs)
    grid: [K],   sorted tensor of bin centers (e.g., -1, -0.75, ..., 1)
    tau:  >0,    temperature; lower = sharper (closer to hard rounding)
    """
    if x.numel() == 0:
        return x

    # Handle NaNs (skip them during soft assignment, then put them back)
    nan_mask = torch.isnan(x)
    if nan_mask.any():
        x_work = x[~nan_mask]
    else:
        x_work = x

    # [..., 1] - [K] -> [..., K]
    diffsq = (x_work.unsqueeze(-1) - grid)**2
    logits = -diffsq / tau
    p = torch.softmax(logits, dim=-1)         # [..., K]
    x_tilde = (p * grid).sum(dim=-1)          # [...]

    if nan_mask.any():
        out = x.clone()
        out[~nan_mask] = x_tilde
        return out
    else:
        return x_tilde
class Task1Loss(LossFunction):
    def __init__(self, *args, **kwargs):
        if kwargs['loss_fn'] == 'CCC':
            self.loss_fn = ccc_loss
        elif kwargs['loss_fn'] == 'MSE':
            self.loss_fn = torch.nn.functional.mse_loss
        elif kwargs['loss_fn'] == 'weighted_ccc':
            self.loss_fn = weighted_ccc_loss
        elif kwargs['loss_fn'] == 'softmax5_ccc':
            # Just need to safely softmax the values before ccc
            self.loss_fn = ccc_loss
            self.safe_softmax = True
            self.softmax_size = 5
        elif kwargs['loss_fn'] == 'softmax9_ccc':
            self.loss_fn = ccc_loss
            self.safe_softmax = True
            self.softmax_size = 9
        elif kwargs['loss_fn'] == 'pearson':
            self.loss_fn = pearson_loss
        else:
            raise ValueError(f'Unknown loss type {kwargs["loss_fn"]}')
        self.loss_string = kwargs['loss_fn']
        del kwargs['loss_fn']
        kwargs['required_model_outputs'] = ['soft_act_preds', 'soft_val_preds']
        kwargs['required_target_labels'] = ['y_padded_individual_annotators_act', 'y_padded_individual_annotators_val']
        super().__init__(*args, **kwargs)
        if hasattr(self, 'safe_softmax') and self.safe_softmax:
            self.grid = torch.linspace(-1, 1, self.softmax_size).to(self.device)
            self.tau = 0.1

    def __call__(self, model_output, targets, masks, helper_outputs=None):
        ind_evals = targets['y_padded_individual_annotators_act']
        ind_val_evals = targets['y_padded_individual_annotators_val']

        soft_act_preds, soft_val_preds = model_output['soft_act_preds'], model_output['soft_val_preds']

        if type(soft_act_preds) == list:
            device = ind_evals.device
            soft_act_preds = torch.nn.utils.rnn.pad_sequence([y for x in soft_act_preds for y in torch.nn.utils.rnn.unpad_sequence(x, batch_first=True, lengths=(~x.isnan()).sum(dim=-1))], batch_first=True, padding_value=torch.nan).squeeze().to(device)
            soft_val_preds = torch.nn.utils.rnn.pad_sequence([y for x in soft_val_preds for y in torch.nn.utils.rnn.unpad_sequence(x, batch_first=True, lengths=(~x.isnan()).sum(dim=-1))], batch_first=True, padding_value=torch.nan).squeeze().to(device)
        # if len(soft_act_preds.shape) > 2: # Old method where we use the average of output_size to train all predictions
        #     # Mean and squeeze the last dimension (in case the model output is more than one observation per annotator)
        #     soft_act_preds = soft_act_preds.nanmean(dim=-1).squeeze()
        #     soft_val_preds = soft_val_preds.nanmean(dim=-1).squeeze()

        if helper_outputs is None: # For eval it makes sense to track the task1_helper outputs rather than the inputs to task1_helper so allow this to be overridden
            act_preds, val_preds, act_targets, val_targets = task1_helper(model_output, targets, masks)
        else:
            act_preds, val_preds, act_targets, val_targets = helper_outputs

        if hasattr(self, 'safe_softmax') and self.safe_softmax:
            g = self.grid.to(act_preds.device)
            act_preds = soft_round_to_grid(act_preds, g, self.tau)
            g = self.grid.to(val_preds.device)
            val_preds = soft_round_to_grid(val_preds, g, self.tau)
        act_loss = self.loss_fn(act_preds, act_targets)
        val_loss = self.loss_fn(val_preds, val_targets)
        if torch.isnan(act_loss) or torch.isnan(val_loss):
            print('NaN loss detected in Task1Loss')
            print('act_preds', act_preds)
            print('act_targets', act_targets)
            print('val_preds', val_preds)
            print('val_targets', val_targets)
            print('act_loss', act_loss)
            print('val_loss', val_loss)
            raise ValueError('NaN loss detected in Task1Loss')
        return {f'Individual Annotator Activation {self.loss_string} Loss': act_loss, f'Individual Annotator Valence {self.loss_string} Loss': val_loss}

class CCCIndLoss(LossFunction):
    def __init__(self, *args, **kwargs):
        if kwargs['loss_fn'] == 'CCC':
            self.loss_fn = ccc_loss
        elif kwargs['loss_fn'] == 'MSE':
            self.loss_fn = torch.nn.functional.mse_loss
        else:
            raise ValueError(f'Unknown loss type {kwargs["loss_fn"]}')
        self.loss_string = kwargs['loss_fn']
        del kwargs['loss_fn']
        kwargs['required_model_outputs'] = ['soft_act_preds', 'soft_val_preds']
        kwargs['required_target_labels'] = ['y_padded_individual_annotators_act', 'y_padded_individual_annotators_val', 'annotator_masks']
        super().__init__(*args, **kwargs)

    def convert_to_individual_annotator_level(self, predictions, ground_truth, weight_mask, output_mask, ringbuffer, ringbuffer_target):
        device = predictions.device
        dtype  = predictions.dtype
        R = ringbuffer.size(1)

        # 1) Flatten valid batch predictions/targets and their annotator ids
        flat_pred   = predictions[output_mask]              # (T,)
        flat_target = ground_truth[output_mask]                  # (T,)
        annot_ids   = weight_mask.to(torch.long)            # (T,)

        # 2) Get the unique annotators present in this batch, compact to 0..Z-1
        #    inv maps each element in annot_ids to its compact group index [0..Z-1]
        unique_ann, inv = torch.unique(weight_mask, return_inverse=True, sorted=True)  # unique_ann: (Z,)
        Z = unique_ann.numel()

        # 3) Count how many predictions each present annotator has in this batch
        counts = torch.bincount(inv, minlength=Z)           # (Z,)
        max_count = counts.max().item()
        X = R + max_count                                   # total padded width per annotator

        # 4) Allocate outputs (padded with NaN)
        per_ann_pred   = torch.full((Z, X), float('nan'), device=device, dtype=dtype)
        per_ann_target = torch.full((Z, X), float('nan'), device=device, dtype=dtype)

        # 5) Place ringbuffers into the left R columns
        per_ann_pred[:, :R]   = ringbuffer[unique_ann]      # (Z, R)
        per_ann_target[:, :R] = ringbuffer_target[unique_ann]      # or another ringbuffer if targets use a different one

        # 6) Group predictions by annotator (stable sort by inv), compute within-group column offsets
        perm = torch.argsort(inv, stable=True)
        inv_sorted   = inv[perm]                            # (T,)
        pred_sorted  = flat_pred[perm]                      # (T,)
        targ_sorted  = flat_target[perm]                    # (T,)

        # unique groups in order and their counts (aligned with inv_sorted)
        _, counts_sorted = torch.unique(inv_sorted, return_counts=True)  # (Z,)

        # start index of each group in the sorted array
        starts = torch.cat([
            torch.zeros(1, device=device, dtype=torch.long),
            torch.cumsum(counts_sorted, dim=0)[:-1]
        ])                                                  # (Z,)

        # broadcast starts to each element position
        starts_per_pos = torch.repeat_interleave(starts, counts_sorted)  # (T,)

        # within-group positions 0..count-1 for each element
        within = torch.arange(pred_sorted.numel(), device=device) - starts_per_pos  # (T,)

        # final row/col indices to scatter into the right block
        rows = inv_sorted                                   # (T,) in [0..Z-1]
        cols = R + within                                   # (T,) in [R..R+max_count-1]

        # 7) Scatter predictions/targets into the right block
        per_ann_pred[rows, cols]   = pred_sorted
        per_ann_target[rows, cols] = targ_sorted
        return per_ann_pred, per_ann_target

    def __call__(self, model_output, targets, masks, helper_outputs=None):
        ind_evals = targets['y_padded_individual_annotators_act']
        ind_val_evals = targets['y_padded_individual_annotators_val']
        soft_act_preds, soft_val_preds = model_output['soft_act_preds'], model_output['soft_val_preds']
        batch_mask, weight_mask, output_mask, seen_annotator_mask = targets['annotator_masks']

        act_preds, act_targets = self.convert_to_individual_annotator_level(soft_act_preds, ind_evals, weight_mask, output_mask, targets['ringbuffers']['act'], targets['ringbuffers']['act_ground_truth'])
        val_preds, val_targets = self.convert_to_individual_annotator_level(soft_val_preds, ind_val_evals, weight_mask, output_mask, targets['ringbuffers']['val'], targets['ringbuffers']['val_ground_truth'])

        # Store latest predictions in the ringbuffers
        with torch.no_grad():
            print('Num nan values in act ringbuffer:', targets['ringbuffers']['act'].isnan().sum())
            targets['ringbuffers']['act'][weight_mask] = targets['ringbuffers']['act'][weight_mask].scatter(1, targets['ringbuffers']['indices'][weight_mask].unsqueeze(dim=-1), soft_act_preds[output_mask].unsqueeze(dim=-1).detach())
            targets['ringbuffers']['act_ground_truth'][weight_mask] = targets['ringbuffers']['act_ground_truth'][weight_mask].scatter(1, targets['ringbuffers']['indices'][weight_mask].unsqueeze(dim=-1), ind_evals[output_mask].unsqueeze(dim=-1))
            targets['ringbuffers']['val'][weight_mask] = targets['ringbuffers']['val'][weight_mask].scatter(1, targets['ringbuffers']['indices'][weight_mask].unsqueeze(dim=-1), soft_val_preds[output_mask].unsqueeze(dim=-1).detach())
            targets['ringbuffers']['val_ground_truth'][weight_mask] = targets['ringbuffers']['val_ground_truth'][weight_mask].scatter(1, targets['ringbuffers']['indices'][weight_mask].unsqueeze(dim=-1), ind_val_evals[output_mask].unsqueeze(dim=-1))
            targets['ringbuffers']['indices'][weight_mask] = (targets['ringbuffers']['indices'][weight_mask] + 1) % targets['ringbuffers']['act'].size(1)

        act_loss = 1 - ccc_nan_safe(act_preds, act_targets, multiple_annotators=True).nanmean()
        val_loss = 1 - ccc_nan_safe(val_preds, val_targets, multiple_annotators=True).nanmean()

        return {f'Individual Annotator Activation {self.loss_string} Loss': act_loss, f'Individual Annotator Valence {self.loss_string} Loss': val_loss}

def ccc_nan_safe(prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        multiple_annotators: bool = False,
        eps: float = 1e-12) -> torch.Tensor:
    """
    Returns CCC per column (annotator). Columns with <2 valid pairs are NaN.
    """
    # Ensure 2D: shape (T, K) where K = num annotators (or 1)
    if prediction.ndim == 1:
        prediction = prediction.view(-1, 1)
    if ground_truth.ndim == 1:
        ground_truth = ground_truth.view(-1, 1)

    if not multiple_annotators:
        assert prediction.shape[-1] == 1, f'Expected last dim=1, got {prediction.shape}'
    assert prediction.shape == ground_truth.shape and prediction.ndim == 2, \
        f'Shapes must match and be 2D, got {prediction.shape=} {ground_truth.shape=}'

    x = prediction.to(torch.float64)
    y = ground_truth.to(torch.float64)

    # Pairwise finite mask (intersection): only positions where BOTH are finite
    m = torch.isfinite(x) & torch.isfinite(y)             # (T, K)
    n = m.sum(dim=0)                                      # (K,)
    valid = n >= 2

    # Per-column means over valid pairs
    n_safe = n.clamp_min(1).to(torch.float64)             # avoid 0/0 in empty columns
    x_sum = torch.where(m, x, 0.0).sum(dim=0)             # (K,)
    y_sum = torch.where(m, y, 0.0).sum(dim=0)
    mx = x_sum / n_safe
    my = y_sum / n_safe

    # Centered with masking
    xz = torch.where(m, x - mx, 0.0)                      # (T, K)
    yz = torch.where(m, y - my, 0.0)

    # Sum of squares and cross term
    ssx = (xz ** 2).sum(dim=0)                            # (K,)
    ssy = (yz ** 2).sum(dim=0)                            # (K,)
    sxy = (xz * yz).sum(dim=0)                            # (K,)

    # Pearson correlation (masked): sxy / sqrt(ssx*ssy)
    denom_cor = (ssx * ssy).clamp_min(eps).sqrt()
    cor = sxy / denom_cor                                 # (K,)

    # Unbiased variances (divide by n-1), safe for n<2
    denom_unbiased = (n - 1).clamp_min(1).to(torch.float64)
    var_x = ssx / denom_unbiased
    var_y = ssy / denom_unbiased
    sd_x = var_x.clamp_min(eps).sqrt()
    sd_y = var_y.clamp_min(eps).sqrt()

    # CCC = (2 * rho * sd_x * sd_y) / (var_x + var_y + (mx - my)^2)
    mean_diff2 = (mx - my) ** 2
    denom_ccc = (var_x + var_y + mean_diff2).clamp_min(eps)
    ccc_col = (2.0 * cor * sd_x * sd_y) / denom_ccc       # (K,)

    # Invalidate columns with <2 pairs
    ccc_col = torch.where(valid, ccc_col, torch.tensor(float('nan'), device=ccc_col.device, dtype=ccc_col.dtype))

    return ccc_col.to(prediction.dtype)

class Task1LossIAEarlyStopping(LossFunction):
    def __init__(self, *args, **kwargs):
        if kwargs['loss_fn'] == 'CCC':
            self.loss_fn = ccc_loss
        elif kwargs['loss_fn'] == 'MSE':
            self.loss_fn = torch.nn.functional.mse_loss
        else:
            raise ValueError(f'Unknown loss type {kwargs["loss_fn"]}')
        self.loss_string = kwargs['loss_fn']
        kwargs['required_model_outputs'] = ['soft_act_preds', 'soft_val_preds']
        kwargs['required_target_labels'] = ['y_padded_individual_annotators_act', 'y_padded_individual_annotators_val']
        self.training_loss_fn = Task1Loss(*args, **kwargs)
        if 'loss_fn' in kwargs:
            del kwargs['loss_fn']
        self.use_individual_validation_loss = False # This will be changed to true when early stopping is triggered
        # Once triggered validation loss will instead be processed per annotator the the base model will be frozen allowing further finetuning of the prediction heads
        super().__init__(*args, **kwargs)

    def __call__(self, model_output, targets, masks, helper_outputs=None):
        if model_output['soft_act_preds'].requires_grad or not self.use_individual_validation_loss:
            return self.training_loss_fn(model_output, targets, masks, helper_outputs)
        validation_losses = {}
        for annotator in targets['ccc_ind_annotators']:
            y_act, y_val = targets[f'y_act_{annotator}'], targets[f'y_val_{annotator}']
            act_pred, val_pred = model_output[f'pred_act_{annotator}'], model_output[f'pred_cal_{annotator}']
            # print(annotator, y_act.shape, act_pred.shape)
            act_loss = self.loss_fn(act_pred, y_act)
            val_loss = self.loss_fn(val_pred, y_val)
            validation_losses[annotator] = {'act': act_loss, 'val': val_loss}
        # crash
        return validation_losses

class Task2Loss(LossFunction):
    def __init__(self, *args, **kwargs):
        kwargs['required_model_outputs'] = ['mean_act_preds', 'mean_val_preds']
        kwargs['required_target_labels'] = ['y_act', 'y_val']
        super().__init__(*args, **kwargs)

    def __call__(self, model_output, targets, masks):
        act = targets['y_act']
        val = targets['y_val']

        mean_act_preds, mean_val_preds = model_output['mean_act_preds'], model_output['mean_val_preds']

        # Values will already be of shape batch_size x model_output_size
        if len(mean_act_preds.shape) > 1 and mean_act_preds.shape[-1] > 1:
            # If outputting multiple observations per annotator then mean the observations
            mean_act_preds = mean_act_preds.nanmean(dim=-1)
            mean_val_preds = mean_val_preds.nanmean(dim=-1)

        mean_act_preds = mean_act_preds.squeeze()
        mean_val_preds = mean_val_preds.squeeze()

        act_loss = ccc_loss(mean_act_preds, act)
        val_loss = ccc_loss(mean_val_preds, val)

        return {'Mean Activation CCC Loss': act_loss, 'Mean Valence CCC Loss': val_loss}

class Task3Loss(LossFunction):
    def __init__(self, *args, **kwargs):
        kwargs['required_model_outputs'] = ['probability_logits']
        kwargs['required_target_labels'] = ['y_kde_2d_probability']
        super().__init__(*args, **kwargs)

    def __call__(self, model_output, targets, masks):
        target_probs = targets['kde_2d_probability']

        kde = model_output['probability_logits']

        ce_loss = probability_loss_no_softmax(kde, target_probs)

        return {'Cross-entropy Loss': ce_loss}

def lognansumexp(v, dim):
    is_nan = torch.isnan(v)
    
    # Check if all values are NaN
    if (~is_nan).sum() == 0:
        result_shape = list(v.shape)
        result_shape.pop(dim)
        return torch.full(result_shape, float('nan'), device=v.device)
    
    x_max = v[~is_nan].max()
    output = torch.full(v.shape, float('nan'), device=v.device)
    output[~is_nan] = (v[~is_nan]-x_max).exp()
    output = output.nansum(dim=dim)
    
    return x_max + torch.log(output)

class KLDivRegularisationLoss(LossFunction):
    def __init__(self, *args, **kwargs):
        if not kwargs['model_determinism_type']:
            raise ValueError('Task 2 loss must use some form of determinism')
        self.model_determinism_type = kwargs['model_determinism_type']
        kwargs['required_model_outputs'] = ['soft_act_preds', 'soft_val_preds', 'soft_act_zs', 'soft_val_zs', 'act_log_vars', 'val_log_vars']
        kwargs['required_target_labels'] = ['y_act', 'y_val']
        if self.model_determinism_type == 'utterance_variance':
            # Define targets as the utterance averages
            kwargs['required_target_labels'].extend(['y_act_variance', 'y_val_variance'])
        else:
            raise ValueError('Other methods underperformed')
        del kwargs['model_determinism_type']
        super().__init__(*args, **kwargs)

    def __call__(self, model_output, targets, masks):
        soft_act_preds, soft_val_preds, (_, act_log_var), (_, val_log_var) = model_output['soft_act_preds'], model_output['soft_val_preds'], model_output['act_z_log_var'], model_output['val_z_log_var']
        if type(soft_act_preds) == list:
            device = targets['act'].device
            soft_act_preds = torch.nn.utils.rnn.pad_sequence([y for x in soft_act_preds for y in torch.nn.utils.rnn.unpad_sequence(x, batch_first=True, lengths=(~x.isnan()).sum(dim=-1))], batch_first=True, padding_value=torch.nan).squeeze().to(device)
            soft_val_preds = torch.nn.utils.rnn.pad_sequence([y for x in soft_val_preds for y in torch.nn.utils.rnn.unpad_sequence(x, batch_first=True, lengths=(~x.isnan()).sum(dim=-1))], batch_first=True, padding_value=torch.nan).squeeze().to(device)

        # During full-dataset evaluation the prediction matrices are rebuilt by
        # process_masks_and_full_padded_values on the GPU, while the accumulated
        # log-variance tensors and targets are deliberately stored on the CPU.
        # Keep the whole KL-divergence calculation on the log-variance device.
        # In training these tensors already share the GPU, so these calls are
        # no-ops.
        loss_device = act_log_var.device
        soft_act_preds = soft_act_preds.to(loss_device)
        soft_val_preds = soft_val_preds.to(loss_device)

        if self.model_determinism_type == 'utterance_variance':
            # Define targets as the utterance averages
            act = targets['y_act'].to(loss_device)
            val = targets['y_val'].to(loss_device)
            act_var = targets['y_act_variance'].to(loss_device)
            val_var = targets['y_val_variance'].to(loss_device)

            # Now average over the annotator dimensions to get average observations
            mean_act_pred = soft_act_preds.nanmean(dim=-1) # \hat\mu in latex
            mean_val_pred = soft_val_preds.nanmean(dim=-1)

            # Now turn N log variances -> variance of combination of random variables (sum[variance]/n^2)
            with torch.no_grad(): # No grad as we just want to figure out how many evaluators there were 
                num_evaluators = (~soft_act_preds.isnan()).sum(dim=-1)

            # Now calculate \hat\sigma^2 (from the latex paper)
            def nanvar(tensor, dim=None, keepdim=False):
                # mean((tensor - tensor_mean)^2), the gradient should work out to 0
                # so we should just detach this value
                tensor_mean = tensor.nanmean(dim=dim, keepdim=True).detach()
                output = torch.fill(torch.empty(tensor.shape, device=tensor.device), float('nan'))
                output[~tensor.isnan()] = (tensor-tensor_mean)[~tensor.isnan()].square()
                output = output.nanmean(dim=dim, keepdim=keepdim)
                output[output<1e-6] = 1e-6 # Add small epsilon when variance in target label is 0 to prevent division by 0, this value is going to be put through log() 
                return output

            act_log_var = lognansumexp(act_log_var, dim=1) - torch.log(num_evaluators)
            val_log_var = lognansumexp(val_log_var, dim=1) - torch.log(num_evaluators)
            act_log_var = torch.logaddexp(act_log_var, nanvar(soft_act_preds, dim=-1).log())
            val_log_var = torch.logaddexp(val_log_var, nanvar(soft_val_preds, dim=-1).log())
            act_kld = kldiv(act, act_var, mean_act_pred, act_log_var)
            val_kld = kldiv(val, val_var, mean_val_pred, val_log_var)

            if act_kld.requires_grad: # Prevent explosion during training, but log real values when in validation/test loop
                print(f'Pre-clamp kld: {act_kld.item()=} {val_kld.item()=}')
                act_kld = torch.clamp(act_kld, min=0, max=5)
                val_kld = torch.clamp(val_kld, min=0, max=5)

            # Return results here as other methods share the same return/loss 
            return {'Activation KL-Div': act_kld, 'Valence KL-Div': val_kld}#, 'Activation CCC': ccc_loss(act_preds, act), 'Valence CCC': ccc_loss(val_preds, val)}
        elif self.model_determinism_type == 'annotator_grid':
            act = targets['padded_individual_evaluators_act']
            val = targets['padded_individual_evaluators_val']
            act_var = targets['act_variance_grid']
            val_var = targets['val_variance_grid']
        elif self.model_determinism_type == 'annotator_knn':
            act = targets['padded_individual_evaluators_act']
            val = targets['padded_individual_evaluators_val']
            act_var = targets['act_variance_knn']
            val_var = targets['val_variance_knn']
        else:
            raise ValueError(f'Unknown model determinism type: {self.model_determinism_type}')

        act_kld = kldiv(act[~act.isnan()], act_var[~act_var.isnan()], soft_act_preds[~soft_act_preds.isnan()], act_log_var[~act_log_var.isnan()])
        val_kld = kldiv(val[~val.isnan()], val_var[~val_var.isnan()], soft_val_preds[~soft_val_preds.isnan()], val_log_var[~val_log_var.isnan()])

        if act_kld.requires_grad: # Prevent explosion during training, but log real values when in validation/test loop
            act_kld = torch.clamp(act_kld, min=-5, max=5)
            val_kld = torch.clamp(val_kld, min=-5, max=5)

        return {'Activation KL-Div': -act_kld, 'Valence KL-Div': -val_kld}#, 'Activation CCC': ccc_loss(act_preds, act), 'Valence CCC': ccc_loss(val_preds, val)}

class LogVarSimpleLoss(LossFunction):
    def __init__(self, *args, **kwargs):
        if not kwargs['model_determinism_type']:
            raise ValueError('Log-Var loss requires determinism type')
        self.model_determinism_type = kwargs['model_determinism_type']
        kwargs['required_model_outputs'] = ['soft_act_preds', 'soft_val_preds', 'soft_act_zs', 'soft_val_zs', 'act_log_vars', 'val_log_vars']
        kwargs['required_target_labels'] = ['y_act', 'y_val']
        if self.model_determinism_type == 'utterance_variance':
            # Define targets as the utterance averages
            kwargs['required_target_labels'].extend(['y_act_variance', 'y_val_variance'])
        else:
            raise ValueError('Other methods underperformed')
        del kwargs['model_determinism_type']
        super().__init__(*args, **kwargs)

    def __call__(self, model_output, targets, masks):
        (soft_act_z, act_log_var), (_, val_log_var) = model_output['act_z_log_var'], model_output['val_z_log_var']

        if self.model_determinism_type == 'utterance_variance':
            # Define targets as the utterance averages
            act_var = targets['y_act_variance']
            val_var = targets['y_val_variance']

            # Now turn N log variances -> variance of combination of random variables (sum[variance]/n^2)
            with torch.no_grad(): # No grad as we just want to figure out how many evaluators there were 
                num_evaluators = (~soft_act_z.sum(dim=-1).isnan()).sum(dim=-1).to(act_log_var.device) # soft_act_z will be on cpu initially due to huge data size for podcast at eval, so make sure this calculation ends on the correct device
            # act_log_var_ = (act_log_var.exp().nansum(dim=1) / torch.pow(num_evaluators, 2)).log()
            # val_log_var_ = (val_log_var.exp().nansum(dim=1) / torch.pow(num_evaluators, 2)).log()
            act_log_var = lognansumexp(act_log_var, dim=1)
            act_log_var = act_log_var - 2*torch.log(num_evaluators)
            val_log_var = lognansumexp(val_log_var, dim=1)
            val_log_var = val_log_var - 2*torch.log(num_evaluators)
            # assert torch.allclose(act_log_var, act_log_var_)
            # assert torch.allclose(val_log_var, val_log_var_)
            act_var[act_var<1e-6] = 1e-6
            val_var[val_var<1e-6] = 1e-6
            act_log_var_ccc = ccc_loss(act_log_var, act_var.log())
            val_log_var_ccc = ccc_loss(val_log_var, val_var.log())

            # Return results here as other methods share the same return/loss 
            return {'Activation log-var CCC': act_log_var_ccc, 'Valence log-var CCC': val_log_var_ccc}#, 'Activation CCC': ccc_loss(act_preds, act), 'Valence CCC': ccc_loss(val_preds, val)}
        raise NotImplementedError('')

class FullDEERLoss(LossFunction):
    def __init__(self, *args, **kwargs):
        kwargs['required_model_outputs'] = ['gamma', 'v', 'alpha', 'beta']
        kwargs['required_target_labels'] = ['y_act', 'y_val', 'y_padded_individual_annotators_act', 'y_padded_individual_annotators_val']
        super().__init__(*args, **kwargs)

    def __call__(self, model_output, targets, masks):
        label_refa = targets['y_act']
        label_refv = targets['y_val']
        label_ref = torch.stack((label_refa, label_refv), dim=-1)
        labela = targets['y_padded_individual_annotators_act']
        labelv = targets['y_padded_individual_annotators_val']
        label = torch.stack((labela,labelv), dim=-1)
        gamma, v, alpha, beta = model_output['gamma'], model_output['v'], model_output['alpha'], model_output['beta'] # Or similar tbd 
        l = DEER_loss(label, label_ref, (gamma, v, alpha, beta))
        if l.isnan().any():
            raise ValueError(f'Loss returned {l.isnan().sum()} NaN values for loss with shape {l.shape}')
        return {'DEER Loss': l.sum()} # Check for default arguments in paper

class CCCIndLoss(LossFunction):
    def __init__(self, *args, **kwargs):
        if kwargs['loss_fn'] == 'CCC':
            self.loss_fn = ccc_loss
        elif kwargs['loss_fn'] == 'MSE':
            self.loss_fn = torch.nn.functional.mse_loss
        else:
            raise ValueError(f'Unknown loss type {kwargs["loss_fn"]}')
        self.loss_string = kwargs['loss_fn']
        del kwargs['loss_fn']
        kwargs['required_model_outputs'] = ['soft_act_preds', 'soft_val_preds']
        kwargs['required_target_labels'] = ['y_padded_individual_annotators_act', 'y_padded_individual_annotators_val', 'annotator_masks']
        super().__init__(*args, **kwargs)

    def convert_to_individual_annotator_level(self, predictions, ground_truth, weight_mask, output_mask, ringbuffer, ringbuffer_target):
        device = predictions.device
        dtype  = predictions.dtype
        R = ringbuffer.size(1)

        # 1) Flatten valid batch predictions/targets and their annotator ids
        flat_pred   = predictions[output_mask]              # (T,)
        flat_target = ground_truth[output_mask]                  # (T,)
        annot_ids   = weight_mask.to(torch.long)            # (T,)

        # 2) Get the unique annotators present in this batch, compact to 0..Z-1
        #    inv maps each element in annot_ids to its compact group index [0..Z-1]
        unique_ann, inv = torch.unique(weight_mask, return_inverse=True, sorted=True)  # unique_ann: (Z,)
        Z = unique_ann.numel()

        # 3) Count how many predictions each present annotator has in this batch
        counts = torch.bincount(inv, minlength=Z)           # (Z,)
        max_count = counts.max().item()
        X = R + max_count                                   # total padded width per annotator

        # 4) Allocate outputs (padded with NaN)
        per_ann_pred   = torch.full((Z, X), float('nan'), device=device, dtype=dtype)
        per_ann_target = torch.full((Z, X), float('nan'), device=device, dtype=dtype)

        # 5) Place ringbuffers into the left R columns
        per_ann_pred[:, :R]   = ringbuffer[unique_ann]      # (Z, R)
        per_ann_target[:, :R] = ringbuffer_target[unique_ann]      # or another ringbuffer if targets use a different one

        # 6) Group predictions by annotator (stable sort by inv), compute within-group column offsets
        perm = torch.argsort(inv, stable=True)
        inv_sorted   = inv[perm]                            # (T,)
        pred_sorted  = flat_pred[perm]                      # (T,)
        targ_sorted  = flat_target[perm]                    # (T,)

        # unique groups in order and their counts (aligned with inv_sorted)
        _, counts_sorted = torch.unique(inv_sorted, return_counts=True)  # (Z,)

        # start index of each group in the sorted array
        starts = torch.cat([
            torch.zeros(1, device=device, dtype=torch.long),
            torch.cumsum(counts_sorted, dim=0)[:-1]
        ])                                                  # (Z,)

        # broadcast starts to each element position
        starts_per_pos = torch.repeat_interleave(starts, counts_sorted)  # (T,)

        # within-group positions 0..count-1 for each element
        within = torch.arange(pred_sorted.numel(), device=device) - starts_per_pos  # (T,)

        # final row/col indices to scatter into the right block
        rows = inv_sorted                                   # (T,) in [0..Z-1]
        cols = R + within                                   # (T,) in [R..R+max_count-1]

        # 7) Scatter predictions/targets into the right block
        per_ann_pred[rows, cols]   = pred_sorted
        per_ann_target[rows, cols] = targ_sorted
        return per_ann_pred, per_ann_target

    def __call__(self, model_output, targets, masks, helper_outputs=None):
        ind_evals = targets['y_padded_individual_annotators_act']
        ind_val_evals = targets['y_padded_individual_annotators_val']
        soft_act_preds, soft_val_preds = model_output['soft_act_preds'], model_output['soft_val_preds']
        batch_mask, weight_mask, output_mask, seen_annotator_mask = targets['annotator_masks']

        act_preds, act_targets = self.convert_to_individual_annotator_level(soft_act_preds, ind_evals, weight_mask, output_mask, targets['ringbuffers']['act'], targets['ringbuffers']['act_ground_truth'])
        val_preds, val_targets = self.convert_to_individual_annotator_level(soft_val_preds, ind_val_evals, weight_mask, output_mask, targets['ringbuffers']['val'], targets['ringbuffers']['val_ground_truth'])

        # Store latest predictions in the ringbuffers
        with torch.no_grad():
            print('Num nan values in act ringbuffer:', targets['ringbuffers']['act'].isnan().sum())
            targets['ringbuffers']['act'][weight_mask] = targets['ringbuffers']['act'][weight_mask].scatter(1, targets['ringbuffers']['indices'][weight_mask].unsqueeze(dim=-1), soft_act_preds[output_mask].unsqueeze(dim=-1).detach())
            targets['ringbuffers']['act_ground_truth'][weight_mask] = targets['ringbuffers']['act_ground_truth'][weight_mask].scatter(1, targets['ringbuffers']['indices'][weight_mask].unsqueeze(dim=-1), ind_evals[output_mask].unsqueeze(dim=-1))
            targets['ringbuffers']['val'][weight_mask] = targets['ringbuffers']['val'][weight_mask].scatter(1, targets['ringbuffers']['indices'][weight_mask].unsqueeze(dim=-1), soft_val_preds[output_mask].unsqueeze(dim=-1).detach())
            targets['ringbuffers']['val_ground_truth'][weight_mask] = targets['ringbuffers']['val_ground_truth'][weight_mask].scatter(1, targets['ringbuffers']['indices'][weight_mask].unsqueeze(dim=-1), ind_val_evals[output_mask].unsqueeze(dim=-1))
            targets['ringbuffers']['indices'][weight_mask] = (targets['ringbuffers']['indices'][weight_mask] + 1) % targets['ringbuffers']['act'].size(1)

        act_loss = 1 - ccc_nan_safe(act_preds, act_targets, multiple_annotators=True).nanmean()
        val_loss = 1 - ccc_nan_safe(val_preds, val_targets, multiple_annotators=True).nanmean()

        return {f'Individual Annotator Activation {self.loss_string} Loss': act_loss, f'Individual Annotator Valence {self.loss_string} Loss': val_loss}

def ccc_nan_safe(prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        multiple_annotators: bool = False,
        eps: float = 1e-12) -> torch.Tensor:
    """
    Returns CCC per column (annotator). Columns with <2 valid pairs are NaN.
    """
    # Ensure 2D: shape (T, K) where K = num annotators (or 1)
    if prediction.ndim == 1:
        prediction = prediction.view(-1, 1)
    if ground_truth.ndim == 1:
        ground_truth = ground_truth.view(-1, 1)

    if not multiple_annotators:
        assert prediction.shape[-1] == 1, f'Expected last dim=1, got {prediction.shape}'
    assert prediction.shape == ground_truth.shape and prediction.ndim == 2, \
        f'Shapes must match and be 2D, got {prediction.shape=} {ground_truth.shape=}'

    x = prediction.to(torch.float64)
    y = ground_truth.to(torch.float64)

    # Pairwise finite mask (intersection): only positions where BOTH are finite
    m = torch.isfinite(x) & torch.isfinite(y)             # (T, K)
    n = m.sum(dim=0)                                      # (K,)
    valid = n >= 2

    # Per-column means over valid pairs
    n_safe = n.clamp_min(1).to(torch.float64)             # avoid 0/0 in empty columns
    x_sum = torch.where(m, x, 0.0).sum(dim=0)             # (K,)
    y_sum = torch.where(m, y, 0.0).sum(dim=0)
    mx = x_sum / n_safe
    my = y_sum / n_safe

    # Centered with masking
    xz = torch.where(m, x - mx, 0.0)                      # (T, K)
    yz = torch.where(m, y - my, 0.0)

    # Sum of squares and cross term
    ssx = (xz ** 2).sum(dim=0)                            # (K,)
    ssy = (yz ** 2).sum(dim=0)                            # (K,)
    sxy = (xz * yz).sum(dim=0)                            # (K,)

    # Pearson correlation (masked): sxy / sqrt(ssx*ssy)
    denom_cor = (ssx * ssy).clamp_min(eps).sqrt()
    cor = sxy / denom_cor                                 # (K,)

    # Unbiased variances (divide by n-1), safe for n<2
    denom_unbiased = (n - 1).clamp_min(1).to(torch.float64)
    var_x = ssx / denom_unbiased
    var_y = ssy / denom_unbiased
    sd_x = var_x.clamp_min(eps).sqrt()
    sd_y = var_y.clamp_min(eps).sqrt()

    # CCC = (2 * rho * sd_x * sd_y) / (var_x + var_y + (mx - my)^2)
    mean_diff2 = (mx - my) ** 2
    denom_ccc = (var_x + var_y + mean_diff2).clamp_min(eps)
    ccc_col = (2.0 * cor * sd_x * sd_y) / denom_ccc       # (K,)

    # Invalidate columns with <2 pairs
    ccc_col = torch.where(valid, ccc_col, torch.tensor(float('nan'), device=ccc_col.device, dtype=ccc_col.dtype))

    return ccc_col.to(prediction.dtype)

from condor_pytorch.losses import CondorOrdinalCrossEntropy
class CONDORLoss(LossFunction):
    def __init__(self, *args, **kwargs):
        kwargs['required_model_outputs'] = ['act_probability_logits', 'val_probability_logits']
        kwargs['required_target_labels'] = ['y_padded_individual_annotators_act', 'y_padded_individual_annotators_val', 'annotator_masks']
        self.boundaries = torch.as_tensor([-1, -2/3, -1/3, 0, 1/3, 2/3], device='cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__(*args, **kwargs)

    def __call__(self, model_output, targets, masks):
        act_logits = model_output['act_probability_logits'] # Output will be batch x num evaluators x 6
        val_logits = model_output['val_probability_logits'] # Output will be batch x num evaluators x 6
        act_labels = targets['y_padded_individual_annotators_act'] # Output will be batch x num evaluators x 1
        val_labels = targets['y_padded_individual_annotators_val'] # Output will be batch x num evaluators x 1
        # Convert to be a simulated batch x 6 for each
        # print('pre-reshape', 'act_logits', act_logits.shape, 'val_logits', val_logits.shape)
        # print('pre-reshape', 'act_labels', act_labels.shape, 'val_labels', val_labels.shape)
        act_logits = act_logits.view(-1, 6)
        val_logits = val_logits.view(-1, 6)
        act_labels = act_labels.view(-1, 1)
        val_labels = val_labels.view(-1, 1)
        # print('act_logits', act_logits.shape, 'val_logits', val_logits.shape)
        # print('act_labels', act_labels.shape, 'val_labels', val_labels.shape)
        mask = ~act_labels.isnan().squeeze().bool()
        # print('mask', mask.shape, mask.sum())
        act_logits = act_logits[mask]
        val_logits = val_logits[mask]
        act_labels = act_labels[mask]
        val_labels = val_labels[mask]
        # Now we want to build the correct binary labels
        # If level < -2/3 then output is [0,0,0,0,0,0], if level < -1/3 then output is [1,0,0,0,0,0], ...
        # print(act_logits.shape, act_labels.shape)
        # act_levels = torch.zeros(act_logits.shape)
        # val_levels = torch.zeros(val_logits.shape)
        self.boundaries = self.boundaries.to(act_labels.device)
        act_levels = (self.boundaries < act_labels).float()
        val_levels = (self.boundaries < val_labels).float()
        # print('Final shapes', act_levels.shape, val_levels.shape, act_logits.shape, val_logits.shape, act_labels.shape, val_labels.shape)
        act_loss = CondorOrdinalCrossEntropy(act_logits, act_levels, reduction='mean')
        val_loss = CondorOrdinalCrossEntropy(val_logits, val_levels, reduction='mean')
        return {'CONDOR Activation Loss': act_loss, 'CONDOR Valence Loss': val_loss}

class CONDORTask1CombinedLoss(LossFunction):
    def __init__(self, *args, **kwargs):
        self.condor_loss = CONDORLoss(*args, **kwargs)
        kwargs['loss_fn'] = 'CCC'
        self.task1_loss = Task1Loss(*args, **kwargs)
        kwargs['required_model_outputs'] = ['act_probability_logits', 'val_probability_logits', 'soft_act_preds', 'soft_val_preds']
        kwargs['required_target_labels'] = ['y_padded_individual_annotators_act', 'y_padded_individual_annotators_val', 'annotator_masks']
        del kwargs['loss_fn']
        super().__init__(*args, **kwargs)

    def __call__(self, model_output, targets, masks, helper_outputs=None):
        condor_loss = self.condor_loss(model_output, targets, masks)
        task1_loss = self.task1_loss(model_output, targets, masks, helper_outputs)
        return {**condor_loss, **task1_loss}

class CategoricalLoss(LossFunction):
    def __init__(self, *args, **kwargs):
        kwargs['required_model_outputs'] = ['act_probability_logits', 'val_probability_logits']
        kwargs['required_target_labels'] = ['y_padded_individual_annotators_act', 'y_padded_individual_annotators_val', 'annotator_masks']
        self.boundaries = torch.as_tensor([-1, -2/3, -1/3, 0, 1/3, 2/3], device='cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__(*args, **kwargs)

    def __call__(self, model_output, targets, masks):
        act_logits, val_logits = model_output['act_probability_logits'], model_output['act_probability_logits']
        act_labels = targets['y_padded_individual_annotators_act']
        val_labels = targets['y_padded_individual_annotators_val']
        # Need to convert the act labels to categorical labels. 
        # Labels are [-1, -2/3, -1/3, 0, 1/3, 2/3, 1], pick the closest class based on closest absolute distance to boundaries
        self.boundaries = self.boundaries.to(act_labels.device)

        act_logits = act_logits.view(-1, 7)
        val_logits = val_logits.view(-1, 7)
        act_labels = act_labels.view(-1, 1)
        val_labels = val_labels.view(-1, 1)
        # print('act_logits', act_logits.shape, 'val_logits', val_logits.shape)
        # print('act_labels', act_labels.shape, 'val_labels', val_labels.shape)
        mask = ~act_labels.isnan().squeeze().bool()
        # print('mask', mask.shape, mask.sum())
        act_logits = act_logits[mask]
        val_logits = val_logits[mask]
        act_labels = act_labels[mask]
        val_labels = val_labels[mask]

        print('cross-entropy loss shapes', act_logits.shape, act_labels.shape, val_logits.shape, val_labels.shape)
        act_labels = torch.argmin(torch.abs(act_labels - self.boundaries), dim=-1).long()
        val_labels = torch.argmin(torch.abs(val_labels - self.boundaries), dim=-1).long()
        act_loss = torch.nn.functional.cross_entropy(act_logits, act_labels, reduction='mean')
        val_loss = torch.nn.functional.cross_entropy(val_logits, val_labels, reduction='mean')
        return {'Categorical Activation Loss': act_loss, 'Categorical Valence Loss': val_loss}


class OrthogonalLoss(LossFunction):
    def __init__(self, *args, **kwargs):
        kwargs['required_model_outputs'] = ['basis_annotator_embeddings_activation', 'basis_annotator_embeddings_valence'] # Probably will also add a shared part to the embedding later
        kwargs['required_target_labels'] = []
        super().__init__(*args, **kwargs)

    def __call__(self, model_output, targets, masks):
        act_embeddings = model_output['basis_annotator_embeddings_activation']
        val_embeddings = model_output['basis_annotator_embeddings_valence'] # Embeddings are [num_basis_annotators, layer_size], so orthogonal matrix should be [num_basis_annotators, num_basis_annotators]
        G_act = act_embeddings @ act_embeddings.transpose(-1,-2)
        I_act = torch.eye(act_embeddings.shape[0], device=act_embeddings.device)
        G_val = val_embeddings @ val_embeddings.transpose(-1,-2)
        I_val = torch.eye(val_embeddings.shape[0], device=val_embeddings.device)
        act_loss = torch.nn.functional.mse_loss(G_act, I_act)
        val_loss = torch.nn.functional.mse_loss(G_val, I_val)
        # print('Orthogonal loss outputs:')
        # print('Activation embeddings (input):', act_embeddings)
        # print('G_act (output):', G_act)
        # print('I_act (expected):', I_act)
        # print('act_loss:', act_loss)
        return {'Orthogonal Activation Loss': act_loss, 'Orthogonal Valence Loss': val_loss} # This is actually orthonormal 

import torch.nn.functional as F
class TrueOrthogonalLoss(LossFunction):
    def __init__(self, *args, **kwargs):
        kwargs['required_model_outputs'] = ['basis_annotator_embeddings_activation', 'basis_annotator_embeddings_valence'] # Probably will also add a shared part to the embedding later
        kwargs['required_target_labels'] = []
        super().__init__(*args, **kwargs)

    def __call__(self, model_output, targets, masks):
        # Normalize the embeddings first to prevent loss from just minimising all values to 0 
        act_embeddings = F.normalize(model_output['basis_annotator_embeddings_activation'], p=2, dim=-1)
        val_embeddings = F.normalize(model_output['basis_annotator_embeddings_valence'], p=2, dim=-1) # Embeddings are [num_basis_annotators, layer_size], so orthogonal matrix should be [num_basis_annotators, num_basis_annotators]
        G_act = act_embeddings @ act_embeddings.transpose(-1,-2)
        G_val = val_embeddings @ val_embeddings.transpose(-1,-2)
        I = torch.eye(G_act.shape[-1], device=G_act.device, dtype=G_act.dtype)
        act_loss = torch.nn.functional.mse_loss(G_act, I)
        val_loss = torch.nn.functional.mse_loss(G_val, I)
        return {'Orthogonal Activation Loss': act_loss, 'Orthogonal Valence Loss': val_loss}

class OrthogonalPlusCCCLoss(LossFunction):
    def __init__(self, *args, **kwargs):
        self.orthogonal_loss = OrthogonalLoss(*args, **kwargs)
        kwargs['loss_fn'] = 'CCC'
        self.task1_loss = Task1Loss(*args, **kwargs)
        kwargs['required_model_outputs'] = ['act_probability_logits', 'val_probability_logits', 'soft_act_preds', 'soft_val_preds', 'basis_annotator_embeddings_activation', 'basis_annotator_embeddings_valence']
        kwargs['required_target_labels'] = ['y_padded_individual_annotators_act', 'y_padded_individual_annotators_val', 'annotator_masks']
        del kwargs['loss_fn']
        super().__init__(*args, **kwargs)

    def __call__(self, model_output, targets, masks):
        orthogonal_loss = self.orthogonal_loss(model_output, targets, masks)
        orthogonal_loss = {f'(1e-4 scaled) {k}': 1e-4*v for k, v in orthogonal_loss.items()}
        task1_loss = self.task1_loss(model_output, targets, masks)
        return {**orthogonal_loss, **task1_loss}

class TrueOrthogonalPlusCCCLoss(LossFunction):
    def __init__(self, *args, **kwargs):
        self.orthogonal_loss = TrueOrthogonalLoss(*args, **kwargs)
        kwargs['loss_fn'] = 'CCC'
        self.task1_loss = Task1Loss(*args, **kwargs)
        kwargs['required_model_outputs'] = ['act_probability_logits', 'val_probability_logits', 'soft_act_preds', 'soft_val_preds', 'basis_annotator_embeddings_activation', 'basis_annotator_embeddings_valence']
        kwargs['required_target_labels'] = ['y_padded_individual_annotators_act', 'y_padded_individual_annotators_val', 'annotator_masks']
        del kwargs['loss_fn']
        super().__init__(*args, **kwargs)

    def __call__(self, model_output, targets, masks):
        orthogonal_loss = self.orthogonal_loss(model_output, targets, masks)
        orthogonal_loss = {f'(1e-4 scaled) {k}': 1e-4*v for k, v in orthogonal_loss.items()}
        task1_loss = self.task1_loss(model_output, targets, masks)
        return {**orthogonal_loss, **task1_loss}

class TrueOrthogonalPlusCCCLossPlussWavLMInspiredLoss(LossFunction):
    def __init__(self, *args, **kwargs):
        self.orthogonal_loss = TrueOrthogonalPlusCCCLoss(*args, **kwargs)
        kwargs['required_model_outputs'] = ['act_probability_logits', 'val_probability_logits', 'soft_act_preds', 'soft_val_preds', 'basis_annotator_embeddings_activation', 'basis_annotator_embeddings_valence']
        kwargs['required_target_labels'] = ['y_padded_individual_annotators_act', 'y_padded_individual_annotators_val', 'annotator_masks']
        super().__init__(*args, **kwargs)

    def peaky_loss(self, weights, mask, eps=1e-8):
        # Calculate loss to encourage peaky distributions
        # entropy per (sample, annotator)
        entropy = -(weights * (weights + eps).log()).sum(dim=-1)  # [B, A]

        # keep only valid annotators
        entropy = entropy * mask.float()

        # average over *valid* annotators only
        denom = mask.sum() + eps
        return entropy.sum() / denom

    def __call__(self, model_output, targets, masks):
        task1_and_orthogonal_loss = self.orthogonal_loss(model_output, targets, masks)
        if not model_output['basis_annotator_embeddings_activation'].requires_grad:
            return task1_and_orthogonal_loss # Skip peaky loss for validation because batching the basis weights is not easy during validation and not worth the effort 
        act_basis_weights = model_output['act_basis_weights']
        val_basis_weights = model_output['val_basis_weights']
        batch_mask, weight_mask, output_mask, seen_annotator_mask = targets['annotator_masks']
        output_mask = output_mask.to(act_basis_weights.device)
        act_peaky_loss = self.peaky_loss(act_basis_weights, output_mask)
        val_peaky_loss = self.peaky_loss(val_basis_weights, output_mask)

        # return {**task1_and_orthogonal_loss, '1e-3*Activation WavLM Inspired Loss': 1e-3*act_peaky_loss, '1e-3*Valence WavLM Inspired Loss': 1e-3*val_peaky_loss}
        return {**task1_and_orthogonal_loss, '1e-1*Activation WavLM Inspired Loss': 1e-1*act_peaky_loss, '1e-1*Valence WavLM Inspired Loss': 1e-1*val_peaky_loss}

class ConsensusOnlyLoss(LossFunction):
    def __init__(self, *args, **kwargs):
        kwargs['required_model_outputs'] = ['act_probability_logits', 'val_probability_logits', 'soft_act_preds', 'soft_val_preds', 'basis_annotator_embeddings_activation', 'basis_annotator_embeddings_valence']
        kwargs['required_target_labels'] = ['y_padded_individual_annotators_act', 'y_padded_individual_annotators_val', 'annotator_masks']
        super().__init__(*args, **kwargs)

    def __call__(self, model_output, targets, masks):
        # Calculate CCC for the consensus predictions
        if 'consensus_act_predictions' in model_output:
            act_preds = model_output['consensus_act_predictions']
            val_preds = model_output['consensus_val_predictions']
        else:
            act_preds = model_output['mean_act_preds']
            val_preds = model_output['mean_val_preds']
        act_targets = targets['y_act']
        val_targets = targets['y_val']

        act_conc_loss = ccc_loss(act_preds, act_targets)
        val_conc_loss = ccc_loss(val_preds, val_targets)
        return {'CCC Concensus Activation Loss': act_conc_loss, 'CCC Concensus Valence Loss': val_conc_loss}

class CCCIndOnlyLoss(LossFunction):
    def __init__(self, *args, **kwargs):
        self.use_minimum_validation_count = kwargs.pop('use_minimum_validation_count', 1)
        kwargs['required_model_outputs'] = ['act_probability_logits', 'val_probability_logits', 'soft_act_preds', 'soft_val_preds', 'basis_annotator_embeddings_activation', 'basis_annotator_embeddings_valence']
        kwargs['required_target_labels'] = ['y_padded_individual_annotators_act', 'y_padded_individual_annotators_val', 'annotator_masks']
        super().__init__(*args, **kwargs)

    def __call__(self, model_output, targets, masks):
        # Calculate the CCC for the target annotator of this batch 
        batch_mask, weight_mask, output_mask, _ = masks
        flat_act = model_output['soft_act_preds'][output_mask]
        flat_val = model_output['soft_val_preds'][output_mask]
        flat_act_targets = targets['y_padded_individual_annotators_act'][output_mask]
        flat_val_targets = targets['y_padded_individual_annotators_val'][output_mask]
        if flat_act.requires_grad:
            act_target_pred = flat_act[weight_mask == targets['target_annotator_idx']]
            val_target_pred = flat_val[weight_mask == targets['target_annotator_idx']]
            act_target_targets = flat_act_targets[weight_mask == targets['target_annotator_idx']]
            val_target_targets = flat_val_targets[weight_mask == targets['target_annotator_idx']]
            act_ind_loss = ccc_loss(act_target_pred, act_target_targets)
            val_ind_loss = ccc_loss(val_target_pred, val_target_targets)
            return_dict = {f'CCC Target Annotator ({targets["target_annotator"]}) Activation Loss': act_ind_loss, f'CCC Target Annotator ({targets["target_annotator"]}) Valence Loss': val_ind_loss}
            return return_dict
        else:
            # When doing validation or testing, instead just calculate the CCC ind for all annotators and average 
            act_cccs = []
            val_cccs = []
            for annotator in weight_mask.unique():
                act_pred = flat_act[weight_mask == annotator]
                val_pred = flat_val[weight_mask == annotator]
                act_targets = flat_act_targets[weight_mask == annotator]
                val_targets = flat_val_targets[weight_mask == annotator]
                if act_pred.shape[0] > self.use_minimum_validation_count:
                    act_cccs.append(ccc_loss(act_pred, act_targets))
                    val_cccs.append(ccc_loss(val_pred, val_targets))
            act_cccs = torch.stack(act_cccs)
            val_cccs = torch.stack(val_cccs)
            return_dict = {'Mean CCC Target Annotator Activation Loss': act_cccs.mean(), 'Mean CCC Target Annotator Valence Loss': val_cccs.mean()}
            return return_dict

class ConsensusAndCCCIndLoss(LossFunction):
    def __init__(self, *args, **kwargs):
        self.use_ccc_flat_loss = kwargs.pop('use_ccc_flat_loss', False)
        self.use_consensus = kwargs.pop('use_consensus', False)
        self.use_ccc_ind = kwargs.pop('use_ccc_ind', False)
        print('CCC Consensus and CCC Ind loss init - use_ccc_flat_loss:', self.use_ccc_flat_loss)
        if self.use_ccc_flat_loss:
            kwargs['loss_fn'] = 'CCC'
            self.ccc_flat_loss = Task1Loss(*args, **kwargs)
            del kwargs['loss_fn']
        if self.use_ccc_ind:
            self.ccc_ind_loss = CCCIndOnlyLoss(*args, **kwargs)
            if 'use_minimum_validation_count' in kwargs:
                del kwargs['use_minimum_validation_count']
        if self.use_consensus:
            self.consensus_loss = ConsensusOnlyLoss(*args, **kwargs)
        kwargs['required_model_outputs'] = ['act_probability_logits', 'val_probability_logits', 'soft_act_preds', 'soft_val_preds', 'basis_annotator_embeddings_activation', 'basis_annotator_embeddings_valence']
        kwargs['required_target_labels'] = ['y_padded_individual_annotators_act', 'y_padded_individual_annotators_val', 'annotator_masks']
        super().__init__(*args, **kwargs)

    def __call__(self, model_output, targets, masks):
        return_dict = {}
        # Calculate CCC for the consensus predictions
        conc_loss = self.consensus_loss(model_output, targets, masks) if self.use_consensus else {}
        return_dict.update(conc_loss)
        ccc_ind_loss = self.ccc_ind_loss(model_output, targets, masks) if self.use_ccc_ind else {}
        return_dict.update(ccc_ind_loss)
        # CCC Flat Loss
        ccc_flat_loss = self.ccc_flat_loss(model_output, targets, masks) if self.use_ccc_flat_loss else {}
        return_dict.update(ccc_flat_loss)
        return return_dict

class ConsensusAndCCCIndLossPlusOrthogonalLoss(LossFunction):
    def __init__(self, *args, **kwargs):
        self.decov_loss_on_outputs = kwargs.pop('decov_loss_on_outputs', False)
        self.add_load_balancing = kwargs.pop('add_load_balancing', False)
        self.add_importance_balancing = kwargs.pop('add_importance_balancing', False)
        self.consensus_and_ccc_ind_loss = ConsensusAndCCCIndLoss(*args, **kwargs)
        if 'use_ccc_ind' in kwargs:
            del kwargs['use_ccc_ind']
        if 'use_ccc_flat_loss' in kwargs:
            del kwargs['use_ccc_flat_loss']
        if 'use_minimum_validation_count' in kwargs:
            del kwargs['use_minimum_validation_count']
        if 'use_consensus' in kwargs:
            del kwargs['use_consensus']
        self.orthogonal_loss = TrueOrthogonalLoss(*args, **kwargs)
        kwargs['required_model_outputs'] = ['act_probability_logits', 'val_probability_logits', 'soft_act_preds', 'soft_val_preds', 'basis_annotator_embeddings_activation', 'basis_annotator_embeddings_valence']
        kwargs['required_target_labels'] = ['y_padded_individual_annotators_act', 'y_padded_individual_annotators_val', 'annotator_masks']
        super().__init__(*args, **kwargs)

    def decov_loss(self, h, eps=1e-8):
        """
        Compute DeCov loss as defined in https://arxiv.org/pdf/1511.06068
        
        DeCov = 1/2 * (||C||_F^2 - ||diag(C)||_2^2)
        
        where C is the cross-covariance matrix of activations h.
        This penalizes off-diagonal correlations while allowing variance.
        
        Args:
            h: activations of shape [batch_size, num_features]
            eps: small constant for numerical stability
            
        Returns:
            scalar DeCov loss
        """
        batch_size = h.size(0)
        num_features = h.size(1)
        
        # Center the features (zero mean)
        h_centered = h - h.mean(dim=0, keepdim=True)
        
        # Compute covariance matrix: C = (1/n) * H^T H
        # where H is the centered activation matrix
        C = torch.mm(h_centered.t(), h_centered) / (batch_size - 1)
        
        # Frobenius norm of C (squared): sum of all squared elements
        frob_norm_sq = torch.sum(C ** 2)
        
        # 2-norm of diagonal (squared): sum of squared diagonal elements
        diag_norm_sq = torch.sum(torch.diag(C) ** 2)
        
        # DeCov loss: penalize off-diagonal elements
        decov = 0.5 * (frob_norm_sq - diag_norm_sq)
        
        return decov
    
    def __call__(self, model_output, targets, masks):
        consensus_and_ccc_ind_loss = self.consensus_and_ccc_ind_loss(model_output, targets, masks)
        orthogonal_loss = self.orthogonal_loss(model_output, targets, masks)
        orthogonal_loss = {f'0.1 * {k}': 0.1*v for k, v in orthogonal_loss.items()}
        
        return_dict = {**consensus_and_ccc_ind_loss, **orthogonal_loss}
        if self.decov_loss_on_outputs:
            # Apply DeCov loss on basis predictions to encourage diverse representations
            # basis_act_predictions: [batch_size, num_basis_annotators]
            # basis_val_predictions: [batch_size, num_basis_annotators]
            act_basis = model_output['basis_act_predictions']
            val_basis = model_output['basis_val_predictions']

            # Only compute if we have enough samples
            if act_basis.size(0) > 1:
                act_decov = self.decov_loss(act_basis)
                val_decov = self.decov_loss(val_basis)

                # Scale the DeCov loss (typically needs small weight)
                decov_dict = {
                    '0.1 * DeCov Activation Loss': 0.1 * act_decov,
                    '0.1 * DeCov Valence Loss': 0.1 * val_decov
                }
                
                return_dict.update(decov_dict)

        if self.add_importance_balancing:
            importance_act = model_output['importance_act']
            importance_val = model_output['importance_val']
            eps = 1e-8
            cv_squared_act_importance = importance_act.var(unbiased=False) / (importance_act.mean()**2 + eps)
            cv_squared_val_importance = importance_val.var(unbiased=False) / (importance_val.mean()**2 + eps)
            importance_dict = {
                '0.1 * CV Squared Activation Importance Loss': 0.1*cv_squared_act_importance,
                '0.1 * CV Squared Valence Importance Loss': 0.1*cv_squared_val_importance
            }
            return_dict.update(importance_dict)

        if self.add_load_balancing:
            fraction_selection_act = model_output['fraction_selection_act']
            fraction_selection_val = model_output['fraction_selection_val']
            p_selection_act = model_output['p_selection_act']
            p_selection_val = model_output['p_selection_val']
            K = fraction_selection_act.shape[-1]
            load_act_loss = K * torch.sum(fraction_selection_act * fraction_selection_act)
            load_val_loss = K * torch.sum(fraction_selection_val * fraction_selection_val)
            load_balancing_dict = {
                '0.1 * Load-Balancing Activation Loss': 0.1*load_act_loss,
                '0.1 * Load-Balancing Valence Loss': 0.1*load_val_loss
            }
            return_dict.update(load_balancing_dict)

        return return_dict
