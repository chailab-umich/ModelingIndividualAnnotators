from scipy.stats import spearmanr
from tqdm import tqdm
from MIA.metrics import ccc, pearson
from MIA.config import ModelType
import torch
import numpy as np
from collections import defaultdict
from typing import Dict, Optional, Union

# Receive
# dataloader of training samples
# dictionary of name: model_trainer class 
# output size for probability grid (i.e. 4: 4x4 probability grid size)
# Will then computer 1 training epoch for each model 
def training_epoch(train_dataloader, model_trainer, epoch, weight_restore_hook=None, consensus_dataloader=None):
    train_pbar = tqdm(train_dataloader)
    model_trainer.train()
    consensus_dataloader_iter = iter(consensus_dataloader) if consensus_dataloader is not None else None

    for batch in train_pbar:
        audio, text, target_probs, y_act_var, y_val_var = batch['audio'], batch['transcript'], batch['kde_2d_probability'], batch['act_variance'], batch['val_variance']

        targets = {'y_act': batch['act'], 'y_val': batch['val'], 'y_kde_2d_probability': target_probs, 'y_act_variance': y_act_var, 'y_val_variance': y_val_var}
        if 'padded_individual_annotators_act' in batch:
            targets = {**targets, 'y_padded_individual_annotators_act': batch['padded_individual_annotators_act'], 'y_padded_individual_annotators_val': batch['padded_individual_annotators_val']}
            if 'annotator_masks' in batch:
                targets = {**targets, 'annotator_masks': batch['annotator_masks']}
        if 'target_annotator' in batch:
            targets['target_annotator'] = batch['target_annotator']
            targets['target_annotator_idx'] = model_trainer.model.prediction_head.act_heads.annotator_mapper.map_to_idx[batch['target_annotator']]

        current_batch_type = 'ccc_ind' if consensus_dataloader is not None else None
        # if current_batch_type == 'ccc_ind': # TODO: Revisit this. The issue is that we can't train consensus without predicting all annotators for most models, so this is too strong. Without it, however, orthogonal losses will be applied against all annotators
        #     # We only need to predict the target annotator for the CCC ind batch, so we can change the input masks 
        #     batch_mask = torch.arange(audio.shape[0], device=audio.device)
        #     weight_mask = torch.stack([torch.tensor(targets['target_annotator_idx'], device=audio.device) for _ in range(audio.shape[0])], dim=0)
        #     output_mask = torch.ones((audio.shape[0], 1), device=audio.device, dtype=torch.bool) # Predict the 1 target annotator for each sample
        #     masks = (batch_mask, weight_mask, output_mask, output_mask)
        #     targets = {**targets, 'annotator_masks': masks}
        output_str = model_trainer.training_step(audio, text, targets, epoch, current_batch_type=current_batch_type)
        val_loss_str = model_trainer.get_validation_loss_display()

        train_pbar.set_description(f'Training loss {output_str}{val_loss_str}')
        if weight_restore_hook is not None:
            removed_annotators, annotator_best_weights, restore_annotator_weights = weight_restore_hook
            for ann in removed_annotators:
                restore_annotator_weights(model_trainer.model, ann, annotator_best_weights)
        if consensus_dataloader is not None:
            consensus_batch = next(consensus_dataloader_iter)
            # if consensus_batch is None: # This should never happen as there are more batches in this dataloader, so comment out and force a crash if it happens 
            #     consensus_dataloader_iter = iter(consensus_dataloader)
            #     consensus_batch = next(consensus_dataloader_iter)
            consensus_audio, consensus_text = consensus_batch['audio'], consensus_batch['transcript']
            consensus_targets = {'y_act': consensus_batch['act'], 'y_val': consensus_batch['val']}
            if 'padded_individual_annotators_act' in consensus_batch:
                consensus_targets = {**consensus_targets, 'y_padded_individual_annotators_act': consensus_batch['padded_individual_annotators_act'], 'y_padded_individual_annotators_val': consensus_batch['padded_individual_annotators_val']}
            if 'annotator_masks' in consensus_batch:
                consensus_targets = {**consensus_targets, 'annotator_masks': consensus_batch['annotator_masks']}
            current_batch_type = 'consensus'
            output_str = model_trainer.training_step(consensus_audio, consensus_text, consensus_targets, epoch, current_batch_type=current_batch_type)
            val_loss_str = model_trainer.get_validation_loss_display()
            train_pbar.set_description(f'Training loss {output_str}{val_loss_str}')

    model_trainer.log_average_loss(epoch)

def build_mask(fixed_annotators, batch_size, device):
    # fixed_annotators = [seed_results[f'seed_{seed}']['annotator_to_podcast_annotator'][ann] for ann in batch['annotators'][0] if ann in seed_results[f'seed_{seed}']['annotator_to_podcast_annotator']]
    # print(fixed_annotators)
    batch_mask = torch.as_tensor([i for i in range(batch_size) for _ in range(len(fixed_annotators))], device=device)
    # Same annotators in each batch
    annotator_masks = torch.as_tensor([fixed_annotators for _ in range(batch_size)], device=device).long()
    output_mask = ~annotator_masks.isnan()
    annotator_masks = annotator_masks[~annotator_masks.isnan()]
    # Output mask is entirely true with shape of annotator masks 
    # All annotators are seen in this case so seen annotator mask can just be output mask
    masks = (batch_mask, annotator_masks, output_mask, output_mask)
    return masks

def find_best_annotators(eval_dataloader, model_trainer, N, num_samples_to_use_for_map, logger, map_act_val_separately=False, randomly_map_annotators=False, dataset_name=None, map_settings=False):
    model_trainer.model.eval()
    annotator_level_results = defaultdict(lambda: {'y_act': [], 'y_val': [], 'podcast_acts': [], 'podcast_vals': [], 'FileName': []})
    batch_size = 1
    if model_trainer.model.args.model_type == ModelType.AGGREGATE_GROUND_TRUTH:
        all_annotators = ['Aggregate Ground Truth']
        all_annotators_masks = None
    else:
        all_annotators = list(model_trainer.model.prediction_head.act_heads.annotator_mapper.map_to_annotator.keys())
        all_annotators_masks = build_mask(all_annotators, batch_size, model_trainer.model.device)
    with torch.no_grad():
        if map_settings is not None and 'use_CORAL' in map_settings and map_settings['use_CORAL']:
            from MIA.models.CORAL_wrapper import CORAL_Process
            coral_set = map_settings['CORAL_enrollment_set']
            coral_process_act = CORAL_Process(num_ordinal_bins=map_settings['num_ordinal_bins'])
            coral_process_val = CORAL_Process(num_ordinal_bins=map_settings['num_ordinal_bins'])
            z_act_items = []
            z_val_items = []
            for batch in tqdm(coral_set):
                audio, text = batch['audio'], batch['transcript']
                model_output = model_trainer.model(audio, text, skip_kde=True, annotator_masks=all_annotators_masks, curr_task='zero-shot adaptation')
                z_act_items.append(model_output['soft_act_preds'][~model_output['soft_act_preds'].isnan()].view(-1))
                z_val_items.append(model_output['soft_val_preds'][~model_output['soft_val_preds'].isnan()].view(-1))
            coral_process_act.init_from_enrollment_set(torch.cat(z_act_items))
            coral_process_val.init_from_enrollment_set(torch.cat(z_val_items))
            model_trainer.model.coral_processes = {'act': coral_process_act, 'val': coral_process_val}
        else:
            model_trainer.model.coral_processes = None

        for batch in tqdm(eval_dataloader):
            # First prepare values from batch
            audio, text = batch['audio'], batch['transcript']
            assert len(audio) == 1, 'Batch size must be 1 during finding best annotators'

            if randomly_map_annotators:
                # We just want to track annotators in the annotator level results 
                for i, annotator in enumerate(batch['annotators'][0]):
                    annotator_level_results[annotator]['FileName'].extend(batch['FileName'])
            else:
                # Make predictions using every trained annotator on podcast
                model_output = model_trainer.model(audio, text, skip_kde=True, annotator_masks=all_annotators_masks, curr_task='zero-shot adaptation') # curr_task only used for logging the last seen task during model output explosion to diagnose culprit
                if model_trainer.model.args.model_type == ModelType.AGGREGATE_GROUND_TRUTH:
                    model_output = {'act': batch['act'], 'val': batch['val']}
                    # We are just trying to predict the aggregate ground truth mapping
                    # so make fake soft act labels
                    # If later want to extend this beyond aggregate ground truth could instead just do this
                    # assignment when soft_act_labels is missing for example
                    model_output['soft_act_preds'] = model_output['act']
                    model_output['soft_val_preds'] = model_output['val']

                # Now store annotator level results 
                # These values are used for finding most similar annotators or for reporting annotator-level metrics 
                for i, annotator in enumerate(batch['annotators'][0]):
                    ann_act, ann_val = batch['soft_act_labels'][0][i], batch['soft_val_labels'][0][i]

                    # Store annotator y values
                    annotator_level_results[annotator]['y_act'].append(ann_act)
                    annotator_level_results[annotator]['y_val'].append(ann_val)

                    # Store every single prediction from all pre-trained annotator heads 
                    act_pred = model_output['soft_act_preds'].squeeze()
                    val_pred = model_output['soft_val_preds'].squeeze()
                    if map_settings is not None and 'use_CORAL' in map_settings and map_settings['use_CORAL']:
                        num_bins = act_pred.shape[-1]
                        scale = lambda x: -1 + (x)*(2)/(num_bins-1)
                        idxs = torch.arange(num_bins, device=act_pred.device)
                        act_probs = (act_pred * idxs.view(1, -1)).sum(dim=-1)
                        val_probs = (val_pred * idxs.view(1, -1)).sum(dim=-1)
                        act_pred = scale(act_probs.round().clamp(0, num_bins-1))
                        val_pred = scale(val_probs.round().clamp(0, num_bins-1))

                    annotator_level_results[annotator]['podcast_acts'].append(act_pred)
                    annotator_level_results[annotator]['podcast_vals'].append(val_pred)

                    # Store filenames of current sample to filter later to only selected idxs
                    annotator_level_results[annotator]['FileName'].extend(batch['FileName'])

    mapping = {}
    if randomly_map_annotators:
        num_annotators_to_choose_from = np.arange(len(all_annotators))
        for ann in annotator_level_results:
            if map_act_val_separately:
                mapping[ann] = {
                    'act': np.random.choice(num_annotators_to_choose_from, size=N, replace=False).tolist(),
                    'val': np.random.choice(num_annotators_to_choose_from, size=N, replace=False).tolist(),
                }
            else:
                mapping[ann] = np.random.choice(num_annotators_to_choose_from, size=N, replace=False).tolist()
    else:
        print(f'During mapping annotators, found {len(annotator_level_results)} new annotators')
        for ann in annotator_level_results:
            for key in annotator_level_results[ann]:
                if key == 'FileName':
                    continue
                annotator_level_results[ann][key] = torch.stack(annotator_level_results[ann][key])

        if map_settings is None:
            map_settings = {}
        avg_act_diff = []
        avg_val_diff = []
        for ann in annotator_level_results:
            map_at_monologue_level = 'map_at_monologue_level' in map_settings and map_settings['map_at_monologue_level'] and 'self_report' in ann
            if map_at_monologue_level:
                print(f'Mapping {ann} at monologue level')
                # We need to reduce the predictions to monologue level before computing the correlation
                get_monologue = lambda fname: '_'.join(fname.split('_')[1:-2])
                y_act_monologue = {}
                y_val_monologue = {}
                act_preds_monologue = {}
                val_preds_monologue = {}
                assert len(annotator_level_results[ann]['FileName']) == len(annotator_level_results[ann]['y_act'])
                for i, fname in enumerate(annotator_level_results[ann]['FileName']):
                    monologue = get_monologue(fname)
                    if monologue not in y_act_monologue:
                        y_act_monologue[monologue] = []
                        y_val_monologue[monologue] = []
                        act_preds_monologue[monologue] = []
                        val_preds_monologue[monologue] = []
                    y_act_monologue[monologue].append(annotator_level_results[ann]['y_act'][i])
                    y_val_monologue[monologue].append(annotator_level_results[ann]['y_val'][i])
                    act_preds_monologue[monologue].append(annotator_level_results[ann]['podcast_acts'][i])
                    val_preds_monologue[monologue].append(annotator_level_results[ann]['podcast_vals'][i])
                    print(fname, monologue, len(y_act_monologue[monologue]))
                ann_y_act = []
                ann_y_val = []
                act_preds = []
                val_preds = []
                for monologue in y_act_monologue:
                    assert all([y_act_monologue[monologue][0] == y for y in y_act_monologue[monologue]]), f'{y_act_monologue[monologue]}'
                    assert all([y_val_monologue[monologue][0] == y for y in y_val_monologue[monologue]]), f'{y_val_monologue[monologue]}'
                    ann_y_act.append(y_act_monologue[monologue][0])
                    ann_y_val.append(y_val_monologue[monologue][0])
                    # print(f'Monologue: {monologue} | ann_y_act: {torch.stack(ann_y_act).shape} | ann_act_preds: {torch.stack(act_preds_monologue[monologue]).shape}')
                    act_preds.append(torch.stack(act_preds_monologue[monologue]).mean(dim=0))
                    val_preds.append(torch.stack(val_preds_monologue[monologue]).mean(dim=0)) # Average over all samples (dim=0) to get average prediction for monologue for each pre-trained annotators (dim=1)
                ann_y_act = torch.stack(ann_y_act)
                ann_y_val = torch.stack(ann_y_val)
                act_preds = torch.stack(act_preds)
                val_preds = torch.stack(val_preds)
            else:
                ann_y_act = annotator_level_results[ann]['y_act']
                ann_y_val = annotator_level_results[ann]['y_val']
                act_preds = annotator_level_results[ann]['podcast_acts']
                val_preds = annotator_level_results[ann]['podcast_vals']

            if num_samples_to_use_for_map != 'all':
                if len(ann_y_act) < num_samples_to_use_for_map:
                    raise ValueError(f'Value less than 30 should have been filtered out. Found {len(ann_y_act)=} < {num_samples_to_use_for_map=} which should never be possible if filtered correctly.')
                # If there are <= num_samples_to_use_for_map samples then just use all samples for the mapping
                # otherwise, at this point we want to randomly select num_samples_to_use_for_map samples to use for the mapping process
                sampled_idxs = torch.randperm(len(ann_y_act))[:num_samples_to_use_for_map]
                ann_y_act = ann_y_act[sampled_idxs]
                ann_y_val = ann_y_val[sampled_idxs]
                act_preds = act_preds[sampled_idxs]
                val_preds = val_preds[sampled_idxs]
                # print('[INFO]: Number of samples used for mapping =', act_preds.shape)
            if model_trainer.model.args.model_type == ModelType.AGGREGATE_GROUND_TRUTH:
                m2a = {0: 'Aggregate Ground Truth'}
            else:
                m2a = model_trainer.model.prediction_head.act_heads.annotator_mapper.map_to_annotator
            logger.log_scalar(f'Mapping Information Training {dataset_name}', f'{ann} Number of Samples Used For Mapping', model_trainer.name, len(ann_y_act), 0)
            map_with_pearson = 'use_pearson' in map_settings and map_settings['use_pearson']
            if map_with_pearson:
                print(f'Using Pearson correlation for best annotator mapping')
            comp_fn = pearson if map_with_pearson else ccc
            if len(ann_y_act) > 1:
                # act_cccs1 = comp_fn(act_preds, ann_y_act, multiple_annotators=True)
                # val_cccs1 = comp_fn(val_preds, ann_y_val, multiple_annotators=True)
                # if map_settings is not None and 'use_CORAL' in map_settings and map_settings['use_CORAL']:
                #     # Use spearman correlation for CORAL as both values will be ordinal 
                #     # act_preds is Batch x Num Pre-trained Annotators
                #     act_cccs = []
                #     val_cccs = []
                #     print(f'Using Spearman correlation for CORAL')
                #     for ann in range(act_preds.shape[1]):
                #         act_cccs.append(spearmanr(act_preds[:, ann].cpu().numpy(), ann_y_act.cpu().numpy()).correlation)
                #         val_cccs.append(spearmanr(val_preds[:, ann].cpu().numpy(), ann_y_val.cpu().numpy()).correlation)
                #         print('activation:', act_preds[:, ann], ann_y_act, 'CCC would have been', act_cccs1[ann], 'spearman was', act_cccs[-1])
                #         print('valence:', val_preds[:, ann], ann_y_val, 'CCC would have been', val_cccs1[ann], 'spearman was', val_cccs[-1])

                #     act_cccs = torch.as_tensor(act_cccs)
                #     val_cccs = torch.as_tensor(val_cccs)
                # else:
                act_cccs = comp_fn(act_preds, ann_y_act, multiple_annotators=True)
                val_cccs = comp_fn(val_preds, ann_y_val, multiple_annotators=True)

                if map_act_val_separately:
                    _, best_act_anns = act_cccs.topk(N)
                    _, best_val_anns = val_cccs.topk(N)
                    best_anns = {'act': best_act_anns, 'val': best_val_anns}
                    print(f'Best annotator(s) for {ann} = {best_anns}={[m2a[a.item()] for a in torch.cat([best_anns["act"], best_anns["val"]])]} | act_cccs={[act_cccs[a] for a in best_act_anns]} val_cccs={[val_cccs[a] for a in best_val_anns]}')
                    for i, a in enumerate(best_anns['act']):
                        logger.log_scalar(f'Mapping Information Training {dataset_name}', f'{ann} Mapped Activation CCC', model_trainer.name, act_cccs[a], i)
                        logger.log_scalar(f'Mapping Information Training {dataset_name}', f'{ann}-Activation Mapped To Pre-trained Annotators', model_trainer.name, m2a[a.item()], i)
                    for i, a in enumerate(best_anns['val']):
                        logger.log_scalar(f'Mapping Information Training {dataset_name}', f'{ann} Mapped Valence CCC', model_trainer.name, val_cccs[a], i)
                        logger.log_scalar(f'Mapping Information Training {dataset_name}', f'{ann}-Valence Mapped To Pre-trained Annotators', model_trainer.name, m2a[a.item()], i)
                else:
                    _, best_anns = (act_cccs+val_cccs).topk(N)
                    print(f'Best annotator(s) for {ann} = {best_anns}={[m2a[a.item()] for a in best_anns]} | act_cccs={[act_cccs[a] for a in best_anns]} val_cccs={[val_cccs[a] for a in best_anns]}')
                    for i, a in enumerate(best_anns):
                        logger.log_scalar(f'Mapping Information Training {dataset_name}', f'{ann} Mapped Activation CCC', model_trainer.name, act_cccs[a], i)
                        logger.log_scalar(f'Mapping Information Training {dataset_name}', f'{ann} Mapped Valence CCC', model_trainer.name, val_cccs[a], i)
                        logger.log_scalar(f'Mapping Information Training {dataset_name}', f'{ann}-Combined Mapped To Pre-trained Annotators', model_trainer.name, m2a[a.item()], i)
            else:
                act_diff = torch.abs(ann_y_act - act_preds).squeeze()
                val_diff = torch.abs(ann_y_val - val_preds).squeeze()
                if map_act_val_separately:
                    _, best_act_anns = act_diff.topk(N, largest=False)
                    _, best_val_anns = val_diff.topk(N, largest=False)
                    best_anns = {'act': best_act_anns, 'val': best_val_anns}
                    print(f'Best annotator(s) for {ann} = {best_anns}={[m2a[a.item()] for a in torch.cat([best_anns["act"], best_anns["val"]])]} | act_diff={[act_diff[a] for a in best_act_anns]} val_diff={[val_diff[a] for a in best_val_anns]}')
                    for i, a in enumerate(best_anns['act']):
                        logger.log_scalar(f'Mapping Information Training {dataset_name}', f'{ann} Mapped Activation MAE', model_trainer.name, act_diff[a], i)
                        logger.log_scalar(f'Mapping Information Training {dataset_name}', f'{ann}-Activation Mapped To Pre-trained Annotators', model_trainer.name, m2a[a.item()], i)
                    for i, a in enumerate(best_anns['val']):
                        logger.log_scalar(f'Mapping Information Training {dataset_name}', f'{ann} Mapped Valence MAE', model_trainer.name, val_diff[a], i)
                        logger.log_scalar(f'Mapping Information Training {dataset_name}', f'{ann}-Valence Mapped To Pre-trained Annotators', model_trainer.name, m2a[a.item()], i)
                else:
                    _, best_anns = (act_diff+val_diff).topk(N, largest=False)
                    print(f'Best annotator(s) for {ann} = {best_anns}={[m2a[a.item()] for a in best_anns]} | act_diff={[act_diff[a] for a in best_anns]} val_diff={[val_diff[a] for a in best_anns]}')
                    for i, a in enumerate(best_anns):
                        logger.log_scalar(f'Mapping Information Training {dataset_name}', f'{ann} Mapped Activation MAE', model_trainer.name, act_diff[a], i)
                        logger.log_scalar(f'Mapping Information Training {dataset_name}', f'{ann} Mapped Valence MAE', model_trainer.name, val_diff[a], i)
                        logger.log_scalar(f'Mapping Information Training {dataset_name}', f'{ann}-Combined Mapped To Pre-trained Annotators', model_trainer.name, m2a[a.item()], i)
            mapping[ann] = best_anns
        print(f'avg_act_diff={np.mean(avg_act_diff)}, avg_val_diff={np.mean(avg_val_diff)}')
    model_trainer.model.train()
    return mapping

def ridge_with_bias(A: torch.Tensor, y: torch.Tensor, lam: float = 1e-2):
    """
    Solve argmin_w,b ||A w + b - y||^2 + lam ||w||^2
    A: [n, K], y: [n]
    Returns: w [K], b scalar
    """
    n, k = A.shape
    ones = torch.ones(n, 1, device=A.device, dtype=A.dtype)
    A_aug = torch.cat([A, ones], dim=1)  # [n, K+1]

    # Ridge only on w, not on bias term
    reg = torch.zeros(k + 1, k + 1, device=A.device, dtype=A.dtype)
    reg[:k, :k] = lam * torch.eye(k, device=A.device, dtype=A.dtype)

    theta = torch.linalg.solve(A_aug.T @ A_aug + reg, A_aug.T @ y)  # [K+1]
    w = theta[:k]
    b = theta[k]
    return w, b

def find_optimal_basis_coefficients(
    eval_dataloader,
    model_trainer,
    num_samples_to_use_for_map: Union[str, int],
    logger,
    map_act_val_separately: bool = False,
    dataset_name: Optional[str] = None
) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
    """
    Find optimal linear combination of basis annotators for each new annotator in the evaluation dataset.
    
    This function solves a least-squares problem to find coefficients w such that:
    w[0] * basis_pred_0 + w[1] * basis_pred_1 + ... + w[7] * basis_pred_7
    minimizes the error with respect to each new annotator's labels.
    
    Args:
        eval_dataloader: DataLoader for evaluation data
        model_trainer: Model trainer containing the orthogonal model with basis annotators
        num_samples_to_use_for_map: Number of samples to use for computing coefficients ('all' or int)
        logger: Logger for recording mapping information
        map_act_val_separately: Whether to compute separate coefficients for activation and valence
        dataset_name: Name of the dataset being finetuned on
        
    Returns:
        Dictionary mapping annotator names to their optimal basis coefficients
        Format: {annotator_name: coefficients} or {annotator_name: {'act': act_coeffs, 'val': val_coeffs}}
    """
    model_trainer.model.eval()
    
    # Verify this is an orthogonal model
    if model_trainer.model.args.model_type != ModelType.INDIVIDUAL_ANNOTATOR_ORTHOGONAL:
        raise ValueError(f"find_optimal_basis_coefficients only works with orthogonal models, got {model_trainer.model.args.model_type}")
    
    num_basis_annotators = model_trainer.model.args.basis_annotator_rank
    print(f'Computing optimal coefficients for {num_basis_annotators} basis annotators')
    
    # Collect basis predictions and target labels for each new annotator
    annotator_data = defaultdict(lambda: {
        'basis_act_preds': [],  # List of [num_basis_annotators] tensors
        'basis_val_preds': [],  # List of [num_basis_annotators] tensors
        'y_act': [],            # List of scalar tensors
        'y_val': [],            # List of scalar tensors
        'FileName': []
    })
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Collecting basis predictions"):
            audio, text = batch['audio'], batch['transcript']
            assert len(audio) == 1, 'Batch size must be 1 during coefficient computation'

            # Get basis predictions from the model
            # We don't need annotator-specific predictions, just the basis predictions
            # Create a dummy mask to get all basis predictions
            batch_size = 1
            # Use a simple forward pass to get basis predictions
            model_output = model_trainer.model(audio, text, skip_kde=True)

            # Extract basis predictions
            # basis_act_predictions shape: [batch_size, num_basis_annotators]
            # basis_val_predictions shape: [batch_size, num_basis_annotators]
            basis_act_preds = model_output['basis_act_predictions']  # [1, num_basis]
            basis_val_preds = model_output['basis_val_predictions']  # [1, num_basis]

            # Store data for each annotator in this sample
            for i, annotator in enumerate(batch['annotators'][0]):
                ann_act = batch['soft_act_labels'][0][i]
                ann_val = batch['soft_val_labels'][0][i]

                # Store basis predictions and target labels
                annotator_data[annotator]['basis_act_preds'].append(basis_act_preds.squeeze(0))  # [num_basis]
                annotator_data[annotator]['basis_val_preds'].append(basis_val_preds.squeeze(0))  # [num_basis]
                annotator_data[annotator]['y_act'].append(ann_act)
                annotator_data[annotator]['y_val'].append(ann_val)
                annotator_data[annotator]['FileName'].append(batch['FileName'][0])
    
    # Compute optimal coefficients for each annotator using least-squares
    mapping = {}
    print(f'Computing optimal basis coefficients for {len(annotator_data)} new annotators')
    
    for ann in annotator_data:
        # Stack all samples for this annotator
        basis_act_preds = torch.stack(annotator_data[ann]['basis_act_preds'])  # [n_samples, num_basis]
        basis_val_preds = torch.stack(annotator_data[ann]['basis_val_preds'])  # [n_samples, num_basis]
        y_act = torch.stack(annotator_data[ann]['y_act'])  # [n_samples]
        y_val = torch.stack(annotator_data[ann]['y_val'])  # [n_samples]
        
        # Subsample if requested
        if num_samples_to_use_for_map != 'all':
            if len(y_act) < num_samples_to_use_for_map:
                raise ValueError(f'Annotator {ann} has {len(y_act)} samples but {num_samples_to_use_for_map} requested')
            
            sampled_idxs = torch.randperm(len(y_act))[:num_samples_to_use_for_map]
            basis_act_preds = basis_act_preds[sampled_idxs]
            basis_val_preds = basis_val_preds[sampled_idxs]
            y_act = y_act[sampled_idxs]
            y_val = y_val[sampled_idxs]
        
        logger.log_scalar(
            f'Mapping Information Training {dataset_name}',
            f'{ann} Number of Samples Used For Basis Coefficient Computation',
            model_trainer.name,
            len(y_act),
            0
        )
        
        lam = 1e-2  # tune (1e-3, 1e-2, 1e-1 are good starting points)

        if map_act_val_separately:
            act_w, act_b = ridge_with_bias(basis_act_preds, y_act, lam=lam)
            val_w, val_b = ridge_with_bias(basis_val_preds, y_val, lam=lam)

            act_pred = basis_act_preds @ act_w + act_b
            val_pred = basis_val_preds @ val_w + val_b

            act_residual = torch.mean((act_pred - y_act) ** 2).item()
            val_residual = torch.mean((val_pred - y_val) ** 2).item()
            act_ccc_val = ccc(act_pred.unsqueeze(-1), y_act.unsqueeze(-1)).item()
            val_ccc_val = ccc(val_pred.unsqueeze(-1), y_val.unsqueeze(-1)).item()

            print(f'Ridge+bias coefficients for {ann}:')
            print(f'  Act: w={act_w.cpu().numpy()} b={act_b.item():+.4f} (MSE={act_residual:.4f}, CCC={act_ccc_val:.4f})')
            print(f'  Val: w={val_w.cpu().numpy()} b={val_b.item():+.4f} (MSE={val_residual:.4f}, CCC={val_ccc_val:.4f})')

            logger.log_scalar(f'Mapping Information Training {dataset_name}', f'{ann} Ridge+bias Activation CCC', model_trainer.name, act_ccc_val, 0)
            logger.log_scalar(f'Mapping Information Training {dataset_name}', f'{ann} Ridge+bias Valence CCC', model_trainer.name, val_ccc_val, 0)
            logger.log_scalar(f'Mapping Information Training {dataset_name}', f'{ann} Ridge+bias Activation MSE', model_trainer.name, act_residual, 0)
            logger.log_scalar(f'Mapping Information Training {dataset_name}', f'{ann} Ridge+bias Valence MSE', model_trainer.name, val_residual, 0)

            mapping[ann] = {'act_w': act_w, 'act_b': act_b, 'val_w': val_w, 'val_b': val_b}

        else:
            # joint solve (not recommended, but here for completeness)
            A = torch.cat([basis_act_preds, basis_val_preds], dim=0)  # [2n, K]
            y = torch.cat([y_act, y_val], dim=0)                     # [2n]
            w, b = ridge_with_bias(A, y, lam=lam)

            act_pred = basis_act_preds @ w + b
            val_pred = basis_val_preds @ w + b

            act_residual = torch.mean((act_pred - y_act) ** 2).item()
            val_residual = torch.mean((val_pred - y_val) ** 2).item()
            act_ccc_val = ccc(act_pred.unsqueeze(-1), y_act.unsqueeze(-1)).item()
            val_ccc_val = ccc(val_pred.unsqueeze(-1), y_val.unsqueeze(-1)).item()
            combined_ccc = (act_ccc_val + val_ccc_val) / 2

            print(f'Ridge+bias coefficients for {ann}: w={w.cpu().numpy()} b={b.item():+.4f} (CCC={combined_ccc:.4f})')

            mapping[ann] = {'w': w, 'b': b}
            
    model_trainer.model.train()
    return mapping