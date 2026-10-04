import torch

from typing import Dict, Any
import torch

def concatenate_outputs(
    model_outputs: Dict[str, torch.Tensor],
    target_values: Dict[str, torch.Tensor]
):
    """Concatenate lists of tensors into single tensors for metrics and loss computation."""
    # Keys that need special unpad/repad handling
    padded_keys = [
        'soft_act_preds', 'soft_val_preds', 'act_log_vars', 'val_log_vars',
        'y_padded_individual_annotators_act', 'y_padded_individual_annotators_val',
        'soft_act_zs', 'soft_val_zs', 'act_probability_logits', 'val_probability_logits'
    ]
    
    # Process model outputs
    for key, value in model_outputs.items():
        if isinstance(value, list) and len(value) > 0:
            if isinstance(value[0], torch.Tensor):
                # print('Concatenating', key, value[0].shape)
                if len(value) == 1:
                    # Single tensor, just unwrap from list
                    model_outputs[key] = value[0]
                else:
                    # Check if this key needs special unpad/repad handling
                    if key in padded_keys:
                        model_outputs[key] = unpad_repad_tensors(value, key)
                    else:
                        # Regular concatenation
                        model_outputs[key] = torch.cat(value)
                
    # Process target values
    for key, value in target_values.items():
        if isinstance(value, list) and len(value) > 0:
            if isinstance(value[0], torch.Tensor):
                if len(value) == 1:
                    # Single tensor, just unwrap from list
                    target_values[key] = value[0]
                else:
                    # Check if this key needs special unpad/repad handling
                    if key in padded_keys:
                        target_values[key] = unpad_repad_tensors(value, key)
                    else:
                        # Regular concatenation
                        target_values[key] = torch.cat(value)

def process_outputs(
    outputs: Dict[str, torch.Tensor],
    model_outputs: Dict[str, torch.Tensor],
):
    """Process and store model outputs.
    
    For mean predictions:
    - Baseline/aggregate models: Use 'act'/'val' as mean predictions
    - Individual annotator models: Use 'mean_act_preds'/'mean_val_preds' if available,
        otherwise use 'act'/'val' as mean predictions
    """
    # First handle mean predictions
    if 'act' in outputs and 'val' in outputs:
        # For baseline/aggregate models or when mean preds are in act/val
        if 'mean_act_preds' not in model_outputs:
            model_outputs['mean_act_preds'] = []
        if 'mean_val_preds' not in model_outputs:
            model_outputs['mean_val_preds'] = []
        model_outputs['mean_act_preds'].append(outputs['act'].cpu())
        model_outputs['mean_val_preds'].append(outputs['val'].cpu())
    
    # Process all other outputs
    for key, value in outputs.items():
        if key not in model_outputs:
            model_outputs[key] = []
            
        if key.endswith('z_log_var'):
            # Handle latent variables
            dim = 'act' if 'act' in key else 'val'
            soft_z, log_var = value
            if soft_z is not None:
                if f'soft_{dim}_zs' not in model_outputs:
                    model_outputs[f'soft_{dim}_zs'] = []
                model_outputs[f'soft_{dim}_zs'].append(soft_z.cpu())
            if log_var is not None:
                if f'{dim}_log_vars' not in model_outputs:
                    model_outputs[f'{dim}_log_vars'] = []
                model_outputs[f'{dim}_log_vars'].append(log_var.cpu())
        else:
            if ('kde' in key or 'probability' in key) and value is None:
                model_outputs[key].append(None)
            else:
                if value is None:
                    raise ValueError(f"Unexpected None value for output key '{key}'. "
                                    f"Only KDE-related outputs should be None.")
                model_outputs[key].append(value.cpu())

def process_targets(
    batch: Dict[str, Any],
    target_values: Dict[str, torch.Tensor]
):
    """Process and store target values."""
    for key, value in batch.items():
        if key in ['audio', 'transcript', 'annotator_masks', 'annotators', 'FileName']:
            continue
            
        target_key = f'y_{key}'
        if target_key not in target_values:
            target_values[target_key] = []
            
        if isinstance(value, list):
            target_values[target_key].append(value)
        else:
            target_values[target_key].append(value.cpu())

def unpad_repad_tensors(tensor_list, key):
    """Unpad a list of padded tensors and repad them into a single tensor."""
    # if key == 'soft_act_preds' or key == 'soft_val_preds':
    #     print('should 2d pad soft_act_preds for coral')
    #     print(key, len(tensor_list), [x.shape for x in tensor_list])

    two_d_pad_keys = ['soft_act_zs', 'soft_val_zs', 'act_probability_logits', 'val_probability_logits']#, 'soft_act_preds', 'soft_val_preds']
    two_d_pad = any(k in key for k in two_d_pad_keys)

    if not two_d_pad:
        results = [
            torch.nn.utils.rnn.unpad_sequence(item, (~item.isnan()).sum(dim=-1), batch_first=True)
            for item in tensor_list
        ]
    else:
        results = [
            torch.nn.utils.rnn.unpad_sequence(item, (~item.sum(dim=-1).isnan()).sum(dim=-1), batch_first=True)
            for item in tensor_list
        ]

    results = [i for y in results for i in y]
    results = torch.nn.utils.rnn.pad_sequence(results, batch_first=True, padding_value=torch.nan)
    return results

def process_masks_and_full_padded_values(batch, outputs, masks_and_full_padded_values, i):
    batch_mask, weight_mask, output_mask, _ = batch['annotator_masks']
    largest_seen_batch = 0 if 'largest_seen_batch' not in masks_and_full_padded_values else masks_and_full_padded_values['largest_seen_batch']
    batch_size = batch['padded_individual_annotators_act'].shape[0]
    corrected_batch_mask = batch_mask + i * batch_size # This is used to duplicate batches, so when we stack them we just need to correct this to match the new index after stacking
    masks_and_full_padded_values['batch_mask'].append(corrected_batch_mask)
    masks_and_full_padded_values['weight_mask'].append(weight_mask)
    new_annotator_batch_size = batch['padded_individual_annotators_act'].shape[-1]
    if new_annotator_batch_size > largest_seen_batch:
        num_new_columns = new_annotator_batch_size - largest_seen_batch
        samples_to_fix = len(masks_and_full_padded_values['padded_y_act'])
        for j in range(samples_to_fix):
            # print(f'Padding {num_new_columns} columns onto samples with shape: {masks_and_full_padded_values['padded_y_act'][j].shape}, {masks_and_full_padded_values['padded_y_val'][j].shape}, {masks_and_full_padded_values['padded_act_preds'][j].shape}, {masks_and_full_padded_values['padded_val_preds'][j].shape}')
            new_empty_columns = torch.full((masks_and_full_padded_values['padded_y_act'][j].shape[0], num_new_columns), torch.nan, device=masks_and_full_padded_values['padded_y_act'][j].device)
            # print(f'New empty columns shape: {new_empty_columns.shape}')
            masks_and_full_padded_values['padded_y_act'][j] = torch.cat([masks_and_full_padded_values['padded_y_act'][j], new_empty_columns], dim=-1)
            masks_and_full_padded_values['padded_y_val'][j] = torch.cat([masks_and_full_padded_values['padded_y_val'][j], new_empty_columns], dim=-1)
            masks_and_full_padded_values['padded_act_preds'][j] = torch.cat([masks_and_full_padded_values['padded_act_preds'][j], new_empty_columns], dim=-1)
            masks_and_full_padded_values['padded_val_preds'][j] = torch.cat([masks_and_full_padded_values['padded_val_preds'][j], new_empty_columns], dim=-1)
        masks_and_full_padded_values['largest_seen_batch'] = new_annotator_batch_size
    elif new_annotator_batch_size < largest_seen_batch:
        num_new_columns = largest_seen_batch - new_annotator_batch_size
        new_empty_columns = torch.full((batch['padded_individual_annotators_act'].shape[0], num_new_columns), torch.nan, device=batch['padded_individual_annotators_act'].device)
        # print(f'Padding {num_new_columns} columns onto samples with shape: {batch['padded_individual_annotators_act'].shape}, {batch['padded_individual_annotators_val'].shape}, {outputs['soft_act_preds'].shape}, {outputs['soft_val_preds'].shape}')
        # print(f'New empty columns shape: {new_empty_columns.shape}')
        batch['padded_individual_annotators_act'] = torch.cat([batch['padded_individual_annotators_act'], new_empty_columns], dim=-1)
        batch['padded_individual_annotators_val'] = torch.cat([batch['padded_individual_annotators_val'], new_empty_columns], dim=-1)
        outputs['soft_act_preds'] = torch.cat([outputs['soft_act_preds'], new_empty_columns], dim=-1)
        outputs['soft_val_preds'] = torch.cat([outputs['soft_val_preds'], new_empty_columns], dim=-1)

    masks_and_full_padded_values['padded_y_act'].append(batch['padded_individual_annotators_act'])
    masks_and_full_padded_values['padded_y_val'].append(batch['padded_individual_annotators_val'])
    masks_and_full_padded_values['padded_act_preds'].append(outputs['soft_act_preds'])
    masks_and_full_padded_values['padded_val_preds'].append(outputs['soft_val_preds'])
    return masks_and_full_padded_values