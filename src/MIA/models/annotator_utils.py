import torch
from typing import Dict, Any


def save_annotator_weights(model, annotator: str, annotator_best_weights: Dict[str, Any]) -> None:
    """
    Save the current weights for a specific annotator
    
    Args:
        model: The model containing the prediction heads
        annotator: The annotator ID to save weights for
        annotator_best_weights: Dictionary to store the saved weights
    """
    if not hasattr(model, 'prediction_head') or not hasattr(model.prediction_head, 'act_heads'):
        return
    
    annotator_mapper = model.prediction_head.act_heads.annotator_mapper
    if annotator not in annotator_mapper.map_to_idx:
        return
        
    ann_idx = annotator_mapper.map_to_idx[annotator]
    
    # Save activation weights
    act_weight = model.prediction_head.act_heads.weight[ann_idx, :, :].clone().detach()
    act_bias = model.prediction_head.act_heads.bias[ann_idx, :].clone().detach()
    
    # Save valence weights
    val_weight = model.prediction_head.val_heads.weight[ann_idx, :, :].clone().detach()
    val_bias = model.prediction_head.val_heads.bias[ann_idx, :].clone().detach()
    
    # Save variance weights if they exist
    act_var_weight = None
    act_var_bias = None
    val_var_weight = None
    val_var_bias = None
    
    if hasattr(model.prediction_head, 'act_head_var'):
        act_var_weight = model.prediction_head.act_head_var.weight[ann_idx, :, :].clone().detach()
        act_var_bias = model.prediction_head.act_head_var.bias[ann_idx, :].clone().detach()
        
    if hasattr(model.prediction_head, 'val_head_var'):
        val_var_weight = model.prediction_head.val_head_var.weight[ann_idx, :, :].clone().detach()
        val_var_bias = model.prediction_head.val_head_var.bias[ann_idx, :].clone().detach()
    
    annotator_best_weights[annotator] = {
        'act_weight': act_weight,
        'act_bias': act_bias,
        'val_weight': val_weight,
        'val_bias': val_bias,
        'act_var_weight': act_var_weight,
        'act_var_bias': act_var_bias,
        'val_var_weight': val_var_weight,
        'val_var_bias': val_var_bias
    }


def restore_annotator_weights(model, annotator: str, annotator_best_weights: Dict[str, Any]) -> None:
    """
    Restore the best weights for a specific annotator
    
    Args:
        model: The model containing the prediction heads
        annotator: The annotator ID to restore weights for
        annotator_best_weights: Dictionary containing the saved weights
    """
    if annotator not in annotator_best_weights:
        return
        
    if not hasattr(model, 'prediction_head') or not hasattr(model.prediction_head, 'act_heads'):
        return
    
    annotator_mapper = model.prediction_head.act_heads.annotator_mapper
    if annotator not in annotator_mapper.map_to_idx:
        return
        
    ann_idx = annotator_mapper.map_to_idx[annotator]
    weights = annotator_best_weights[annotator]
    
    # Restore activation weights
    with torch.no_grad():
        model.prediction_head.act_heads.weight[ann_idx, :, :] = weights['act_weight']
        model.prediction_head.act_heads.bias[ann_idx, :] = weights['act_bias']
        
        # Restore valence weights
        model.prediction_head.val_heads.weight[ann_idx, :, :] = weights['val_weight']
        model.prediction_head.val_heads.bias[ann_idx, :] = weights['val_bias']
        
        # Restore variance weights if they exist
        if weights['act_var_weight'] is not None and hasattr(model.prediction_head, 'act_head_var'):
            model.prediction_head.act_head_var.weight[ann_idx, :, :] = weights['act_var_weight']
            model.prediction_head.act_head_var.bias[ann_idx, :] = weights['act_var_bias']
            
        if weights['val_var_weight'] is not None and hasattr(model.prediction_head, 'val_head_var'):
            model.prediction_head.val_head_var.weight[ann_idx, :, :] = weights['val_var_weight']
            model.prediction_head.val_head_var.bias[ann_idx, :] = weights['val_var_bias']


def compute_annotator_ccc(validation_results, removed_annotators: set) -> Dict[str, float]:
    """
    Compute CCC (Concordance Correlation Coefficient) for each annotator
    
    Args:
        validation_results: Results from validation containing model outputs and target values
        removed_annotators: Set of annotators that have been removed from training
        
    Returns:
        Dictionary mapping annotator IDs to their CCC values
    """
    from MIA.metrics import ccc
    
    metrics = {}
    annotators = validation_results.target_values.get('ccc_ind_annotators', [])
    
    for ann in annotators:
        if ann in removed_annotators:
            pass # Temporarily disable skipping of removed annotators
            # continue
        try:
            # The tensors are already concatenated during evaluation, so don't concatenate again
            pred_act = validation_results.model_outputs[f'pred_act_{ann}']
            pred_val = validation_results.model_outputs[f'pred_val_{ann}']
            y_act = validation_results.target_values[f'y_act_{ann}']
            y_val = validation_results.target_values[f'y_val_{ann}']
        except KeyError:
            continue
        if len(pred_act) < 2:
            continue
        metrics[ann] = (ccc(pred_act, y_act) + ccc(pred_val, y_val)) / 2
    
    return metrics


def compute_annotator_ccc_separate(validation_results, removed_annotators: set) -> Dict[str, Dict[str, float]]:
    """
    Compute separate activation and valence CCC (Concordance Correlation Coefficient) for each annotator
    
    Args:
        validation_results: Results from validation containing model outputs and target values
        removed_annotators: Set of annotators that have been removed from training
        
    Returns:
        Dictionary mapping annotator IDs to dictionaries containing 'activation' and 'valence' CCC values
    """
    from MIA.metrics import ccc
    
    metrics = {}
    annotators = validation_results.target_values.get('ccc_ind_annotators', [])
    
    for ann in annotators:
        if ann in removed_annotators:
            pass # Temporarily disable skipping of removed annotators
            # continue
        try:
            # The tensors are already concatenated during evaluation, so don't concatenate again
            pred_act = validation_results.model_outputs[f'pred_act_{ann}']
            pred_val = validation_results.model_outputs[f'pred_val_{ann}']
            y_act = validation_results.target_values[f'y_act_{ann}']
            y_val = validation_results.target_values[f'y_val_{ann}']
        except KeyError:
            continue
        if len(pred_act) < 2:
            continue
        metrics[ann] = {
            'activation': ccc(pred_act, y_act),
            'valence': ccc(pred_val, y_val)
        }
    
    return metrics