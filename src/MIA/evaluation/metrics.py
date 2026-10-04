"""Metrics calculation for emotion recognition models."""

from typing import Dict, List, Optional, Tuple, Any
import torch
from MIA.csv_logging import CSVWriter
from MIA.metrics import ccc, total_variation, total_variation_1d, JSD, JSD_1d
import numpy as np
from sklearn.metrics import recall_score

def calculate_ccc_metrics(
    model_outputs: Dict[str, torch.Tensor],
    target_values: Dict[str, torch.Tensor],
    type_: str,
    name: str,
    step: int,
    logger: Optional[CSVWriter] = None
) -> Dict[str, float]:
    """Calculate CCC metrics.
    
    Args:
        model_outputs: Dictionary of model outputs
        target_values: Dictionary of target values
        type_: Type of evaluation (validation/test)
        name: Model name for logging
        step: Current step/epoch
        logger: Logger instance
        
    Returns:
        Dictionary of metric values
    """
    if len(model_outputs['act'].shape) > 1 and model_outputs['act'].shape[-1] > 1:
        num_bins = model_outputs['act'].shape[-1]
        scale = lambda x: -1 + (x)*(2)/(num_bins-1)
        idxs = torch.arange(num_bins, device=model_outputs['act'].device)
        act_probs = (model_outputs['act'] * idxs.view(1, -1)).sum(dim=-1)
        val_probs = (model_outputs['val'] * idxs.view(1, -1)).sum(dim=-1)
        model_outputs['act'] = scale(act_probs.round().clamp(0, num_bins-1))
        model_outputs['val'] = scale(val_probs.round().clamp(0, num_bins-1))

    act_ccc = ccc(model_outputs['act'], target_values['y_act'])
    val_ccc = ccc(model_outputs['val'], target_values['y_val'])

    if logger:
        logger.log_scalar(type_, 'Activation CCC', name, act_ccc, step)
        logger.log_scalar(type_, 'Valence CCC', name, val_ccc, step)

    return {
        'activation_ccc': act_ccc,
        'valence_ccc': val_ccc
    }

def calculate_per_annotator_metrics(
    model_outputs: Dict[str, torch.Tensor],
    target_values: Dict[str, torch.Tensor],
    type_: str,
    name: str,
    step: int,
    logger: Optional[CSVWriter] = None,
    return_individual_ccc: bool = False
) -> Dict[str, float] | Tuple[Dict[str, float], Dict[str, float]]:
    """Calculate per-annotator metrics.
    
    Args:
        model_outputs: Dictionary of model outputs
        target_values: Dictionary of target values
        type_: Type of evaluation (validation/test)
        name: Model name for logging
        step: Current step/epoch
        logger: Logger instance
        return_individual_ccc: Whether to return individual CCCs for each annotator
        
    Returns:
        Dictionary of metric values
    """
    metrics = {}
    annotators = list(set(target_values['ccc_ind_annotators']))
    
    act_cccs = []
    val_cccs = []
    individual_cccs = {}
    for annotator in annotators:
        pred_act = model_outputs[f'pred_act_{annotator}']
        pred_val = model_outputs[f'pred_val_{annotator}']
        if len(pred_act.shape) > 1 and pred_act.shape[-1] > 1:
            num_bins = pred_act.shape[-1]
            scale = lambda x: -1 + (x)*(2)/(num_bins-1)
            idxs = torch.arange(num_bins, device=pred_act.device)
            act_probs = (pred_act * idxs.view(1, -1)).sum(dim=-1)
            val_probs = (pred_val * idxs.view(1, -1)).sum(dim=-1)
            pred_act = scale(act_probs.round().clamp(0, num_bins-1))
            pred_val = scale(val_probs.round().clamp(0, num_bins-1))

        target_act = target_values[f'y_act_{annotator}']
        target_val = target_values[f'y_val_{annotator}']
        
        # Check if only 1 sample - CCC is undefined
        if len(pred_act) < 2:
            act_ccc = "N/A"
            val_ccc = "N/A"
        else:
            act_ccc = ccc(pred_act, target_act)
            val_ccc = ccc(pred_val, target_val)
            act_cccs.append(act_ccc)
            val_cccs.append(val_ccc)
        
        if return_individual_ccc:
            individual_cccs[annotator] = {
                'act_ccc': act_ccc,
                'val_ccc': val_ccc
            }

        if logger:
            logger.log_scalar(type_, f'Annotator {annotator} Activation CCC_ind', name, act_ccc, step)
            logger.log_scalar(type_, f'Annotator {annotator} Valence CCC_ind', name, val_ccc, step)
    
    mean_act_ccc = sum(act_cccs) / len(act_cccs) if act_cccs else "N/A"
    mean_val_ccc = sum(val_cccs) / len(val_cccs) if val_cccs else "N/A"
    
    if logger:
        logger.log_scalar(type_, 'Mean Activation CCC_ind', name, mean_act_ccc, step)
        logger.log_scalar(type_, 'Mean Valence CCC_ind', name, mean_val_ccc, step)
    
    metrics.update({
        'mean_annotator_activation_ccc': mean_act_ccc,
        'mean_annotator_valence_ccc': mean_val_ccc
    })
    
    if return_individual_ccc:
        return metrics, individual_cccs
    else:
        return metrics

def calculate_kde_metrics(
    model_outputs: Dict[str, torch.Tensor],
    target_values: Dict[str, torch.Tensor],
    type_: str,
    name: str,
    step: int,
    logger: Optional[CSVWriter] = None
) -> Dict[str, float]:
    """Calculate KDE metrics.
    
    Args:
        model_outputs: Dictionary of model outputs
        target_values: Dictionary of target values
        type_: Type of evaluation (validation/test)
        name: Model name for logging
        step: Current step/epoch
        logger: Logger instance
        
    Returns:
        Dictionary of metric values
    """
    # Check if KDE predictions are available (not None when skip_kde=True)
    prediction_key = 'probability_preds'# if 'probability_preds' in model_outputs else 'probability_logits'
    
    # Helper function to check if value is None or a list containing only None values
    def is_none_or_list_of_none(value):
        if value is None:
            return True
        if isinstance(value, list):
            return len(value) == 0 or all(v is None for v in value)
        return False
    
    if (prediction_key not in model_outputs or is_none_or_list_of_none(model_outputs[prediction_key]) or
        'y_kde_2d_probability' not in target_values or is_none_or_list_of_none(target_values['y_kde_2d_probability'])):
        # Return N/A values when KDE is disabled/None or targets are None/list of None
        if logger:
            logger.log_scalar(type_, 'Total Variation Distance', name, "N/A", step)
            logger.log_scalar(type_, 'Jensen-Shannon Divergence', name, "N/A", step)
            logger.log_scalar(type_, 'Total Variation Distance 1D Activation', name, "N/A", step)
            logger.log_scalar(type_, 'Total Variation Distance 1D Valence', name, "N/A", step)
            logger.log_scalar(type_, 'Jensen-Shannon Divergence 1D Activation', name, "N/A", step)
            logger.log_scalar(type_, 'Jensen-Shannon Divergence 1D Valence', name, "N/A", step)
            logger.log_scalar(type_, 'Argmax Activation UAR', name, "N/A", step)
            logger.log_scalar(type_, 'Argmax Valence UAR', name, "N/A", step)
        
        return {
            'Total Variation Distance': "N/A",
            'Jensen-Shannon Divergence': "N/A",
            'Total Variation Distance 1D Activation': "N/A",
            'Total Variation Distance 1D Valence': "N/A",
            'Jensen-Shannon Divergence 1D Activation': "N/A",
            'Jensen-Shannon Divergence 1D Valence': "N/A",
            'Argmax Activation UAR': "N/A",
            'Argmax Valence UAR': "N/A"
        }
    
    predictions = model_outputs[prediction_key]
    targets = target_values['y_kde_2d_probability']
    
    # Calculate 2D metrics
    total_var = total_variation(predictions, targets)
    jsd_val = JSD(predictions, targets)
    
    # Calculate 1D metrics by summing over dimensions
    act_dimension = predictions.sum(dim=-1)  # Sum over valence dimension
    val_dimension = predictions.sum(dim=-2)  # Sum over activation dimension
    target_act_dimension = targets.sum(dim=-1)
    target_val_dimension = targets.sum(dim=-2)
    
    total_var_1d_act = total_variation_1d(act_dimension, target_act_dimension)
    total_var_1d_val = total_variation_1d(val_dimension, target_val_dimension)
    jsd_1d_act = JSD_1d(act_dimension, target_act_dimension)
    jsd_1d_val = JSD_1d(val_dimension, target_val_dimension)
    
    # Recalculate sizes in the case of experiments where size might change 
    argmax_act = act_dimension.argmax(dim=-1)
    argmax_val = val_dimension.argmax(dim=-1)

    argmax_y_act = target_act_dimension.argmax(dim=-1)
    argmax_y_val = target_val_dimension.argmax(dim=-1)  

    argmax_act_uar = recall_score(argmax_y_act.cpu(), argmax_act.cpu(), average='macro')
    argmax_val_uar = recall_score(argmax_y_val.cpu(), argmax_val.cpu(), average='macro')    

    if logger:
        logger.log_scalar(type_, 'Total Variation Distance', name, total_var, step)
        logger.log_scalar(type_, 'Jensen-Shannon Divergence', name, jsd_val, step)
        logger.log_scalar(type_, 'Total Variation Distance 1D Activation', name, total_var_1d_act, step)
        logger.log_scalar(type_, 'Total Variation Distance 1D Valence', name, total_var_1d_val, step)
        logger.log_scalar(type_, 'Jensen-Shannon Divergence 1D Activation', name, jsd_1d_act, step)
        logger.log_scalar(type_, 'Jensen-Shannon Divergence 1D Valence', name, jsd_1d_val, step)
        logger.log_scalar(type_, 'Argmax Activation UAR', name, argmax_act_uar, step)
        logger.log_scalar(type_, 'Argmax Valence UAR', name, argmax_val_uar, step)
    
    return {
        'Total Variation Distance': total_var,
        'Jensen-Shannon Divergence': jsd_val,
        'Total Variation Distance 1D Activation': total_var_1d_act,
        'Total Variation Distance 1D Valence': total_var_1d_val,
        'Jensen-Shannon Divergence 1D Activation': jsd_1d_act,
        'Jensen-Shannon Divergence 1D Valence': jsd_1d_val,
        'Argmax Activation UAR': argmax_act_uar,
        'Argmax Valence UAR': argmax_val_uar
    }

def calculate_bmc_metrics(
    model_outputs: Dict[str, torch.Tensor],
    target_values: Dict[str, torch.Tensor],
    type_: str,
    name: str,
    step: int,
    logger: Optional[CSVWriter],
    bmc_calculator: Any
) -> Dict[str, float]:
    """Calculate BMC metrics.
    
    Args:
        model_outputs: Dictionary of model outputs
        target_values: Dictionary of target values
        type_: Type of evaluation (validation/test)
        name: Model name for logging
        step: Current step/epoch
        logger: Logger instance
        bmc_calculator: BMC calculator instance
        
    Returns:
        Dictionary of metric values
    """
    predictions = torch.cat(model_outputs['soft_act_preds'])
    targets = torch.cat(target_values['y_padded_individual_annotators_act'])
    
    metrics = {}
    bmc_calculator(predictions, targets, logger, type_, name, step)
    
    return metrics