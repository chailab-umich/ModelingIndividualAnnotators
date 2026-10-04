"""Evaluation module for emotion recognition models."""

from typing import Optional, List, Any, Tuple
from torch.utils.data import DataLoader
from MIA.config.experiment_config import ExperimentType
from .evaluator import ModelEvaluator, EvaluationResults

def validate_epoch(
    dataloader: DataLoader,
    model_trainer: Any,
    prob_grid_size: int,
    epoch: int,
    soft_hist: bool = True,
    pretrain: bool = False,
    test_name: str = 'Validation',
    fixed_annotators: Optional[List[str]] = None,
    skip_kde: bool = False,
    benchmarking: bool = False
) -> Tuple[EvaluationResults, bool]:
    """Run validation epoch.
    
    Args:
        dataloader: DataLoader for validation data
        model_trainer: Model trainer instance
        prob_grid_size: Size of probability grid for KDE
        epoch: Current epoch number
        soft_hist: Whether to use soft histograms
        pretrain: Whether in pretraining mode
        test_name: Name for logging
        fixed_annotators: List of annotator IDs to evaluate on
        skip_kde: Whether to skip KDE evaluation
        
    Returns:
        Tuple containing:
            - EvaluationResults containing model outputs, targets and metrics
            - Boolean indicating whether training should continue
    """
    evaluator = ModelEvaluator(model_trainer, prob_grid_size, model_trainer.logger)
    results = evaluator.evaluate(
        dataloader,
        type_="validation",
        step=epoch,
        fixed_annotators=fixed_annotators,
        skip_kde=skip_kde,
        check_for_aggregate=True,
        soft_hist=soft_hist,
        pretrain=pretrain,
        benchmarking=benchmarking
    )
    
    # Handle early stopping based on validation loss (but not during pretraining)
    if results.validation_loss is not None and not pretrain:
        model_trainer.continue_training = model_trainer.early_stopping.log_value(
            results.validation_loss, 
            model_trainer.model
        )
        
        # Update learning rate
        model_trainer.lr_scheduler.step(results.validation_loss)
        new_lr = model_trainer.lr_scheduler.get_last_lr()
        if new_lr != model_trainer.current_lr:
            print(f'Learning rate reduced from {model_trainer.current_lr} to {new_lr}')
            model_trainer.current_lr = new_lr
            
        if not model_trainer.continue_training:
            print(f'{model_trainer.name} triggered early stopping')
    
    return results, model_trainer.continue_training

def test_epoch(
    dataloader: DataLoader,
    model_trainer: Any,
    prob_grid_size: int,
    bmc_calculator: Any = None,
    soft_hist: bool = True,
    test_name: str = 'Test',
    fixed_annotators: Optional[List[str]] = None,
    check_for_aggregate: bool = True,
    skip_kde: bool = False,
    csv_path_for_stored_evaluations: str = None,
    seed: int = None,
    fold: int = None,
    benchmarking: bool = False
) -> EvaluationResults:
    """Run test epoch.
    
    Args:
        dataloader: DataLoader for test data
        model_trainer: Model trainer instance
        prob_grid_size: Size of probability grid for KDE
        bmc_calculator: BMC calculator instance
        soft_hist: Whether to use soft histograms
        test_name: Name for logging
        fixed_annotators: List of annotator IDs to evaluate on
        check_for_aggregate: Whether to check for aggregate predictions
        skip_kde: Whether to skip KDE evaluation
        
    Returns:
        EvaluationResults containing model outputs, targets and metrics
    """
    evaluator = ModelEvaluator(
        model_trainer,
        prob_grid_size,
        model_trainer.logger,
        bmc_calculator=bmc_calculator
    )
    if csv_path_for_stored_evaluations is not None:
        evaluator.store_predictions(
            dataloader,
            type_=test_name,
            step=0,
            fixed_annotators=fixed_annotators,
            skip_kde=skip_kde,
            check_for_aggregate=check_for_aggregate,
            soft_hist=soft_hist,
            pretrain=False,
            csv_path=csv_path_for_stored_evaluations,
            seed=seed,
            fold=fold
        )
    return evaluator.evaluate(
        dataloader,
        type_=test_name,
        step=0,
        fixed_annotators=fixed_annotators,
        skip_kde=skip_kde,
        check_for_aggregate=check_for_aggregate,
        soft_hist=soft_hist,
        pretrain=False,
        benchmarking=benchmarking
    )