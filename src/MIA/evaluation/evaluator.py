"""Evaluation loops for emotion recognition models."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from datetime import datetime
import numpy as np
from MIA.config import ModelType
from MIA.csv_logging import CSVWriter
from MIA.loss_metrics import BMC, PreTrainLoss
from .utils import concatenate_outputs, process_outputs, process_targets, process_masks_and_full_padded_values
from .metrics import (
    calculate_ccc_metrics,
    calculate_per_annotator_metrics,
    calculate_kde_metrics,
    calculate_bmc_metrics
)
import time

@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    model_outputs: Dict[str, torch.Tensor]
    target_values: Dict[str, torch.Tensor]
    metrics: Dict[str, float]
    validation_loss: Optional[float] = None

class ModelEvaluator:
    """Evaluator for emotion recognition models."""
    def __init__(
        self,
        model_trainer,
        prob_grid_size: int,
        logger: CSVWriter,
        bmc_calculator: Optional[BMC] = None
    ):
        """Initialize evaluator.
        
        Args:
            model_trainer: Model trainer instance
            prob_grid_size: Size of probability grid for KDE
            logger: Logger instance for logging metrics
            bmc_calculator: BMC calculator instance
        """
        self.model_trainer = model_trainer
        self.prob_grid_size = prob_grid_size
        self.logger = logger
        self.bmc_calculator = bmc_calculator

    def infer_on_dataloader(self, dataloader: DataLoader, type_: str, skip_kde: bool, soft_hist: bool, check_for_aggregate: bool, fixed_annotators: Optional[List[str]]) -> Dict[str, torch.Tensor]:
        self.model_trainer.eval()
        model_outputs = {}
        target_values = {}
        # Construct a weight mask and output mask that would theoretically match the output of the model in a real forward call
        masks_and_full_padded_values = {'batch_mask': [], 'weight_mask': [], 'padded_y_act': [], 'padded_y_val': [], 'padded_act_preds': [], 'padded_val_preds': []}
        model_type_check = self.model_trainer.model.args.model_type in [ # Check if model should be making individual annotator predictions
            ModelType.INDIVIDUAL_ANNOTATOR, 
            ModelType.INDIVIDUAL_ANNOTATOR_BETA, 
            ModelType.INDIVIDUAL_ANNOTATOR_PLUS,
            ModelType.INDIVIDUAL_ANNOTATOR_CONDOR,
            ModelType.INDIVIDUAL_ANNOTATOR_ORTHOGONAL,
        ] and not self.model_trainer.model.pre_training # Models don't make individual annotator predictions during pretraining

        with torch.no_grad():
            for i, batch in tqdm(enumerate(dataloader), desc=f"Evaluating {type_}", total=len(dataloader)):
                # Get model outputs
                outputs = self._get_model_outputs(
                    batch,
                    skip_kde,
                    soft_hist,
                    fixed_annotators
                )
                # print('Evaluation shapes:', outputs['soft_act_preds'].shape, outputs['soft_val_preds'].shape)
                # Process outputs
                process_outputs(outputs, model_outputs)
                
                # Process targets
                process_targets(batch, target_values)

                if model_type_check:
                    process_masks_and_full_padded_values(batch, outputs, masks_and_full_padded_values, i)

                # Process special cases (aggregate, per-annotator, etc.)
                self._process_special_cases(
                    batch,
                    outputs,
                    model_outputs,
                    target_values,
                    fixed_annotators,
                    check_for_aggregate
                )

        # Concatenate lists to tensors for metrics and loss computation
        concatenate_outputs(model_outputs, target_values)

        if model_type_check:
            del masks_and_full_padded_values['largest_seen_batch']
            for key in masks_and_full_padded_values:
                # print(key, 'has items with shapes:', [x.shape for x in masks_and_full_padded_values[key]])
                masks_and_full_padded_values[key] = torch.cat(masks_and_full_padded_values[key], dim=0)
                # print(f'{key}: {masks_and_full_padded_values[key].shape}')
            masks_and_full_padded_values['output_mask'] = ~masks_and_full_padded_values['padded_y_act'].isnan()
            # print(f'output_mask: {masks_and_full_padded_values["output_mask"].shape}')
            # print(f'Comparison with old method: {model_outputs["soft_act_preds"].shape=} {model_outputs["soft_val_preds"].shape=} {model_outputs["soft_act_preds"].isnan().sum()=} {model_outputs["soft_val_preds"].isnan().sum()=}')
            # print(f'Comparison with new method: {masks_and_full_padded_values["padded_act_preds"].shape=} {masks_and_full_padded_values["padded_val_preds"].shape=} {masks_and_full_padded_values["padded_act_preds"].isnan().sum()=} {masks_and_full_padded_values["padded_val_preds"].isnan().sum()=}')
            # print(f'Comparison with old method: {target_values["y_padded_individual_annotators_act"].shape=} {target_values["y_padded_individual_annotators_val"].shape=} {target_values["y_padded_individual_annotators_act"].isnan().sum()=} {target_values["y_padded_individual_annotators_val"].isnan().sum()=}')
            # print(f'Comparison with new method: {masks_and_full_padded_values["padded_y_act"].shape=} {masks_and_full_padded_values["padded_y_val"].shape=} {masks_and_full_padded_values["padded_y_act"].isnan().sum()=} {masks_and_full_padded_values["padded_y_val"].isnan().sum()=}')
            model_outputs['soft_act_preds'] = masks_and_full_padded_values['padded_act_preds']
            model_outputs['soft_val_preds'] = masks_and_full_padded_values['padded_val_preds']
            target_values['y_padded_individual_annotators_act'] = masks_and_full_padded_values['padded_y_act']
            target_values['y_padded_individual_annotators_val'] = masks_and_full_padded_values['padded_y_val'] #TODO: Remove earlier calculation in process_outputs since this is overwriting those calculations
            target_values['mask'] = (masks_and_full_padded_values['batch_mask'], masks_and_full_padded_values['weight_mask'], masks_and_full_padded_values['output_mask'], None)
        return model_outputs, target_values

    def evaluate(
        self,
        dataloader: DataLoader,
        type_: str = "validation",
        step: int = 0,
        fixed_annotators: Optional[List[str]] = None,
        skip_kde: bool = False,
        check_for_aggregate: bool = True,
        soft_hist: bool = True,
        pretrain: bool = False,
        benchmarking: bool = False
    ) -> EvaluationResults:
        """Run evaluation loop.
        
        Args:
            dataloader: DataLoader for evaluation data
            type_: Type of evaluation (validation/test)
            step: Current step/epoch
            fixed_annotators: List of annotator IDs to evaluate on
            skip_kde: Whether to skip KDE evaluation
            check_for_aggregate: Whether to check for aggregate predictions
            soft_hist: Whether to use soft histograms
            pretrain: Whether in pretraining mode
            
        Returns:
            EvaluationResults containing model outputs, targets and metrics
        """
        self.model_trainer.eval()
        
        # Optimization: Skip KDE during validation if no training loss functions require it
        # KDE is expensive and only needed for baseline/task3 losses, not for individual annotator tasks
        if type_ == "validation" and not skip_kde:
            loss_fns_to_check = [PreTrainLoss(name='CCC loss', sparsity=None, calculate_kde=False, 
                                            after_warmup=False, loss_fn='CCC', train_log_var=False)] if fixed_annotators=='USE-ALL' else self.model_trainer.loss_fns
            kde_needed_for_validation = any(getattr(loss_fn, 'calculate_kde', False) for loss_fn in loss_fns_to_check)
            if not kde_needed_for_validation:
                skip_kde = True
                print(f"[OPTIMIZATION] Skipping KDE during validation - not needed for current loss functions")
        benchmark_stats = {}
        if benchmarking:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            print('[PROFILING]: Memory reset and cache cleared before evaluation')
            benchmark_stats['start_time_evaluation'] = time.time()
            benchmark_stats['start_memory_evaluation'] = torch.cuda.memory_allocated()
            benchmark_stats['start_memory_peak_evaluation'] = torch.cuda.max_memory_allocated()
        model_outputs, target_values = self.infer_on_dataloader(dataloader, type_, skip_kde, soft_hist, check_for_aggregate, fixed_annotators)
        if benchmarking:
            benchmark_stats['end_time_evaluation'] = time.time()
            benchmark_stats['end_memory_evaluation'] = torch.cuda.memory_allocated()
            benchmark_stats['end_memory_peak_evaluation'] = torch.cuda.max_memory_allocated()
            benchmark_stats['evaluation_time'] = benchmark_stats['end_time_evaluation'] - benchmark_stats['start_time_evaluation']
            benchmark_stats['evaluation_memory'] = benchmark_stats['end_memory_evaluation'] - benchmark_stats['start_memory_evaluation']
            benchmark_stats['evaluation_memory_peak'] = benchmark_stats['end_memory_peak_evaluation'] - benchmark_stats['start_memory_peak_evaluation']
            print(f'[PROFILING]: Before Evaluation {type_} stats: {benchmark_stats["start_time_evaluation"]} seconds, {benchmark_stats["start_memory_evaluation"]} bytes usage, {benchmark_stats["start_memory_peak_evaluation"]} bytes peak')
            print('--------------------------------')
            print(f'[PROFILING]: After Evaluation {type_} stats: {benchmark_stats["end_time_evaluation"]} seconds, {benchmark_stats["end_memory_evaluation"]} bytes usage, {benchmark_stats["end_memory_peak_evaluation"]} bytes peak')
            print('--------------------------------')
            print(f'[PROFILING]: Evaluation {type_} time: {benchmark_stats["evaluation_time"]} seconds')
            print(f'[PROFILING]: Evaluation {type_} memory: {benchmark_stats["evaluation_memory"]} bytes')
            print(f'[PROFILING]: Evaluation {type_} memory peak: {benchmark_stats["evaluation_memory_peak"]} bytes')

        if self.model_trainer.model.args.flip_negative_values_in_validation:
            _, individual_cccs = calculate_per_annotator_metrics(
                model_outputs,
                target_values,
                type_,
                self.model_trainer.name,
                step,
                None,
                return_individual_ccc=True
            )
            with torch.no_grad():
                temp_annotators = []
                for annotator, individual_ccc in individual_cccs.items():
                    annotator_idx = target_values['annotator_id_mapper'][annotator]
                    if individual_ccc['val_ccc'] != "N/A" and individual_ccc['val_ccc'] < 0:
                        print(f'Flipping valence weights for annotator {annotator} due to CCC: {individual_ccc["val_ccc"]}')
                        temp_annotators.append(annotator)
                        # valence CCC is negative, we can flip the model weights to immediately align this annotator's predictions
                        if self.model_trainer.model.args.model_type == ModelType.INDIVIDUAL_ANNOTATOR_ORTHOGONAL:
                            self.model_trainer.model.prediction_head.val_heads.embedding.weight[annotator_idx, :] = -self.model_trainer.model.prediction_head.val_heads.embedding.weight[annotator_idx, :]
                            self.model_trainer.model.prediction_head.annotator_val_biases[annotator_idx] = -self.model_trainer.model.prediction_head.annotator_val_biases[annotator_idx]
                            self.model_trainer.model.prediction_head.annotator_consensus_multiplier_val[annotator_idx] = -self.model_trainer.model.prediction_head.annotator_consensus_multiplier_val[annotator_idx]
                            self.model_trainer.model.prediction_head.annotator_consensus_multiplier_act[annotator_idx] = -self.model_trainer.model.prediction_head.annotator_consensus_multiplier_act[annotator_idx]
                        else:
                            self.model_trainer.model.prediction_head.val_heads.weight[annotator_idx, :] = -self.model_trainer.model.prediction_head.val_heads.weight[annotator_idx, :]
                            self.model_trainer.model.prediction_head.val_heads.bias[annotator_idx] = -self.model_trainer.model.prediction_head.val_heads.bias[annotator_idx]
                    if individual_ccc['act_ccc'] != "N/A" and individual_ccc['act_ccc'] < 0:
                        print(f'Flipping activation weights for annotator {annotator} due to CCC: {individual_ccc["act_ccc"]}')
                        temp_annotators.append(annotator)
                        # activation CCC is negative, we can flip the model weights to immediately align this annotator's predictions
                        if self.model_trainer.model.args.model_type == ModelType.INDIVIDUAL_ANNOTATOR_ORTHOGONAL:
                            self.model_trainer.model.prediction_head.act_heads.embedding.weight[annotator_idx, :] = -self.model_trainer.model.prediction_head.act_heads.embedding.weight[annotator_idx, :]
                            self.model_trainer.model.prediction_head.annotator_act_biases[annotator_idx] = -self.model_trainer.model.prediction_head.annotator_act_biases[annotator_idx]
                        else:
                            self.model_trainer.model.prediction_head.act_heads.weight[annotator_idx, :] = -self.model_trainer.model.prediction_head.act_heads.weight[annotator_idx, :]
                            self.model_trainer.model.prediction_head.act_heads.bias[annotator_idx] = -self.model_trainer.model.prediction_head.act_heads.bias[annotator_idx]
                temp_annotators = list(set(temp_annotators))
                if len(temp_annotators):
                    mean_act_ccc = torch.stack([individual_cccs[annotator]["act_ccc"] for annotator in temp_annotators if individual_cccs[annotator]["act_ccc"] != "N/A"]).nanmean()
                    mean_val_ccc = torch.stack([individual_cccs[annotator]["val_ccc"] for annotator in temp_annotators if individual_cccs[annotator]["val_ccc"] != "N/A"]).nanmean()
                    print(f'Average activation ccc before flipping weights: {mean_act_ccc.item()}')
                    print(f'Average valence ccc before flipping weights: {mean_val_ccc.item()}')
                    print('Recalculating model outputs after flipping weights...')
                    model_outputs, target_values = self.infer_on_dataloader(dataloader, type_, skip_kde, soft_hist, check_for_aggregate, fixed_annotators)
                    _, new_individual_cccs = calculate_per_annotator_metrics(
                        model_outputs,
                        target_values,
                        type_,
                        self.model_trainer.name,
                        step,
                        None,
                        return_individual_ccc=True
                    )
                    for annotator in temp_annotators:
                        print(f'After flipping weights for annotator {annotator}, valence CCC was {individual_cccs[annotator]["val_ccc"]} and is now {new_individual_cccs[annotator]["val_ccc"]}')
                        print(f'After flipping weights for annotator {annotator}, activation CCC was {individual_cccs[annotator]["act_ccc"]} and is now {new_individual_cccs[annotator]["act_ccc"]}')
                    mean_act_ccc = torch.stack([new_individual_cccs[annotator]["act_ccc"] for annotator in temp_annotators if new_individual_cccs[annotator]["act_ccc"] != "N/A"]).nanmean()
                    mean_val_ccc = torch.stack([new_individual_cccs[annotator]["val_ccc"] for annotator in temp_annotators if new_individual_cccs[annotator]["val_ccc"] != "N/A"]).nanmean()
                    print(f'Average activation ccc after flipping weights: {mean_act_ccc.item()}')
                    print(f'Average valence ccc after flipping weights: {mean_val_ccc.item()}')
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            model_outputs,
            target_values,
            type_,
            step
        )
        
        # Calculate validation loss over entire dataset
        validation_loss = None
        temp_validation_loss = self._calculate_validation_loss(
                model_outputs,
                target_values,
                type_,
                pretrain=pretrain,
                fixed_annotators=fixed_annotators,
                step=step
            )
        if type_ == "validation":
            validation_loss = temp_validation_loss
        
        # Restore model to train mode after evaluation
        self.model_trainer.train()
        
        return EvaluationResults(
            model_outputs=model_outputs,
            target_values=target_values,
            metrics=metrics,
            validation_loss=validation_loss
        )

    def store_predictions(
        self,
        dataloader: DataLoader,
        type_: str = "validation",
        step: int = 0,
        fixed_annotators: Optional[List[str]] = None,
        skip_kde: bool = False,
        check_for_aggregate: bool = True,
        soft_hist: bool = True,
        pretrain: bool = False,
        csv_path: str = None,
        seed: int = None,
        fold: int = None
    ) -> None:
        """Run evaluation loop.
        
        Args:
            dataloader: DataLoader for evaluation data
            type_: Type of evaluation (validation/test)
            step: Current step/epoch
            fixed_annotators: List of annotator IDs to evaluate on
            skip_kde: Whether to skip KDE evaluation
            check_for_aggregate: Whether to check for aggregate predictions
            soft_hist: Whether to use soft histograms
            pretrain: Whether in pretraining mode
            
        Returns:
            EvaluationResults containing model outputs, targets and metrics
        """
        self.model_trainer.eval()
        options = ['_'.join(f.split('_')[-4:]) for f in os.listdir(csv_path)]
        date_times = [datetime.strptime(dt, '%d_%m_%Y_%H:%M:%S') for dt in options]
        print(f'{csv_path=} {options=} {date_times=}')
        most_recent_date_time = np.argmax(date_times)
        full_path = os.path.join(csv_path, f'{csv_path.split("/")[-1]}_{options[most_recent_date_time]}')
        long_results_df = {'FileName': [], 'annotator_id': [], 'act': [], 'val': [], 'act_pred': [], 'val_pred': [], 'act_probability_logits': [], 'val_probability_logits': []}
        consensus_results_df = {'FileName': [], 'act': [], 'val': [], 'act_pred': [], 'val_pred': []}
        has_logits = self.model_trainer.model.args.model_type == ModelType.INDIVIDUAL_ANNOTATOR_CONDOR
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Storing predictions for {type_}"):
                # Get model outputs
                outputs = self._get_model_outputs(
                    batch,
                    skip_kde,
                    soft_hist,
                    fixed_annotators,
                    remove_soft_preds_when_using_all=False
                )
                # Store long format results
                for sample, filename in enumerate(batch['FileName']):
                    annotators = batch['annotators'][sample]
                    consensus_results_df['FileName'].append(filename)
                    consensus_results_df['act'].append(batch['act'][sample].cpu().item())
                    consensus_results_df['val'].append(batch['val'][sample].cpu().item())
                    consensus_results_df['act_pred'].append(outputs['mean_act_preds'][sample].cpu().item())
                    consensus_results_df['val_pred'].append(outputs['mean_val_preds'][sample].cpu().item())
                    for j, annotator in enumerate(annotators):
                        long_results_df['FileName'].append(filename)
                        long_results_df['annotator_id'].append(annotator)
                        long_results_df['act'].append(batch['soft_act_labels'][sample][j].cpu().item())
                        long_results_df['val'].append(batch['soft_val_labels'][sample][j].cpu().item())
                        # Handle per-annotator predictions based on model type
                        if 'soft_act_preds' in outputs and 'soft_val_preds' in outputs:
                            long_results_df['act_pred'].append(outputs['soft_act_preds'][sample][j].cpu().item())
                            long_results_df['val_pred'].append(outputs['soft_val_preds'][sample][j].cpu().item())
                        elif self.model_trainer.model.args.model_type == ModelType.BASELINE_AGGREGATE:
                            # Aggregate models don't produce per-annotator predictions, use mean for all
                            long_results_df['act_pred'].append(outputs['mean_act_preds'][sample].cpu().item())
                            long_results_df['val_pred'].append(outputs['mean_val_preds'][sample].cpu().item())
                        else:
                            raise KeyError(
                                f"Model type {self.model_trainer.model.args.model_type} should produce "
                                f"'soft_act_preds' and 'soft_val_preds' but they are missing from outputs. "
                                f"Available keys: {list(outputs.keys())}"
                            )
                        if has_logits:
                            long_results_df['act_probability_logits'].append(outputs['act_probability_logits'][sample][j].cpu().numpy())
                            long_results_df['val_probability_logits'].append(outputs['val_probability_logits'][sample][j].cpu().numpy())
        import duckdb
        import pandas as pd
        con = duckdb.connect(f'{full_path}/fold_{fold}_seed_{seed}_test_{type_}.duckdb'"preds.duckdb")
        # Create annotator predictions table
        ratings_df = pd.DataFrame({
            "file_name": long_results_df['FileName'],
            "annotator_id": long_results_df['annotator_id'],
            "act": long_results_df['act'], "val": long_results_df['val'],
            "act_pred": long_results_df['act_pred'], "val_pred": long_results_df['val_pred'],
        })
        con.execute("CREATE OR REPLACE TABLE ratings AS SELECT * FROM ratings_df")

        # Create annotator logits table
        if has_logits:
            logits_df = pd.DataFrame({
                "file_name": long_results_df['FileName'],
                "annotator_id": long_results_df['annotator_id'],
                "act_logits": [x.tolist() for x in long_results_df['act_probability_logits']],
                "val_logits": [x.tolist() for x in long_results_df['val_probability_logits']],
            })
            con.execute("CREATE OR REPLACE TABLE logits AS SELECT * FROM logits_df")

        # Create consensus table
        consensus_df = pd.DataFrame({
            "file_name": consensus_results_df['FileName'],
            "act": consensus_results_df['act'], "val": consensus_results_df['val'],
            "act_pred": consensus_results_df['act_pred'], "val_pred": consensus_results_df['val_pred'],
        })
        con.execute("CREATE OR REPLACE TABLE consensus AS SELECT * FROM consensus_df")

        con.execute("CREATE INDEX IF NOT EXISTS idx_ratings_annot ON ratings(annotator_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_ratings_item ON ratings(file_name)")
        if has_logits:
            con.execute("CREATE INDEX IF NOT EXISTS idx_logits_annot  ON logits(annotator_id)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_logits_item   ON logits(file_name)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_consensus_item   ON consensus(file_name)")

    def _calculate_validation_loss(
        self,
        model_outputs: Dict[str, torch.Tensor],
        target_values: Dict[str, torch.Tensor],
        type_: str,
        pretrain: bool,
        fixed_annotators: Optional[List[str]],
        step: int
    ) -> Optional[float]:
        """Calculate validation loss over entire dataset."""
        total_val_loss = 0.0
        loss_err = False
        
        if pretrain:
            loss_fn = self.model_trainer.pretrain_loss
            losses = loss_fn(model_outputs, target_values, masks=None)
            for loss_name, loss_val in losses.items():
                if loss_val is None:
                    loss_err = True
                    continue
                total_val_loss += loss_val.item()
                if self.logger:
                    self.logger.log_scalar(f'{type_} loss ({loss_name})', 
                                         f'{loss_fn.name}-{loss_name}', 
                                         self.model_trainer.name, 
                                         loss_val.item(), 
                                         step=step)
        else:
            use_kl_div = len(self.model_trainer.loss_fns) == 1
            loss_fns = [PreTrainLoss(name='CCC loss', sparsity=None, calculate_kde=False, 
                                   after_warmup=False, loss_fn='CCC', train_log_var=False)] if fixed_annotators=='USE-ALL' else self.model_trainer.loss_fns
            
            for loss_fn in loss_fns:
                if (loss_fn.name == 'task1' or loss_fn.name == 'task1_mse' or loss_fn.name == 'task1cccind') and fixed_annotators is not None:
                    continue
                
                if 'act_log_vars' in loss_fn.required_model_outputs:
                    soft_act_zs = model_outputs['soft_act_zs'] if 'soft_act_zs' in model_outputs else None
                    soft_val_zs = model_outputs['soft_val_zs'] if 'soft_val_zs' in model_outputs else None
                    model_outputs['act_z_log_var'] = (soft_act_zs, model_outputs['act_log_vars'])
                    model_outputs['val_z_log_var'] = (soft_val_zs, model_outputs['val_log_vars'])
                    
                if ('y_padded_individual_annotators_act' in loss_fn.required_target_labels and 
                    'y_padded_individual_annotators_act_only_seen_annotators' in target_values):
                    target_values['y_padded_individual_annotators_act'] = target_values['y_padded_individual_annotators_act_only_seen_annotators']
                    target_values['y_padded_individual_annotators_val'] = target_values['y_padded_individual_annotators_val_only_seen_annotators']
                if loss_fn.name == 'task1CORAL5' or loss_fn.name == 'task1CORAL9':
                    target_values['CORALProcesses'] = self.model_trainer.model.coral_processes
                if loss_fn.name == 'task1cccind':
                    losses = calculate_per_annotator_metrics(
                        model_outputs,
                        target_values,
                        type_,
                        self.model_trainer.name,
                        step,
                        self.logger
                    )
                    for l in losses:
                        losses[l] = 1 - losses[l]
                else:
                    # Aggregate/KDE models do not produce per-annotator masks.
                    # Their training path passes None in the same situation;
                    # evaluation should do likewise after calculating metrics.
                    losses = loss_fn(
                        model_outputs,
                        target_values,
                        masks=target_values.get('mask'),
                    )
                for loss_name, loss_val in losses.items():
                    if loss_val is None:
                        loss_err = True
                        continue
                    if ('KL-Div' not in loss_name or use_kl_div):
                        if type_ == 'validation':
                            print(f'[Validation loss breakdown]: {loss_name}={loss_val.item():.4f}')
                        if 'orthogonal' not in loss_name.lower() and 'decov' not in loss_name.lower() and 'importance' not in loss_name.lower() and 'load-balancing' not in loss_name.lower():
                            total_val_loss += loss_val.item()
                    if self.logger:
                        self.logger.log_scalar(f'{type_} loss ({loss_name})', 
                                             f'{loss_fn.name}-{loss_name}', 
                                             self.model_trainer.name, 
                                             loss_val.item(), 
                                             step=step)
        
        if loss_err:
            total_val_loss += 99
            
        return total_val_loss
    
    def _get_model_outputs(
        self,
        batch: Dict[str, Any],
        skip_kde: bool,
        soft_hist: bool,
        fixed_annotators: Optional[List[str]],
        remove_soft_preds_when_using_all: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Get model outputs for a batch.
        
        For mean predictions:
        - Aggregate ground truth model: Use batch 'act'/'val' directly
        - Individual annotator models: Get from model outputs
        - USE-ALL mode: Calculate mean from soft predictions
        """
        if self.model_trainer.model.args.model_type == ModelType.AGGREGATE_GROUND_TRUTH:
            return {
                'act': batch['act'],
                'val': batch['val'],
                'mean_act_preds': batch['act'],
                'mean_val_preds': batch['val']
            }
            
        # For zero-shot all-annotators, set annotator_masks=None
        if fixed_annotators == 'USE-ALL':
            annotator_masks = None
        else:
            annotator_masks = batch.get('annotator_masks')
        outputs = self.model_trainer.model(
            batch['audio'],
            batch['transcript'],
            skip_kde=skip_kde,
            soft_hist=soft_hist,
            annotator_masks=annotator_masks
        )
        
        if fixed_annotators == 'USE-ALL':
            mean_act = outputs['soft_act_preds'].nanmean(dim=-1)
            mean_val = outputs['soft_val_preds'].nanmean(dim=-1)
            if remove_soft_preds_when_using_all:
                del outputs['soft_act_preds'], outputs['soft_val_preds']
            outputs.update({
                'act': mean_act,
                'val': mean_val,
                'mean_act_preds': mean_act,
                'mean_val_preds': mean_val
            })
        
        # Ensure mean predictions are present for baseline/aggregate models
        # If the model outputs 'act' and 'val' but not 'mean_act_preds'/'mean_val_preds',
        # use 'act'/'val' as the mean predictions
        if 'act' in outputs and 'val' in outputs:
            if 'mean_act_preds' not in outputs:
                outputs['mean_act_preds'] = outputs['act']
            if 'mean_val_preds' not in outputs:
                outputs['mean_val_preds'] = outputs['val']
            
        return outputs

    def _process_special_cases(
        self,
        batch: Dict[str, Any],
        outputs: Dict[str, torch.Tensor],
        model_outputs: Dict[str, torch.Tensor],
        target_values: Dict[str, torch.Tensor],
        fixed_annotators: Optional[List[str]],
        check_for_aggregate: bool
    ):
        """Process special cases like aggregate predictions and per-annotator metrics.
        
        ZERO-SHOT USE-ALL EVALUATION:
        When fixed_annotators='USE-ALL' during zero-shot evaluation, this performs a special
        evaluation mode where:
        1. The pretrained model makes aggregate predictions using ALL of its own trained annotators
        2. These aggregate predictions are compared against the ground truth from the new dataset
        3. Individual annotators from the new dataset that don't exist in the model are handled
           gracefully by using the aggregate prediction
        
        This tests how well a model trained on one dataset (e.g., MSP-Podcast) generalizes
        to predict emotional labels on a completely different dataset (e.g., IEMOCAP) without
        any annotator mapping or finetuning.
        """
        
        # Initialize per-annotator tracking if needed
        if 'ccc_ind_annotators' not in target_values:
            target_values['ccc_ind_annotators'] = []
            target_values['annotator_id_mapper'] = {}

        # Handle aggregate predictions for individual_annotator_plus model
        if (self.model_trainer.model.args.model_type == ModelType.INDIVIDUAL_ANNOTATOR_PLUS 
            and not self.model_trainer.model.pre_training):
            
            agg_idx = self.model_trainer.model.prediction_head.act_heads.annotator_mapper.map_to_idx['aggregate-annotator']
            
            if (fixed_annotators is None or agg_idx in fixed_annotators) and check_for_aggregate:
                # Initialize aggregate output lists if they don't exist
                if 'aggregate_act' not in model_outputs:
                    model_outputs['aggregate_act'] = []
                if 'aggregate_val' not in model_outputs:
                    model_outputs['aggregate_val'] = []
                    
                id_mask = batch['annotator_masks'][1]
                act_preds = outputs['soft_act_preds']
                val_preds = outputs['soft_val_preds']
                
                agg_act = act_preds[~act_preds.isnan()][id_mask == agg_idx]
                agg_val = val_preds[~val_preds.isnan()][id_mask == agg_idx]
                
                model_outputs['aggregate_act'].append(agg_act.cpu())
                model_outputs['aggregate_val'].append(agg_val.cpu())

        # Process per-annotator predictions and targets
        pre_training_check = not hasattr(self.model_trainer.model, 'pre_training') or not self.model_trainer.model.pre_training
        model_type_check = self.model_trainer.model.args.model_type in [
            ModelType.INDIVIDUAL_ANNOTATOR, 
            ModelType.INDIVIDUAL_ANNOTATOR_BETA, 
            ModelType.INDIVIDUAL_ANNOTATOR_PLUS,
            ModelType.BASELINE_AGGREGATE,
            ModelType.KDE_2D,
            ModelType.AGGREGATE_GROUND_TRUTH,
            ModelType.INDIVIDUAL_ANNOTATOR_CONDOR,
            ModelType.INDIVIDUAL_ANNOTATOR_ORTHOGONAL,
        ]
        
        if pre_training_check and model_type_check:
            has_annotator_mapper = (hasattr(self.model_trainer.model, 'prediction_head') and 
                                  hasattr(self.model_trainer.model.prediction_head, 'act_heads'))
            
            for sample_i, annotators in enumerate(batch['annotators']):
                # Get target values for this sample
                sample_y_act = batch['soft_act_labels'][sample_i]
                sample_y_act = sample_y_act[~sample_y_act.isnan()]
                sample_y_val = batch['soft_val_labels'][sample_i]
                sample_y_val = sample_y_val[~sample_y_val.isnan()]
                
                # Get predictions for this sample
                is_individual_model = self.model_trainer.model.args.model_type in [
                    ModelType.INDIVIDUAL_ANNOTATOR,
                    ModelType.INDIVIDUAL_ANNOTATOR_BETA,
                    ModelType.INDIVIDUAL_ANNOTATOR_PLUS,
                    ModelType.INDIVIDUAL_ANNOTATOR_CONDOR,
                    ModelType.INDIVIDUAL_ANNOTATOR_ORTHOGONAL,
                ]
                has_soft_preds = 'soft_act_preds' in outputs

                if self.model_trainer.model.args.model_type == ModelType.KDE_2D:
                    # Calculate act and val from probability predictions
                    act_dimension = outputs['probability_preds'].sum(dim=-1)
                    val_dimension = outputs['probability_preds'].sum(dim=-2)
                    probability_bins = torch.linspace(-1,1,steps=act_dimension.shape[1], device=act_dimension.device)
                    kde_acts = torch.matmul(act_dimension, probability_bins)
                    probability_bins = torch.linspace(-1,1,steps=val_dimension.shape[1], device=val_dimension.device)
                    kde_vals = torch.matmul(val_dimension, probability_bins)

                if is_individual_model and has_soft_preds:
                    sample_pred_act = outputs['soft_act_preds'][sample_i]
                    prob_shape = None
                    if len(sample_pred_act.shape) > 1:
                        prob_shape = sample_pred_act.shape[-1]
                    sample_pred_act = sample_pred_act[~sample_pred_act.isnan()]
                    sample_pred_val = outputs['soft_val_preds'][sample_i]
                    sample_pred_val = sample_pred_val[~sample_pred_val.isnan()]
                    if prob_shape is not None:
                        sample_pred_act = sample_pred_act.view(-1, prob_shape)
                        sample_pred_val = sample_pred_val.view(-1, prob_shape)
                else:
                    # Model is baseline_aggregate, aggregate_ground_truth, OR individual model in USE-ALL mode
                    # Repeat aggregate prediction for each annotator
                    sample_pred_act = torch.empty(sample_y_act.shape, device=sample_y_act.device)
                    sample_pred_val = torch.empty(sample_y_val.shape, device=sample_y_val.device)
                    if self.model_trainer.model.args.model_type == ModelType.KDE_2D:
                        # Calculate act and val from probability predictions
                        sample_pred_act[:] = kde_acts[sample_i]
                        sample_pred_val[:] = kde_vals[sample_i]
                    else:
                        assert len(outputs['act'][sample_i].shape) == 0 or len(outputs['act'][sample_i]) == 1
                        sample_pred_act[:] = outputs['act'][sample_i]
                        sample_pred_val[:] = outputs['val'][sample_i]
                
                # Verify shapes match
                assert len(annotators) == len(sample_y_act)
                assert len(annotators) == len(sample_y_val)
                if len(sample_pred_act.shape) > 1:
                    num_pred_ann_act = sample_pred_act.shape[0]
                else:
                    num_pred_ann_act = len(sample_pred_act)
                if len(sample_pred_val.shape) > 1:
                    num_pred_ann_val = sample_pred_val.shape[0]
                else:
                    num_pred_ann_val = len(sample_pred_val)
                assert len(annotators) == num_pred_ann_act, f'{len(annotators)} != {num_pred_ann_act} | {annotators} != {sample_pred_act}'
                assert len(annotators) == num_pred_ann_val
                
                # Process each annotator's predictions and targets
                for j, annotator in enumerate(annotators):                        
                    # Check if annotator exists in model's vocabulary (for zero-shot evaluation)
                    annotator_exists_in_model = True
                    if has_annotator_mapper and annotator not in target_values['annotator_id_mapper']:
                        # Check if annotator exists in model's vocabulary
                        if annotator in self.model_trainer.model.prediction_head.act_heads.annotator_mapper.map_to_idx:
                            target_values['annotator_id_mapper'][annotator] = self.model_trainer.model.prediction_head.act_heads.annotator_mapper.map_to_idx[annotator]
                        else:
                            # Annotator doesn't exist in pretrained model (e.g., IEMOCAP annotator 'A-E1' not in MSP-Podcast model)
                            # This happens during zero-shot evaluation when test dataset has different annotators
                            annotator_exists_in_model = False
                            target_values['annotator_id_mapper'][annotator] = float('nan')  # Mark as not mappable
                            
                            if fixed_annotators == 'USE-ALL':
                                # Zero-shot mode: Use aggregate prediction for unmappable annotators
                                print(f"Zero-shot evaluation: Annotator '{annotator}' not in pretrained model vocabulary. Using aggregate prediction.")
                            else:
                                # Regular evaluation: This shouldn't happen, raise an error
                                raise KeyError(f"Annotator '{annotator}' not found in model vocabulary. Available annotators: {list(self.model_trainer.model.prediction_head.act_heads.annotator_mapper.map_to_idx.keys())}")
                    
                    # Initialize storage if needed
                    for key in [f'y_act_{annotator}', f'y_val_{annotator}',
                              f'pred_act_{annotator}', f'pred_val_{annotator}']:
                        if key not in target_values and key.startswith('y_'):
                            target_values[key] = []
                        elif key not in model_outputs and key.startswith('pred_'):
                            model_outputs[key] = []
                    
                    # Store values
                    target_values['ccc_ind_annotators'].append(annotator)
                    target_values[f'y_act_{annotator}'].append(sample_y_act[j].unsqueeze(dim=0))
                    target_values[f'y_val_{annotator}'].append(sample_y_val[j].unsqueeze(dim=0))
                    
                    # Use individual annotator predictions if available
                    model_outputs[f'pred_act_{annotator}'].append(sample_pred_act[j].unsqueeze(dim=0))
                    model_outputs[f'pred_val_{annotator}'].append(sample_pred_val[j].unsqueeze(dim=0))
                        
        target_values['ccc_ind_annotators'] = list(set(target_values['ccc_ind_annotators']))
    
    def _handle_missing_act_val_outputs(self, model_outputs: Dict[str, torch.Tensor]):
        """Handle missing act/val outputs based on model type.
        
        This method is called after _get_model_outputs, so it handles cases where
        the model didn't provide act/val outputs and we need to compute them from
        soft predictions or raise appropriate errors.
        
        Args:
            model_outputs: Dictionary containing model outputs
        """
        model_type = self.model_trainer.model.args.model_type
        
        # For baseline_aggregate models, raise error if act/val are missing
        if model_type == ModelType.BASELINE_AGGREGATE:
            if 'act' not in model_outputs or 'val' not in model_outputs:
                raise ValueError(f"BASELINE_AGGREGATE model must have 'act' and 'val' outputs, but found: {list(model_outputs.keys())}")
        
        # For AGGREGATE_GROUND_TRUTH models, these should already be set in _get_model_outputs
        elif model_type == ModelType.AGGREGATE_GROUND_TRUTH:
            if 'act' not in model_outputs or 'val' not in model_outputs:
                raise ValueError(f"AGGREGATE_GROUND_TRUTH model should have 'act' and 'val' outputs set from batch data, but found: {list(model_outputs.keys())}")
        
        # For individual annotator models, compute mean from soft predictions if missing
        elif model_type in [ModelType.INDIVIDUAL_ANNOTATOR, ModelType.INDIVIDUAL_ANNOTATOR_BETA, ModelType.INDIVIDUAL_ANNOTATOR_PLUS, ModelType.INDIVIDUAL_ANNOTATOR_CONDOR, ModelType.INDIVIDUAL_ANNOTATOR_ORTHOGONAL, ModelType.INDIVIDUAL_ANNOTATOR_LOOP]:
            if 'act' not in model_outputs and 'soft_act_preds' in model_outputs:
                if 'mean_act_preds' in model_outputs:
                    model_outputs['act'] = model_outputs['mean_act_preds']
                else:
                    model_outputs['act'] = model_outputs['soft_act_preds'].nanmean(dim=-1)
            if 'val' not in model_outputs and 'soft_val_preds' in model_outputs:
                if 'mean_val_preds' in model_outputs:
                    model_outputs['val'] = model_outputs['mean_val_preds']
                else:
                    model_outputs['val'] = model_outputs['soft_val_preds'].nanmean(dim=-1)
            # Can get the results of using aggregate annotator to make predictions by just looking at the aggregate-annotator ccc_ind metric in results
        
        elif model_type == ModelType.KDE_2D:
            if 'act' not in model_outputs and 'probability_preds' in model_outputs:
                act_dimension = model_outputs['probability_preds'].sum(dim=-1)
                probability_bins = torch.linspace(-1,1,steps=act_dimension.shape[1], device=act_dimension.device)
                model_outputs['act'] = torch.matmul(act_dimension, probability_bins)
            if 'val' not in model_outputs and 'probability_preds' in model_outputs:
                val_dimension = model_outputs['probability_preds'].sum(dim=-2)
                probability_bins = torch.linspace(-1,1,steps=val_dimension.shape[1], device=val_dimension.device)
                model_outputs['val'] = torch.matmul(val_dimension, probability_bins)
        # For other model types (KDE_2D, DEER, CLUSTERED, HUBI_MEDIUM, ONE_HOT_ANNOTATORS)
        # These should provide their own act/val outputs, so we don't modify them
        else:
            # These models should handle their own act/val outputs
            if 'act' not in model_outputs or 'val' not in model_outputs:
                raise NotImplementedError(f"Model type {model_type} does not yet handle act/val outputs")
    
    def _calculate_metrics(
        self,
        model_outputs: Dict[str, torch.Tensor],
        target_values: Dict[str, torch.Tensor],
        type_: str,
        step: int
    ) -> Dict[str, float]:
        """Calculate all evaluation metrics."""
        metrics = {}
        
        # Handle missing act/val outputs based on model type
        self._handle_missing_act_val_outputs(model_outputs)

        metrics.update(calculate_ccc_metrics(
            model_outputs,
            target_values,
            type_,
            self.model_trainer.name,
            step,
            self.logger
        ))
        
        # Per-annotator metrics (skip during pre-training)
        if 'ccc_ind_annotators' in target_values and len(target_values['ccc_ind_annotators']) > 0:
            metrics.update(calculate_per_annotator_metrics(
                model_outputs,
                target_values,
                type_,
                self.model_trainer.name,
                step,
                self.logger
            ))
        
        # KDE metrics - check for both probability_preds and probability_logits
        if 'probability_preds' in model_outputs or 'probability_logits' in model_outputs:
            metrics.update(calculate_kde_metrics(
                model_outputs,
                target_values,
                type_,
                self.model_trainer.name,
                step,
                self.logger
            ))
        
        # BMC metrics
        if self.bmc_calculator is not None:
            print('Temporarily disabling BMC metrics')
            # metrics.update(calculate_bmc_metrics(
            #     model_outputs,
            #     target_values,
            #     type_,
            #     self.model_trainer.name,
            #     step,
            #     self.logger,
            #     self.bmc_calculator
            # ))
        
        return metrics
