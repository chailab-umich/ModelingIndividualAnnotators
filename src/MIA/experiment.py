import numpy as np
import os
import re
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader, Dataset
import torch
import random
from tqdm import tqdm
from datasets import concatenate_datasets

from MIA.csv_logging import CSVWriter
from MIA.models import GenericModel, ModelTrainer
from MIA.utils import ConfigClass
from MIA.training import training_epoch, find_best_annotators, find_optimal_basis_coefficients, BatchCollator
from MIA.evaluation import validate_epoch, test_epoch
from MIA.config import ExperimentConfig, ModelType, ExperimentType
from MIA.datamanager import DatasetManager, create_dataloader
from MIA.datamanager.dataset_utils import replace_annotators_batch, has_annotators, apply_annotation_filter, remove_annotators_from_dataset
from MIA.models.annotator_utils import save_annotator_weights, restore_annotator_weights, compute_annotator_ccc, compute_annotator_ccc_separate
from MIA.AnnotatorLayer.mapper import AnnotatorMapper

class ExperimentRunner:
    """
    Main class for running experiments
    
    This class orchestrates the entire experimental process, including:
    - Dataset preparation and filtering
    - Model creation and initialization
    - Training and evaluation loops
    - Results logging
    
    Args:
        config: Complete experiment configuration
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.dataset_manager = DatasetManager(config)
        self.csv_writer = CSVWriter(config.csv_results_path, config.experiment_name)
        # Initialize global config
        ConfigClass({
            'model_save_dir': config.model_save_path,
            'log_dir': config.log_path,
            'csv_results_dir': config.csv_results_path,
            'experiment_type': config.experiment_type,
            'model_type': config.model_config.model_type,
            'prob_grid_size': config.model_config.prob_grid_size,
            'training_tasks': config.task_str
        })
        
    def run(self) -> None:
        """
        Main experiment execution
        
        This method:
        1. Sets up deterministic behavior
        2. Prepares and filters datasets
        3. Runs the experiment for each seed and fold
        4. Writes CSV results
        """
        self._setup_deterministic()
        # Run for each seed
        for seed in range(self.config.training_config.number_of_seeds):
            # Prepare datasets - get all folds if finetuning, otherwise single dataset
            datasets = self.dataset_manager.prepare_all_datasets(seed=seed)
            
            # Check if we're in finetuning mode
            is_finetuning = hasattr(self.config.training_config, 'finetune_on') and self.config.training_config.finetune_on
            
            # For journal experiments, we also need the base unfiltered datasets for multi-threshold testing
            # BUT: Skip this during finetuning and 2D KDE baseline model to disable multi-threshold testing
            base_datasets = None
            # Only need to run multiple test sets on MuSE as the datasets are different for naive and closest_label self-report types
            # Journal has multiple test sets for 100/1000+/all annotations, but only when pre-training
            # ICASSP has multiple test sets for MuSE
            if (self.config.experiment_type == ExperimentType.JOURNAL and not is_finetuning and not self.config.model_config.model_type != ModelType.KDE_2D) or (self.config.experiment_type == ExperimentType.ICASSP and self.config.training_config.finetune_on == 'muse'):
                # Create a temporary dataset manager with no filtering for base datasets
                from .config.experiment_config import DataConfig
                base_data_config = DataConfig(
                    regenerate_kde_labels=self.config.data_config.regenerate_kde_labels,
                    min_annotations=None,  # No filtering
                    cluster_other_annotators=False,
                    self_report_type=self.config.data_config.self_report_type  # No self-report filtering for base datasets
                )
                
                # Create temporary config for base dataset preparation
                from dataclasses import replace
                temp_config = replace(self.config, data_config=base_data_config)
                temp_dataset_manager = DatasetManager(temp_config)
                base_datasets = temp_dataset_manager.prepare_all_datasets(seed=seed, for_training=False)
            elif is_finetuning:
                print("Skipping multi-threshold testing as not needed for experiment")
            # If we have multiple folds (when finetuning), run each fold
            if isinstance(datasets[0], list):
                # Cross-validation folds for finetuning
                for fold_num in range(len(datasets[0])):
                    train_dataset = datasets[0][fold_num]
                    val_dataset = datasets[1][fold_num]
                    test_dataset = datasets[2][fold_num]

                    # Apply filters if configured, BUT generally skip filtering during finetuning
                    # ASRU experiments also filter on the finetuning step 
                    if not is_finetuning or self.config.experiment_type == ExperimentType.ASRU:
                        train_dataset, val_dataset, test_dataset = self.dataset_manager.apply_filters(
                            train_dataset, val_dataset, test_dataset
                        )
                    elif self.config.data_config.self_report_type is not None:
                        # Apply self-report filters if configured
                        train_dataset, val_dataset, test_dataset = self.dataset_manager.apply_self_report_filters(
                            train_dataset, val_dataset, test_dataset
                        )
                    else:
                        print(f"Finetuning mode: Skipping annotator filtering for fold {fold_num} - using all annotators")
                    
                    # Get base datasets for multi-threshold testing (None during finetuning)
                    base_train_dataset = None
                    base_test_dataset = None
                    if base_datasets is not None:
                        base_train_dataset = base_datasets[0][fold_num]  # Get base train dataset for this fold
                        base_test_dataset = base_datasets[2][fold_num]  # Get base test dataset for this fold
                    
                    self._run_seed_fold(seed, fold_num, train_dataset, val_dataset, test_dataset, base_test_dataset, base_train_dataset)
            else:
                # Single dataset (no cross-validation)
                train_dataset, val_dataset, test_dataset = datasets
                
                # Apply filters if configured, BUT skip filtering during finetuning
                if not is_finetuning or self.config.experiment_type == ExperimentType.ASRU:
                    train_dataset, val_dataset, test_dataset = self.dataset_manager.apply_filters(
                        train_dataset, val_dataset, test_dataset
                    )
                elif self.config.data_config.self_report_type is not None:
                    # Apply self-report filters if configured
                    train_dataset, val_dataset, test_dataset = self.dataset_manager.apply_self_report_filters(
                        train_dataset, val_dataset, test_dataset
                    )
                else:
                    print("Finetuning mode: Skipping annotator filtering - using all annotators")
                
                # Get base datasets for multi-threshold testing (None during finetuning)
                base_train_dataset = None
                base_test_dataset = None
                if base_datasets is not None:
                    base_train_dataset = base_datasets[0]  # Get base train dataset
                    base_test_dataset = base_datasets[2]  # Get base test dataset
                
                self._run_seed_fold(seed, 0, train_dataset, val_dataset, test_dataset, base_test_dataset, base_train_dataset)

        # Write CSV results at the end
        self.csv_writer.write_csvs()

    def _prepare_evaluation_datasets(self, seed: int, fold: int):
        """Prepare one publication split without running any training.

        This mirrors the dataset-selection and filtering branch in :meth:`run`,
        but selects exactly one seed/fold.  It is used by the checkpoint
        identification utility so recovered checkpoints can be evaluated
        independently of their original directory layout.
        """
        datasets = self.dataset_manager.prepare_all_datasets(seed=seed)
        is_finetuning = bool(getattr(self.config.training_config, 'finetune_on', None))

        if isinstance(datasets[0], list):
            number_of_folds = len(datasets[0])
            if fold < 0 or fold >= number_of_folds:
                raise IndexError(
                    f'Fold {fold} is outside the available range 0..{number_of_folds - 1}'
                )
            train_dataset, val_dataset, test_dataset = (
                split[fold] for split in datasets
            )
        else:
            if fold != 0:
                raise IndexError('Non-cross-validation experiments only have fold 0')
            train_dataset, val_dataset, test_dataset = datasets

        if not is_finetuning or self.config.experiment_type == ExperimentType.ASRU:
            train_dataset, val_dataset, test_dataset = self.dataset_manager.apply_filters(
                train_dataset, val_dataset, test_dataset
            )
        elif self.config.data_config.self_report_type is not None:
            train_dataset, val_dataset, test_dataset = self.dataset_manager.apply_self_report_filters(
                train_dataset, val_dataset, test_dataset
            )

        return train_dataset, val_dataset, test_dataset

    def _prepare_base_evaluation_datasets(self, seed: int, fold: int):
        """Prepare the unfiltered MSP-Podcast split used by Tables II--VI."""
        from dataclasses import replace
        from .config.experiment_config import DataConfig

        base_data_config = DataConfig(
            regenerate_kde_labels=self.config.data_config.regenerate_kde_labels,
            min_annotations=None,
            cluster_other_annotators=False,
            unseen_annotator_strategy=self.config.data_config.unseen_annotator_strategy,
            self_report_type=self.config.data_config.self_report_type,
            sample_batches_by_annotator=self.config.data_config.sample_batches_by_annotator,
            pretrain_with_normal_sampling=self.config.data_config.pretrain_with_normal_sampling,
            training_paradigm_3=self.config.data_config.training_paradigm_3,
        )
        temp_config = replace(self.config, data_config=base_data_config)
        base_datasets = DatasetManager(temp_config).prepare_all_datasets(
            seed=seed, for_training=False
        )

        if isinstance(base_datasets[0], list):
            return base_datasets[0][fold], base_datasets[2][fold]
        if fold != 0:
            raise IndexError('Non-cross-validation experiments only have fold 0')
        return base_datasets[0], base_datasets[2]

    def evaluate_checkpoint(self, checkpoint_path: str, seed: int, fold: int = 0):
        """Evaluate one exact checkpoint on its publication seed/fold.

        Base MSP-Podcast experiments produce all three annotation-threshold
        result sets.  Fine-tuning experiments produce one result set; callers
        may compare it with either the saved zero-shot or final-model target.

        Returns:
            A copy of the CSVWriter dataframe containing the scalar metrics.
        """
        self._setup_deterministic()
        self._set_seed(seed)
        train_dataset, val_dataset, test_dataset = self._prepare_evaluation_datasets(
            seed, fold
        )

        annotators = {
            annotator
            for annotations in (
                train_dataset['annotators']
                + val_dataset['annotators']
                + test_dataset['annotators']
            )
            for annotator in annotations
        }
        model_trainer = self._initialize_model_trainer(seed, fold, annotators)
        model_trainer.load_best_model(model_path=checkpoint_path)
        model_trainer.reset_optim()

        if getattr(self.config.training_config, 'finetune_on', None):
            self._test_model(
                model_trainer,
                test_dataset,
                seed,
                fold,
                fixed_annotators=None,
                test_name='Test',
                store_predictions=False,
            )
        else:
            base_train_dataset, base_test_dataset = self._prepare_base_evaluation_datasets(
                seed, fold
            )
            map_settings = None
            if self.config.training_config.finetune_config is not None:
                map_settings = getattr(
                    self.config.training_config.finetune_config, 'map_settings', None
                )
            self._test_model_multiple_thresholds(
                model_trainer,
                base_test_dataset,
                base_train_dataset,
                seed,
                fold,
                fixed_annotators=None,
                test_name_prefix='Test',
                unseen_strategy=self.config.data_config.unseen_annotator_strategy,
                map_settings=map_settings,
            )

        self.csv_writer.write_csvs()
        if model_trainer.name not in self.csv_writer.csv_data:
            raise RuntimeError(f'No metrics were recorded for {checkpoint_path}')
        return self.csv_writer.csv_data[model_trainer.name].copy()

    def train_and_evaluate_checkpoint(self, seed: int, fold: int = 0):
        """Train one base journal seed and immediately evaluate its best model.

        This single-seed entry point is used to reconstruct a missing release
        checkpoint without rerunning every seed in :meth:`run`.  Dataset
        preparation, RNG ordering, early stopping, and multi-population journal
        testing mirror the original base-experiment workflow.

        Returns:
            ``(metrics_frame, checkpoint_path)`` for the best validation model.
        """
        if getattr(self.config.training_config, 'finetune_on', None):
            raise ValueError('Base checkpoint training does not accept finetune_on')
        if self.config.training_config.test_only:
            raise ValueError('Checkpoint training cannot run in test-only mode')

        self._setup_deterministic()
        train_dataset, val_dataset, test_dataset = self._prepare_evaluation_datasets(
            seed, fold
        )
        base_train_dataset, base_test_dataset = self._prepare_base_evaluation_datasets(
            seed, fold
        )
        # The original run prepared datasets before _run_seed_fold reset the
        # RNG, so retain that order for reproducible model initialisation.
        self._set_seed(seed)
        annotators = {
            annotator
            for annotations in (
                train_dataset['annotators']
                + val_dataset['annotators']
                + test_dataset['annotators']
            )
            for annotator in annotations
        }
        model_trainer = self._initialize_model_trainer(seed, fold, annotators)
        self._train_model(model_trainer, train_dataset, val_dataset, start_epoch=1)
        checkpoint_path = model_trainer.early_stopping.get_best_model_path()
        model_trainer.load_best_model()
        self._test_model_multiple_thresholds(
            model_trainer,
            base_test_dataset,
            base_train_dataset,
            seed,
            fold,
            fixed_annotators=None,
            test_name_prefix='Test',
            unseen_strategy=self.config.data_config.unseen_annotator_strategy,
            map_settings=None,
        )
        self.csv_writer.write_csvs()
        if model_trainer.name not in self.csv_writer.csv_data:
            raise RuntimeError(f'No metrics were recorded for seed {seed}, fold {fold}')
        return (
            self.csv_writer.csv_data[model_trainer.name].copy(),
            checkpoint_path,
        )

    def evaluate_mapped_checkpoint(self, checkpoint_path: str, seed: int,
                                   fold: int = 0):
        """Recreate and evaluate a journal fine-tuning zero-shot model.

        The publication did not need to retain these derived checkpoints: the
        target-corpus annotator heads are produced by loading a base
        MSP-Podcast checkpoint, performing the configured mapping, and testing
        immediately.  This method deliberately mirrors that part of
        :meth:`_run_seed_fold` so release verification does not depend on a
        saved ``zero-shot-checkpoint.ckpt``.
        """
        if not getattr(self.config.training_config, 'finetune_on', None):
            raise ValueError('Annotator mapping requires a fine-tuning config')

        self._setup_deterministic()
        # Dataset preparation happened before _run_seed_fold in the original
        # runs, so reset the RNG only after selecting the historical split.
        train_dataset, val_dataset, test_dataset = self._prepare_evaluation_datasets(
            seed, fold
        )
        self._set_seed(seed)
        annotators = {
            annotator
            for annotations in (
                train_dataset['annotators']
                + val_dataset['annotators']
                + test_dataset['annotators']
            )
            for annotator in annotations
        }
        model_trainer = self._initialize_model_trainer(seed, fold, annotators)
        model_trainer.load_best_model(model_path=checkpoint_path)
        model_trainer.reset_optim()

        if self.config.model_config.model_type in {
            ModelType.INDIVIDUAL_ANNOTATOR,
            ModelType.INDIVIDUAL_ANNOTATOR_PLUS,
            ModelType.CLUSTERED,
            ModelType.INDIVIDUAL_ANNOTATOR_LOOP,
            ModelType.INDIVIDUAL_ANNOTATOR_ORTHOGONAL,
        }:
            finetune_config = self.config.training_config.finetune_config
            map_settings = (
                getattr(finetune_config, 'map_settings', None)
                if finetune_config is not None else None
            )
            self._perform_annotator_mapping(
                model_trainer,
                train_dataset,
                list(annotators),
                map_settings=map_settings,
            )

        self._test_model(
            model_trainer,
            test_dataset,
            seed,
            fold,
            fixed_annotators=None,
            test_name='Test zero-shot',
            store_predictions=False,
        )
        self.csv_writer.write_csvs()
        if model_trainer.name not in self.csv_writer.csv_data:
            raise RuntimeError(f'No metrics were recorded for {checkpoint_path}')
        return self.csv_writer.csv_data[model_trainer.name].copy()

    def reproduce_finetuning(self, checkpoint_path: str, seed: int,
                             fold: int = 0):
        """Recreate a complete journal mapping/fine-tuning/test run.

        This is the fallback for Table VIII rows whose final fine-tuned
        checkpoint was not recovered.  It records the zero-shot, one-shot,
        and final tests and saves newly trained checkpoints under this
        runner's isolated output directory.
        """
        if not getattr(self.config.training_config, 'finetune_on', None):
            raise ValueError('Fine-tuning reproduction requires finetune_on')
        if self.config.training_config.test_only:
            raise ValueError('Fine-tuning reproduction cannot run in test-only mode')

        self._setup_deterministic()
        train_dataset, val_dataset, test_dataset = self._prepare_evaluation_datasets(
            seed, fold
        )
        self._set_seed(seed)
        annotators = {
            annotator
            for annotations in (
                train_dataset['annotators']
                + val_dataset['annotators']
                + test_dataset['annotators']
            )
            for annotator in annotations
        }
        model_trainer = self._initialize_model_trainer(seed, fold, annotators)
        model_trainer.load_best_model(model_path=checkpoint_path)
        model_trainer.reset_optim()

        if self.config.model_config.model_type in {
            ModelType.INDIVIDUAL_ANNOTATOR,
            ModelType.INDIVIDUAL_ANNOTATOR_PLUS,
            ModelType.CLUSTERED,
            ModelType.INDIVIDUAL_ANNOTATOR_LOOP,
            ModelType.INDIVIDUAL_ANNOTATOR_ORTHOGONAL,
        }:
            finetune_config = self.config.training_config.finetune_config
            map_settings = (
                getattr(finetune_config, 'map_settings', None)
                if finetune_config is not None else None
            )
            self._perform_annotator_mapping(
                model_trainer,
                train_dataset,
                list(annotators),
                map_settings=map_settings,
            )

        # Preserve the original operation order: save and test the mapped
        # model, train (which saves epoch-one and early-stopping checkpoints),
        # test epoch one, then test the best validation checkpoint.
        model_trainer.save_special_checkpoint('zero-shot')
        self._test_model(
            model_trainer,
            test_dataset,
            seed,
            fold,
            fixed_annotators=None,
            test_name='Test zero-shot',
            store_predictions=False,
        )
        self._train_model(model_trainer, train_dataset, val_dataset, start_epoch=1)

        model_directory = os.path.join(
            self.config.model_save_path,
            f'model_prob{self.config.model_config.prob_grid_size}_seed_{seed}_fold_{fold}',
        )
        model_trainer.load_best_model(
            model_path=model_directory, checkpoint_type='one-shot'
        )
        self._test_model(
            model_trainer,
            test_dataset,
            seed,
            fold,
            fixed_annotators=None,
            test_name='Test one-shot',
            store_predictions=False,
        )
        model_trainer.load_best_model()
        self._test_model(
            model_trainer,
            test_dataset,
            seed,
            fold,
            fixed_annotators=None,
            test_name='Test',
            store_predictions=False,
        )

        self.csv_writer.write_csvs()
        if model_trainer.name not in self.csv_writer.csv_data:
            raise RuntimeError(f'No metrics were recorded for {checkpoint_path}')
        return self.csv_writer.csv_data[model_trainer.name].copy()
        
    def _setup_deterministic(self) -> None:
        """Setup deterministic training if configured"""
        # torch.autograd.set_detect_anomaly(True)
        torch.backends.cudnn.deterministic = True
        if self.config.training_config.toy_dataset:
            torch.autograd.set_detect_anomaly(self.config.training_config.toy_dataset)
            
    def _perform_annotator_mapping(self, model_trainer: ModelTrainer, train_dataset: Dataset, 
                                   annotators: List[str], map_settings=None) -> None:
        """
        Perform annotator mapping for finetuning
        
        Args:
            model_trainer: The model trainer instance
            train_dataset: Training dataset for mapping
            annotators: List of new annotators
        """
        finetune_config = self.config.training_config.finetune_config
        if not finetune_config:
            print("No finetuning configuration provided - skipping annotator mapping")
            return
            
        finetune_on = self.config.training_config.finetune_on
        
        # Extract parameters from config
        num_best_annotators = finetune_config.num_annotators
        map_act_val_separately = finetune_config.map_act_val_separately
        randomly_map_annotators = finetune_config.random_map
        num_samples_to_use_for_map = finetune_config.num_samples_to_use
        if model_trainer.model.args.model_type in [ModelType.INDIVIDUAL_ANNOTATOR, ModelType.INDIVIDUAL_ANNOTATOR_BETA, ModelType.INDIVIDUAL_ANNOTATOR_LOOP, 
                                                   ModelType.INDIVIDUAL_ANNOTATOR_PLUS, ModelType.CLUSTERED, ModelType.INDIVIDUAL_ANNOTATOR_ORTHOGONAL]:
            # First find the best performing annotators
            batch_collator = BatchCollator(
                annotator_mapper=model_trainer.model.prediction_head.act_heads.annotator_mapper, 
                include_aggregate=False
            )
            train_dataloader = DataLoader(
                train_dataset, batch_size=1, shuffle=True, drop_last=False, 
                collate_fn=batch_collator, num_workers=0, pin_memory=False
            )
            if map_settings is not None and 'use_CORAL' in map_settings and map_settings['use_CORAL']:
                map_settings['CORAL_enrollment_set'] = DataLoader(
                    map_settings['CORAL_enrollment_set'], batch_size=1, shuffle=True, drop_last=False, 
                    collate_fn=batch_collator, num_workers=0, pin_memory=False
                )
            
            # For orthogonal models, check if we should use basis coefficients instead of most similar annotator
            use_basis_coefficients = (
                model_trainer.model.args.model_type == ModelType.INDIVIDUAL_ANNOTATOR_ORTHOGONAL and 
                finetune_config.use_basis_coefficients
            )
            
            if use_basis_coefficients:
                print("Using optimal basis coefficients approach for orthogonal model finetuning")
                best_annotator_mapping = find_optimal_basis_coefficients(
                    train_dataloader, 
                    model_trainer, 
                    num_samples_to_use_for_map=num_samples_to_use_for_map, 
                    logger=self.csv_writer,
                    map_act_val_separately=map_act_val_separately,
                    dataset_name=finetune_on
                )
            else:
                print("Using most similar annotator approach for finetuning")
                best_annotator_mapping = find_best_annotators(
                    train_dataloader, model_trainer, N=num_best_annotators, 
                    num_samples_to_use_for_map=num_samples_to_use_for_map, logger=self.csv_writer, 
                    map_act_val_separately=map_act_val_separately, 
                    randomly_map_annotators=randomly_map_annotators, 
                    dataset_name=finetune_on,
                    map_settings=map_settings
                )
        elif model_trainer.model.args.model_type in [ModelType.AGGREGATE_GROUND_TRUTH]:
            # We want to calculate the enrollment set CCC for aggregate ground truth 
            batch_collator = BatchCollator(include_aggregate=False)
            train_dataloader = DataLoader(
                train_dataset, batch_size=1, shuffle=True, drop_last=False, 
                collate_fn=batch_collator, num_workers=0, pin_memory=False
            )
            _ = find_best_annotators(
                train_dataloader, model_trainer, N=1, 
                num_samples_to_use_for_map=num_samples_to_use_for_map, logger=self.csv_writer, 
                map_act_val_separately=map_act_val_separately, 
                randomly_map_annotators=randomly_map_annotators, 
                dataset_name=finetune_on,
                map_settings=map_settings
            )
            best_annotator_mapping = None
        else:
            best_annotator_mapping = None

        # Replace model weights and annotator mapper 
        model_trainer.model.use_new_annotators(annotators, best_annotator_mapping, map_act_val_separately)
        model_trainer.reset_optim()

    def _test_model(self, model_trainer, test_dataset, seed=None, fold=None,
                    fixed_annotators=None, test_name='Test', store_predictions=True):
        """Run the test epoch (no model loading)."""
        annotator_mapper = None
        if hasattr(model_trainer.model, 'prediction_head') and hasattr(model_trainer.model.prediction_head, 'act_heads'):
            annotator_mapper = model_trainer.model.prediction_head.act_heads.annotator_mapper
        test_loader = self._create_data_loader(
            test_dataset,
            is_training=False,
            annotator_mapper=annotator_mapper
        )
        
        # Check if KDE should be skipped based on model configuration
        should_skip_kde = model_trainer.model.args.disable_kde
        if should_skip_kde:
            print(f"Skipping KDE evaluation for {test_name}: model.disable_kde=True")
        
        store_evaluations = self.config.csv_results_path if store_predictions else None
        # store_evaluations = self.config.csv_results_path if self.config.experiment_type == ExperimentType.ICASSP else None
        test_epoch(test_loader, model_trainer, self.config.model_config.prob_grid_size, 
                  bmc_calculator=None, soft_hist=True, test_name=test_name, 
                  fixed_annotators=fixed_annotators, skip_kde=should_skip_kde,
                  csv_path_for_stored_evaluations=store_evaluations, seed=seed, fold=fold, benchmarking=self.config.training_config.benchmarking)
        self.csv_writer.write_csvs()

    def _create_test_annotator_mapper(self, model_annotator_mapper, test_annotators, model_type=None, unseen_strategy=None, similarity_mapping=None):
        """
        Create a custom annotator mapper for testing that handles unseen annotators.
        
        Args:
            model_annotator_mapper: The model's original annotator mapper
            test_annotators: Set of annotators that will appear in the test set
            model_type: Model type to determine appropriate strategy
            unseen_strategy: Override strategy ('rest', 'aggregate', 'first', 'similarity', or specific annotator name)
            similarity_mapping: Pre-computed similarity mapping (dict: annotator_name -> model_index)
            
        Returns:
            Custom annotator mapper for testing
        """
        from copy import deepcopy
        from .config.model_config import ModelType
        
        # Get known annotators from the model
        model_annotators = model_annotator_mapper.get_annotators()
        
        # Filter out artificial annotators from test set (they shouldn't appear in real data)
        artificial_annotators = {'aggregate-annotator'}
        real_test_annotators = test_annotators - artificial_annotators
        
        # Find unseen annotators (comparing only real annotators)
        real_model_annotators = model_annotators - artificial_annotators  
        unseen_annotators = real_test_annotators - real_model_annotators
        
        if not unseen_annotators:
            # No unseen annotators, use original mapper
            print("No unseen annotators found - using original mapper")
            return model_annotator_mapper
            
        print(f"Found {len(unseen_annotators)} unseen annotators: {sorted(unseen_annotators)}")
        print(f"Model has artificial annotators: {sorted(model_annotators & artificial_annotators)}")
        
        # Handle similarity-based mapping
        if unseen_strategy == 'similarity':
            if similarity_mapping is None:
                raise ValueError("Similarity strategy requested but no similarity_mapping provided")
            
            print("Using similarity-based mapping strategy")
            
            # Create new test mapper that includes all test annotators
            
            # Include artificial annotators in test mapper if they exist in the model
            all_test_annotators = real_test_annotators.copy()
            for artificial in artificial_annotators:
                if artificial in model_annotators:
                    all_test_annotators.add(artificial)
                    print(f"Including artificial annotator '{artificial}' in test mapper")
            
            # Start with all test annotators
            sorted_test_annotators = sorted(all_test_annotators)
            test_mapper = AnnotatorMapper(sorted_test_annotators)
            test_mapper.set_get_idx()
            
            # Apply similarity mapping for each unseen annotator
            for annotator in unseen_annotators:
                if annotator in similarity_mapping:
                    model_target_index = similarity_mapping[annotator]
                    # Convert model index back to annotator name
                    model_annotator_mapper.set_get_annotator()
                    target_annotator_name = model_annotator_mapper[model_target_index]
                    # Use the MODEL's original index directly (not test mapper index)
                    test_mapper.map_to_idx[annotator] = model_target_index
                    print(f"  {annotator} → {target_annotator_name} (model index {model_target_index})")
                else:
                    # Fallback to first annotator if no similarity mapping found
                    first_annotator = sorted(model_annotators - artificial_annotators)[0]
                    model_annotator_mapper.set_get_idx()
                    model_fallback_index = model_annotator_mapper[first_annotator]
                    test_mapper.map_to_idx[annotator] = model_fallback_index
                    print(f"  {annotator} → {first_annotator} (model index {model_fallback_index}, fallback)")
            
            # Update the reverse mapping to be consistent
            test_mapper.map_to_annotator = {idx: ann for ann, idx in test_mapper.map_to_idx.items()}
            return test_mapper
        
        # Handle other strategies (original logic)
        # Determine strategy - explicit override takes precedence
        if unseen_strategy:
            target_annotator = unseen_strategy
            print(f"Using explicit strategy: '{target_annotator}'")
            # Validate that the explicit strategy is compatible with the model
            self._validate_unseen_strategy(target_annotator, model_type, model_annotators)
        else:
            # Auto-detect strategy based on model type and available annotators
            target_annotator = self._detect_unseen_strategy(model_type, model_annotators)
            
        # Check if target annotator exists in model
        if target_annotator not in model_annotators:
            raise ValueError(f"Target annotator '{target_annotator}' not found in model annotators: {sorted(model_annotators)}")
            
        # Get the target annotator's index
        model_annotator_mapper.set_get_idx()
        target_index = model_annotator_mapper[target_annotator]
        print(f"Mapping unseen annotators to '{target_annotator}' (model index {target_index})")
        
        # Create new test mapper that includes all test annotators (including artificial ones if they exist in model)
        
        # Include artificial annotators in test mapper if they exist in the model
        all_test_annotators = real_test_annotators.copy()
        for artificial in artificial_annotators:
            if artificial in model_annotators:
                all_test_annotators.add(artificial)
                print(f"Including artificial annotator '{artificial}' in test mapper")
        
        # Start with all test annotators
        sorted_test_annotators = sorted(all_test_annotators)
        test_mapper = AnnotatorMapper(sorted_test_annotators)
        
        # Now modify the mapping so unseen annotators point to the target annotator's index
        test_mapper.set_get_idx()
        
        for annotator in unseen_annotators:
            # Map this unseen annotator to the same index as the target annotator
            test_mapper.map_to_idx[annotator] = target_index
            print(f"  {annotator} → {target_annotator} (model index {target_index})")
            
        # Update the reverse mapping to be consistent
        # Note: This creates a many-to-one mapping where multiple annotators map to the same index
        test_mapper.map_to_annotator = {idx: ann for ann, idx in test_mapper.map_to_idx.items()}
        
        return test_mapper
    
    def _validate_unseen_strategy(self, target_annotator, model_type, model_annotators):
        """
        Validate that the chosen unseen annotator strategy is compatible with the trained model.
        
        Args:
            target_annotator: The target annotator to use for mapping unseen annotators
            model_type: The model type that was trained
            model_annotators: Set of annotators the model knows about
            
        Raises:
            ValueError: If the strategy is incompatible with the model
        """
        from .config.model_config import ModelType
        
        if target_annotator == 'aggregate-annotator':
            if model_type != ModelType.INDIVIDUAL_ANNOTATOR_PLUS:
                raise ValueError(
                    f"Strategy 'aggregate-annotator' can only be used with INDIVIDUAL_ANNOTATOR_PLUS models, "
                    f"but model type is {model_type}. The aggregate-annotator is only available in "
                    f"individual_annotator_plus models."
                )
            if 'aggregate-annotator' not in model_annotators:
                raise ValueError(
                    f"Strategy 'aggregate-annotator' specified but model does not contain aggregate-annotator. "
                    f"Model annotators: {sorted(model_annotators)}"
                )                
        elif target_annotator in ['first', 'self-report', 'similarity']:
            # These are general strategies that should work with any model
            pass
        else:
            # Assume it's a specific annotator name - just check it exists
            if target_annotator not in model_annotators:
                raise ValueError(
                    f"Strategy '{target_annotator}' specified but this annotator was not found in the model. "
                    f"Model annotators: {sorted(model_annotators)}"
                )
    
    def _detect_unseen_strategy(self, model_type, model_annotators):
        """
        Auto-detect the best strategy for handling unseen annotators based on model type and available annotators.
        
        Args:
            model_type: The model type
            model_annotators: Set of annotators the model knows about
            
        Returns:
            Target annotator name to use for mapping unseen annotators
        """
        from .config.model_config import ModelType
        
        # Check for artificial annotators first
        if 'aggregate-annotator' in model_annotators:
            print(f"Detected 'aggregate-annotator' - using for unseen annotators")
            return 'aggregate-annotator'
        
        # Model-type specific strategies
        if model_type in [ModelType.INDIVIDUAL_ANNOTATOR, ModelType.INDIVIDUAL_ANNOTATOR_PLUS, ModelType.CLUSTERED, ModelType.INDIVIDUAL_ANNOTATOR_LOOP, ModelType.INDIVIDUAL_ANNOTATOR_ORTHOGONAL]:
            # For individual annotator models without special annotators, use first annotator
            first_annotator = sorted(model_annotators - {'aggregate-annotator'})[0]
            print(f"Individual annotator model without special annotators - using first annotator: '{first_annotator}'")
            return first_annotator
        elif model_type in [ModelType.BASELINE_AGGREGATE, ModelType.AGGREGATE_GROUND_TRUTH]:
            # These models should theoretically handle any annotators, but we still need a mapping
            first_annotator = sorted(model_annotators - {'aggregate-annotator'})[0]
            print(f"Aggregate model - using first annotator: '{first_annotator}'")
            return first_annotator
        else:
            # Default fallback
            first_annotator = sorted(model_annotators - {'aggregate-annotator'})[0]
            print(f"Unknown model type {model_type} - using first annotator: '{first_annotator}'")
            return first_annotator

    def _calculate_similarity_mapping(self, model_trainer, train_dataset, unseen_annotators, map_settings=None):
        """
        Calculate similarity-based mapping for unseen annotators using the same approach as finetuning.
        
        Args:
            model_trainer: The model trainer instance
            train_dataset: Training dataset containing the unseen annotators (for fair similarity calculation)
            unseen_annotators: Set of unseen annotator names
            
        Returns:
            Dictionary mapping unseen annotator names to their best similar model annotator index
        """
        print(f"Computing similarity mapping for {len(unseen_annotators)} unseen annotators using training data...")
        
        # Create a mini dataset containing only samples with unseen annotators
        # and filter each sample to keep only the unseen annotator labels. Using
        # the HuggingFace Dataset filtering utilities is significantly faster
        # than iterating in Python.

        original_format = train_dataset.format

        def has_unseen(sample):
            return bool(set(sample['annotators']) & unseen_annotators)

        unseen_samples = train_dataset.filter(has_unseen)

        if len(unseen_samples) == 0:
            print("Warning: No samples found with unseen annotators for similarity calculation")
            return {}
            

        def keep_only_unseen(sample):
            unseen_indices = [i for i, ann in enumerate(sample['annotators']) if ann in unseen_annotators]
            sample['annotators'] = [sample['annotators'][i] for i in unseen_indices]
            if 'soft_act_labels' in sample:
                sample['soft_act_labels'] = sample['soft_act_labels'][unseen_indices]
            if 'soft_val_labels' in sample:
                sample['soft_val_labels'] = sample['soft_val_labels'][unseen_indices]
            return sample

        unseen_samples = unseen_samples.map(keep_only_unseen)
        unseen_samples.set_format(**original_format)            
        print(f"Found {len(unseen_samples)} training samples containing unseen annotators for similarity calculation")
        
        # Get the annotator mapper from the model trainer for the collate function
        model_annotator_mapper = model_trainer.model.prediction_head.act_heads.annotator_mapper
        model_type = self.config.model_config.model_type
        
        similarity_loader = create_dataloader(
            unseen_samples,
            batch_size=1,
            model_type=model_type,
            annotator_mapper=model_annotator_mapper,
            is_training=False
        )
        
        # Use the existing find_best_annotators function with N=1 to get the best single match
        similarity_mapping = find_best_annotators(
            eval_dataloader=similarity_loader,
            model_trainer=model_trainer,
            N=1,  # Just find the single best match
            num_samples_to_use_for_map='all',
            logger=self.csv_writer,  # Use CSV writer as logger
            map_act_val_separately=False,
            randomly_map_annotators=False,
            dataset_name="unseen_annotator_mapping",
            map_settings=map_settings
        )
        
        # Convert the mapping from annotator name -> [index] to annotator name -> index
        final_mapping = {}
        model_annotator_mapper = model_trainer.model.prediction_head.act_heads.annotator_mapper
        
        for annotator_name in unseen_annotators:
            if annotator_name in similarity_mapping:
                best_model_idx = similarity_mapping[annotator_name][0]  # Get first (and only) best match
                # Convert tensor to Python int if needed
                if hasattr(best_model_idx, 'item'):
                    best_model_idx = best_model_idx.item()
                
                # Convert from model index to annotator name and back to ensure consistency
                model_annotator_mapper.set_get_annotator()
                best_annotator_name = model_annotator_mapper[best_model_idx]
                model_annotator_mapper.set_get_idx()
                best_model_idx = model_annotator_mapper[best_annotator_name]
                final_mapping[annotator_name] = best_model_idx
                print(f"  {annotator_name} → {best_annotator_name} (index {best_model_idx})")
            else:
                print(f"  Warning: No similarity mapping found for {annotator_name}")
        
        return final_mapping

    def _get_unseen_annotator_mappings(self, model_annotator_mapper, test_annotators, model_type=None, unseen_strategy=None, similarity_mapping=None):
        """
        Determine mappings for unseen annotators without creating a new mapper.
        
        Args:
            model_annotator_mapper: The model's original annotator mapper
            test_annotators: Set of annotators that will appear in the test set
            model_type: Model type to determine appropriate strategy
            unseen_strategy: Strategy for handling unseen annotators
            similarity_mapping: Pre-computed similarity mapping (dict: annotator_name -> model_index)
            
        Returns:
            Dictionary mapping unseen annotator names to target model indices
        """
        # Get known annotators from the model
        model_annotators = model_annotator_mapper.get_annotators()
        
        # Filter out artificial annotators from test set
        artificial_annotators = {'aggregate-annotator'}
        real_test_annotators = test_annotators - artificial_annotators
        real_model_annotators = model_annotators - artificial_annotators
        unseen_annotators = real_test_annotators - real_model_annotators
        
        if not unseen_annotators:
            print("No unseen annotators to map")
            return {}
            
        print(f"Mapping {len(unseen_annotators)} unseen annotators: {sorted(unseen_annotators)}")
        
        mappings = {}
        
        if unseen_strategy == 'similarity':
            if similarity_mapping is None:
                raise ValueError("Similarity strategy requested but no similarity_mapping provided")
            
            print("Using similarity-based mapping")
            for annotator in unseen_annotators:
                if annotator in similarity_mapping:
                    target_index = similarity_mapping[annotator]
                    mappings[annotator] = target_index
                    # Get target annotator name for logging
                    model_annotator_mapper.set_get_annotator()
                    target_name = model_annotator_mapper[target_index]
                    print(f"  {annotator} → {target_name} (model index {target_index})")
                else:
                    # Fallback to first annotator
                    first_annotator = sorted(real_model_annotators)[0]
                    model_annotator_mapper.set_get_idx()
                    fallback_index = model_annotator_mapper[first_annotator]
                    mappings[annotator] = fallback_index
                    print(f"  {annotator} → {first_annotator} (model index {fallback_index}, fallback)")
        else:
            # Handle other strategies
            if unseen_strategy:
                target_annotator = unseen_strategy
            else:
                # Auto-detect strategy
                target_annotator = self._detect_unseen_strategy(model_type, model_annotators)
            
            # Validate target annotator exists
            if target_annotator not in model_annotators:
                raise ValueError(f"Target annotator '{target_annotator}' not found in model annotators: {sorted(model_annotators)}")
            
            # Get target index
            model_annotator_mapper.set_get_idx()
            target_index = model_annotator_mapper[target_annotator]
            
            print(f"Mapping all unseen annotators to '{target_annotator}' (model index {target_index})")
            for annotator in unseen_annotators:
                mappings[annotator] = target_index
                
        return mappings


    def _determine_unseen_strategies(self, model_type, has_aggregate_annotator):
        """
        Determine which unseen annotator strategies to test based on model configuration.
        
        Args:
            model_type: The model type
            has_aggregate_annotator: Whether the model has aggregate-annotator (IA+ models)
            
        Returns:
            List of strategies to test
        """
        from .config.model_config import ModelType
        
        strategies = []
        
        # All individual annotator models get similarity strategy
        if model_type in [ModelType.INDIVIDUAL_ANNOTATOR, ModelType.INDIVIDUAL_ANNOTATOR_PLUS, ModelType.CLUSTERED, ModelType.INDIVIDUAL_ANNOTATOR_LOOP, ModelType.INDIVIDUAL_ANNOTATOR_ORTHOGONAL]:
            strategies.append('similarity')
        
        # Models with "plus" modifier also get aggregate-annotator strategy
        if has_aggregate_annotator:
            strategies.append('aggregate-annotator')

        return strategies

    def _test_model_multiple_thresholds(self, model_trainer, base_test_dataset, base_train_dataset, seed=None, fold=None, fixed_annotators=None, test_name_prefix='Test', unseen_strategy=None, map_settings=None):
        """
        Test the model on multiple annotation threshold datasets with multiple unseen annotator strategies.
        
        Args:
            model_trainer: The model trainer instance
            base_test_dataset: The original test dataset (before any filtering)
            base_train_dataset: The original train dataset (before any filtering) for similarity calculations
            seed: Random seed
            fold: Fold number
            fixed_annotators: Fixed annotators setting
            test_name_prefix: Prefix for test names
            unseen_strategy: Override strategy - if provided, only this strategy will be used
        """
        # Define the test configurations we want to run
        # For self-report experiments, test both naive and closest_label versions
        # For other experiments, use annotation thresholds
        is_self_report_experiment = (hasattr(self.config.data_config, 'self_report_type') and 
                                    self.config.data_config.self_report_type is not None)
        
        if is_self_report_experiment:
            # Self-report experiments: test both self_report_type types
            print("Detected self-report experiment - testing both naive and closest_label versions")
            test_configs = [
                {'self_report_type': 'naive', 'name_suffix': 'naive'},
                {'self_report_type': 'closest_label', 'name_suffix': 'closest_label'}
            ]
        else:
            # Journal experiments: test multiple annotation thresholds
            print("Detected journal experiment - testing multiple annotation thresholds")
            # Include full dataset plus thresholds that would introduce unseen annotators
            # Never cluster annotators into 'rest' during testing - we want to test real unseen annotators
            test_configs = [
                {'min_annotations': None, 'name_suffix': 'full-dataset'},  # Full dataset test
                {'min_annotations': 100, 'name_suffix': '100plus-annotations'}, 
                {'min_annotations': 1000, 'name_suffix': '1000plus-annotations'}
            ]
        
        # Get the model's annotator mapper and model type
        model_annotator_mapper = None
        model_type = self.config.model_config.model_type
        if hasattr(model_trainer.model, 'prediction_head') and hasattr(model_trainer.model.prediction_head, 'act_heads'):
            if hasattr(model_trainer.model.prediction_head.act_heads, 'annotator_mapper'):
                model_annotator_mapper = model_trainer.model.prediction_head.act_heads.annotator_mapper
        
        # Determine what threshold the model was trained with
        training_min_annotations = getattr(self.config.data_config, 'min_annotations', None)
        print(f"Model was trained with min_annotations: {training_min_annotations}")
        
        # Get model annotators (what the model was trained on)
        model_annotators = set()
        if model_annotator_mapper is not None:
            model_annotators = model_annotator_mapper.get_annotators()
            
        # Determine which strategies this model supports (if it has an annotator mapper)
        available_strategies = []
        if model_annotator_mapper is not None:
            has_aggregate_annotator = 'aggregate-annotator' in model_annotators

            available_strategies = self._determine_unseen_strategies(
                model_type, has_aggregate_annotator
            )
            print(f"Model supports unseen annotator strategies: {available_strategies}")
        else:
            print("Model has no annotator mapper - will use default strategy only")
        
        # Pre-compute similarity mapping if needed (only if model supports similarity strategy)
        similarity_mapping = None
        
        for test_config in test_configs:
            print(f"\n=== Testing on {test_config['name_suffix']} ===")
            
            # Create filtered test dataset for this configuration
            if 'self_report_type' in test_config:
                # Self-report testing - create dataset with specific self-report type
                filtered_test_dataset = self.dataset_manager.create_test_dataset_with_version(
                    base_test_dataset, test_config['self_report_type']
                )
            else:
                # Annotation threshold testing - create dataset with annotation filtering
                if test_config['min_annotations'] is None:
                    # Full dataset - no filtering
                    filtered_test_dataset = base_test_dataset
                    print("Using full dataset (no annotation filtering)")
                else:
                    filtered_test_dataset = apply_annotation_filter(
                        base_test_dataset,
                        min_annotations=test_config['min_annotations'],
                        cluster_other_annotators=False,  # Never cluster during testing
                        all_annotators=[x for annotators in base_train_dataset['annotators'] for x in annotators], # the 100+ and 1000+ sets are based on the training set having 100+/1000+ samples
                        update_consensus=self.config.experiment_type == ExperimentType.ASRU,
                    )
            
            # Get all annotators in this test set
            test_annotators = set([ann for anns in filtered_test_dataset['annotators'] for ann in anns])
            
            print(f"Test set contains {len(test_annotators)} unique annotators")
            
            # Determine if this test set contains unseen annotators
            has_unseen_annotators = False
            
            if model_annotator_mapper is not None:
                # Find unseen annotators (excluding artificial ones)
                artificial_annotators = {'aggregate-annotator'}
                real_test_annotators = test_annotators - artificial_annotators
                real_model_annotators = model_annotators - artificial_annotators
                unseen_annotators = real_test_annotators - real_model_annotators
                
                if unseen_annotators:
                    has_unseen_annotators = True
                    print(f"Found {len(unseen_annotators)} unseen annotators: {sorted(unseen_annotators)}")
                    
                    # Compute similarity mapping if needed and not already computed
                    if 'similarity' in available_strategies and similarity_mapping is None:
                        print(f"Pre-computing similarity mapping for unseen annotators...")
                        similarity_mapping = self._calculate_similarity_mapping(
                            model_trainer, base_train_dataset, unseen_annotators, map_settings=map_settings
                        )
                else:
                    print("No unseen annotators found - all test annotators were seen during training")
            
            # Determine which strategies to test for this specific test configuration
            if unseen_strategy is not None:
                # Override: only test the specified strategy
                strategies_to_test = [unseen_strategy]
                print(f"Using override strategy: {unseen_strategy}")
            elif has_unseen_annotators and available_strategies:
                # Model was trained with filtering and test set has unseen annotators
                # Test all strategies the model supports
                strategies_to_test = available_strategies
                print(f"Testing {len(strategies_to_test)} unseen annotator strategies: {strategies_to_test}")
            else:
                # Either no unseen annotators OR model doesn't support any strategies
                # Just test with default evaluation
                strategies_to_test = [None]
                if has_unseen_annotators:
                    print("Model doesn't support unseen annotator strategies - using default evaluation")
                else:
                    print("No unseen annotators - using default evaluation")
            
            # Test with each strategy
            for strategy in strategies_to_test:
                if strategy is None:
                    strategy_suffix = ""
                    print(f"\n--- Testing with default strategy ---")
                else:
                    strategy_suffix = f" {strategy}"
                    print(f"\n--- Testing with {strategy} strategy ---")
                
                # Handle unseen annotator mapping by temporarily modifying the model's annotator mapper
                original_mappings = {}
                if model_annotator_mapper is not None and strategy is not None:
                    # Determine mapping for unseen annotators
                    unseen_mappings = self._get_unseen_annotator_mappings(
                        model_annotator_mapper, 
                        test_annotators, 
                        model_type=model_type,
                        unseen_strategy=strategy,
                        similarity_mapping=similarity_mapping
                    )
                    
                    # Temporarily modify the model's original mapper in place
                    for unseen_annotator, target_index in unseen_mappings.items():
                        if unseen_annotator in model_annotator_mapper.map_to_idx:
                            # Store original mapping for restoration
                            original_mappings[unseen_annotator] = model_annotator_mapper.map_to_idx[unseen_annotator]
                        else:
                            # Mark as newly added (use None to indicate it should be deleted on restore)
                            original_mappings[unseen_annotator] = None
                        # Set new mapping
                        model_annotator_mapper.map_to_idx[unseen_annotator] = target_index
                        print(f"  Temporarily mapped {unseen_annotator} to model index {target_index}")
                    
                    # Regenerate the reverse mapping to be consistent
                    # CRITICAL: We can't use the standard {idx: ann for ann, idx in map_to_idx.items()} reversal
                    # because with many-to-one mapping (e.g., ann1:0, unseen_ann:0), the reversal would overwrite
                    # and we'd lose annotators. Since get_annotators() only uses .values(), we use sequential
                    # keys 0,1,2... and include ALL annotator names as values.
                    all_annotator_names = list(model_annotator_mapper.map_to_idx.keys())
                    model_annotator_mapper.map_to_annotator = {i: ann for i, ann in enumerate(all_annotator_names)}
                    
                    # Create test loader with the modified original mapper
                    # The BatchCollator will now use the updated mapper to create correct annotator_masks
                    test_loader = self._create_data_loader(
                        filtered_test_dataset, 
                        is_training=False,
                        annotator_mapper=model_annotator_mapper
                    )
                else:
                    # Use original annotator mapper or no mapper (for models without mappers)
                    test_loader = self._create_data_loader(
                        filtered_test_dataset, 
                        is_training=False,
                        annotator_mapper=model_annotator_mapper
                    )
                
                # Run test epoch with the mapper configuration
                test_name = f"{test_name_prefix} {test_config['name_suffix']}{strategy_suffix}"
                
                # Determine if KDE should be skipped during testing
                # Skip KDE if: 
                # 1. Model has disable_kde=True (explicit model configuration), OR
                # 2. We're testing on a subset dataset (100+ or 1000+ annotations)
                #    because KDE ground truth was calculated for full dataset
                model_disable_kde = model_trainer.model.args.disable_kde
                is_subset_dataset = 'min_annotations' in test_config and test_config['min_annotations'] is not None
                should_skip_kde = model_disable_kde or is_subset_dataset
                
                if should_skip_kde:
                    reason = []
                    if model_disable_kde:
                        reason.append("model.disable_kde=True")
                    if is_subset_dataset:
                        reason.append(f"subset dataset (min_annotations={test_config['min_annotations']})")
                    print(f"Skipping KDE evaluation for {test_config['name_suffix']}: {', '.join(reason)}")
                
                store_evaluations = self.config.csv_results_path if self.config.experiment_type == ExperimentType.ICASSP else None
                test_epoch(test_loader, model_trainer, self.config.model_config.prob_grid_size, 
                          bmc_calculator=None, soft_hist=True, test_name=test_name, 
                          fixed_annotators=fixed_annotators, skip_kde=should_skip_kde,
                          csv_path_for_stored_evaluations=store_evaluations, seed=seed, fold=fold, benchmarking=self.config.training_config.benchmarking)
                self.csv_writer.write_csvs()
                # Restore original mappings if they were modified
                if original_mappings:
                    for unseen_annotator, original_index in original_mappings.items():
                        if original_index is not None:
                            # This annotator existed originally, restore its index
                            model_annotator_mapper.map_to_idx[unseen_annotator] = original_index
                            print(f"  Restored {unseen_annotator} to original index {original_index}")
                        else:
                            # This annotator was added temporarily, remove it completely
                            if unseen_annotator in model_annotator_mapper.map_to_idx:
                                del model_annotator_mapper.map_to_idx[unseen_annotator]
                                print(f"  Removed {unseen_annotator} from mapper (was temporarily added)")
                    
                    # Regenerate the reverse mapping to be consistent after restoration
                    model_annotator_mapper.map_to_annotator = {idx: ann for ann, idx in model_annotator_mapper.map_to_idx.items()}
                    print("Restored original annotator mappings")

    def _run_seed_fold(self, seed: int, fold: int, train_dataset: Dataset, 
                       val_dataset: Dataset, test_dataset: Dataset, base_test_dataset: Dataset = None, base_train_dataset: Dataset = None) -> None:
        """
        Run experiment for a specific seed and fold
        
        Args:
            seed: Random seed for reproducibility
            fold: Fold number (0 for non-cross-validation experiments)
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            base_test_dataset: Base test dataset for multi-threshold testing
            base_train_dataset: Base train dataset for multi-threshold testing
        """
        self._set_seed(seed)
        
        # Initialize model trainer (which creates the model internally)
        annotators = [x for annotators in tqdm(train_dataset['annotators'] + val_dataset['annotators'] + test_dataset['annotators'], desc='Calculating annotators present in training set') for x in annotators]
        annotators = set(annotators)  # Convert to set to remove duplicates

        model_trainer = self._initialize_model_trainer(seed, fold, annotators)
        map_settings = self.config.training_config.finetune_config.map_settings if self.config.training_config.finetune_config is not None else None
            # if map_settings is not None and 'use_CORAL' in map_settings and map_settings['use_CORAL']:
            #     map_settings['CORAL_enrollment_set'] = val_dataset
            #     if self.config.training_config.finetune_on == 'muse':
            #         map_settings['num_ordinal_bins'] = 9
            #     else:
            #         map_settings['num_ordinal_bins'] = 5

        # No else because we want it to crash if this value is attempted to be used 
        # Step 1: Load base model and set up (only for finetuning)
        if hasattr(self.config.training_config, 'finetune_on') and self.config.training_config.finetune_on:
            # Determine if we are in zero-shot all-annotators mode (special exit-only run)
            all_annotators_zeroshot = getattr(self.config.training_config, 'all_annotators_zeroshot', False)
            # ASRU experiments should run the all annotators zero-shot experiment, but not exit afterwards
            bonus_all_annotators_zero_shot = self.config.experiment_type == ExperimentType.ASRU
            # Run zero-shot all-annotators experiment
            run_zero_shot_all_annotators = bonus_all_annotators_zero_shot or all_annotators_zeroshot
            if run_zero_shot_all_annotators or not self.config.training_config.test_only:
                # When all annotators zero shot we just want to load the pre-trained model and skip the mapping
                # When not doing test-only we want to load the model and then perform the mapping, so in both cases we want to load the same base model here 
                # Load pretrained base model for finetuning
                base_model_path = self._get_base_model_path(seed, fold) if seed is not None and fold is not None else None
                model_trainer.load_best_model(model_path=base_model_path)
                model_trainer.reset_optim()

                # Step 2)a): Run zero-shot all-annotators experiment, if required
                # This only makes sense if using an individual annotator model, other models have no definition of "all-annotators"
                if run_zero_shot_all_annotators and self.config.model_config.model_type in [ModelType.INDIVIDUAL_ANNOTATOR, ModelType.INDIVIDUAL_ANNOTATOR_PLUS, ModelType.CLUSTERED, ModelType.INDIVIDUAL_ANNOTATOR_LOOP, ModelType.INDIVIDUAL_ANNOTATOR_ORTHOGONAL]:
                    # Step 2)a).1: Zero-shot checkpoint and testing (only for finetuning experiments)
                    print('Running zero-shot all-annotators evaluation...')
                    zero_shot_fixed_annotators = 'USE-ALL'
                    zero_shot_test_name = 'Test zero-shot all-annotators'
                    
                    # Use multi-threshold testing for journal experiments
                    if base_test_dataset is not None and not all_annotators_zeroshot:
                        self._test_model_multiple_thresholds(model_trainer, base_test_dataset, base_train_dataset, seed, fold, 
                                                        fixed_annotators=zero_shot_fixed_annotators, test_name_prefix=zero_shot_test_name, unseen_strategy=self.config.data_config.unseen_annotator_strategy, map_settings=map_settings)
                    else:
                        self._test_model(model_trainer, test_dataset, seed, fold, 
                                    fixed_annotators=zero_shot_fixed_annotators, test_name=zero_shot_test_name)

                # Step 2)a).2: Exit early if this is a zero-shot-only run
                if all_annotators_zeroshot:
                    print('Exiting after zero-shot all-annotators evaluation.')
                    return

                # Step 2)b): Map annotators for individual annotator models (only if not special all-annotators run (already exited) and only during finetuning)
                if self.config.model_config.model_type in [ModelType.INDIVIDUAL_ANNOTATOR, ModelType.INDIVIDUAL_ANNOTATOR_PLUS, ModelType.CLUSTERED, ModelType.INDIVIDUAL_ANNOTATOR_LOOP, ModelType.INDIVIDUAL_ANNOTATOR_ORTHOGONAL] and not all_annotators_zeroshot:
                    print('Performing annotator mapping for finetuning...')
                    self._perform_annotator_mapping(model_trainer, train_dataset, list(annotators), map_settings=map_settings)

                # Training mode: Current model IS the zero-shot model, save it
                if not self.config.training_config.test_only:
                    model_trainer.save_special_checkpoint('zero-shot')
            # Can't use else herre as ASRU + test only can still go through above path, we need to overwrite the loaded zero-shot all-annotator model in this case with normal model below for test only
            if self.config.training_config.test_only:
                # Test-only mode: Load the zero-shot checkpoint
                # Model will already be mapped because it was saved as the zero-shot model above after the mapping step
                model_trainer.load_best_model(model_path=f'{self.config.model_save_path}/model_prob{self.config.model_config.prob_grid_size}_seed_{seed}_fold_{fold}', checkpoint_type='zero-shot')
                model_trainer.reset_optim()

            # Step 3: Zero-shot checkpoint and testing (only for finetuning experiments)
            print('Running zero-shot evaluation...')
            zero_shot_fixed_annotators = None
            zero_shot_test_name = 'Test zero-shot'
            
            # Use multi-threshold testing for journal experiments
            if base_test_dataset is not None and not all_annotators_zeroshot:
                self._test_model_multiple_thresholds(model_trainer, base_test_dataset, base_train_dataset, seed, fold, 
                                                   fixed_annotators=zero_shot_fixed_annotators, test_name_prefix=zero_shot_test_name, unseen_strategy=self.config.data_config.unseen_annotator_strategy, map_settings=map_settings)
            else:
                self._test_model(model_trainer, test_dataset, seed, fold, 
                               fixed_annotators=zero_shot_fixed_annotators, test_name=zero_shot_test_name)

        # Step 5: Full training (continue from current state, only if not test_only) (also does one-shot evaluation)
        if not self.config.training_config.test_only and self.config.model_config.model_type != ModelType.AGGREGATE_GROUND_TRUTH:
            # Continue with full training from current state (after one-shot)
            self._train_model(model_trainer, train_dataset, val_dataset, start_epoch=1)

        # Step 6: One-shot evaluation. If the model ran with test_only, it will have saved a one-shot checkpoint
        # If the model just trained, it created a saved one-shot checkpoint. In either case we can load and test the model
        if hasattr(self.config.training_config, 'finetune_on') and self.config.training_config.finetune_on:
            model_trainer.load_best_model(model_path=f'{self.config.model_save_path}/model_prob{self.config.model_config.prob_grid_size}_seed_{seed}_fold_{fold}', checkpoint_type='one-shot')
            print('Running one-shot evaluation...')
            
            # Use multi-threshold testing for journal experiments
            if base_test_dataset is not None:
                self._test_model_multiple_thresholds(model_trainer, base_test_dataset, base_train_dataset, seed, fold, fixed_annotators=None, test_name_prefix='Test one-shot', unseen_strategy=self.config.data_config.unseen_annotator_strategy, map_settings=map_settings)
            else:
                self._test_model(model_trainer, test_dataset, seed, fold, fixed_annotators=None, test_name='Test one-shot')

        
        # Step 7: Load best model and regular testing
        if self.config.training_config.test_only:
            # Test-only mode: Load from the current experiment's model directory
            model_name = f'model_prob{self.config.model_config.prob_grid_size}_seed_{seed}_fold_{fold}'
            model_path = f'{self.config.model_save_path}/{model_name}'
            model_trainer.load_best_model(model_path=model_path)
        else:
            # Training mode: Load the best model saved by early stopping
            model_trainer.load_best_model()
            
        # Use multi-threshold testing for journal experiments
        if base_test_dataset is not None:
            print("Running multi-threshold testing for journal experiments...")
            self._test_model_multiple_thresholds(model_trainer, base_test_dataset, base_train_dataset, seed, fold, fixed_annotators=None, test_name_prefix='Test', unseen_strategy=self.config.data_config.unseen_annotator_strategy, map_settings=map_settings)
        else:
            # Fallback to single test for non-journal experiments
            self._test_model(model_trainer, test_dataset, seed, fold, fixed_annotators=None, test_name='Test')

        # For orthogonal models with a learned consensus layer, re-run tests with only consensus + per-annotator bias (no basis weight contributions)
        if (self.config.model_config.model_type == ModelType.INDIVIDUAL_ANNOTATOR_ORTHOGONAL
                and self.config.model_config.orthogonal_model_features
                and 'learn_consensus' in self.config.model_config.orthogonal_model_features):
            print("Running consensus-bias-only tests for orthogonal model...")
            model_trainer.model.prediction_head.consensus_bias_only = True
            try:
                if base_test_dataset is not None:
                    self._test_model_multiple_thresholds(model_trainer, base_test_dataset, base_train_dataset, seed, fold, fixed_annotators=None, test_name_prefix='Test consensus-bias-only', unseen_strategy=self.config.data_config.unseen_annotator_strategy, map_settings=map_settings)
                else:
                    self._test_model(model_trainer, test_dataset, seed, fold, fixed_annotators=None, test_name='Test consensus-bias-only')
            finally:
                model_trainer.model.prediction_head.consensus_bias_only = False

        # Run one more all-annotators test after model training for ASRU experiments (used to report performance of pre-trained models)
        # This only makes sense if using an individual annotator model, other models have no definition of "all-annotators"
        if self.config.experiment_type == ExperimentType.ASRU and self.config.model_config.model_type in [ModelType.INDIVIDUAL_ANNOTATOR, ModelType.INDIVIDUAL_ANNOTATOR_PLUS, ModelType.CLUSTERED, ModelType.INDIVIDUAL_ANNOTATOR_LOOP, ModelType.INDIVIDUAL_ANNOTATOR_ORTHOGONAL]:
            if base_test_dataset is not None:
                print("Running multi-threshold testing for journal experiments...")
                self._test_model_multiple_thresholds(model_trainer, base_test_dataset, base_train_dataset, seed, fold, fixed_annotators='USE-ALL', test_name_prefix='Test all-annotators', unseen_strategy=self.config.data_config.unseen_annotator_strategy, map_settings=map_settings)
            else:
                # Fallback to single test for non-journal experiments
                self._test_model(model_trainer, test_dataset, seed, fold, fixed_annotators='USE-ALL', test_name='Test all-annotators')

    def _set_seed(self, seed: int) -> None:
        """
        Set random seeds for reproducibility
        
        Args:
            seed: The random seed to use
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        
    def _initialize_model_trainer(self, seed: int, fold: int, annotators: set) -> ModelTrainer:
        """Initialize model trainer (which creates the model internally)"""
        model_name = f'model_prob{self.config.model_config.prob_grid_size}_seed_{seed}_fold_{fold}'
        trainer = ModelTrainer(
            prob_grid_size=self.config.model_config.prob_grid_size,
            model_name=model_name,
            model_config=self.config.model_config,
            model_save_path=self.config.model_save_path,
            training_tasks=self.config.task_str,
            csv_writer=self.csv_writer,
            annotators=annotators
        )

        if (self.config.model_config.model_type == ModelType.INDIVIDUAL_ANNOTATOR_ORTHOGONAL and
            self.config.model_config.orthogonal_model_features and
            'initialised_from_IA' in self.config.model_config.orthogonal_model_features):
            self._initialize_orthogonal_from_ia(trainer, model_name)

        return trainer

    def _initialize_orthogonal_from_ia(self, trainer: ModelTrainer, model_name: str) -> None:
        """Load a trained IA model and use its weights to initialize the orthogonal prediction head."""
        ia_experiment_name = 'individual_annotator_ccc_ind_loss'  # TODO: make configurable
        ia_model_dir = os.path.join(
            os.path.dirname(self.config.model_save_path),
            ia_experiment_name,
            model_name
        )

        if not os.path.exists(ia_model_dir):
            raise FileNotFoundError(f'IA model directory not found: {ia_model_dir}')

        checkpoint_files = [f for f in os.listdir(ia_model_dir)
                            if f.endswith('_cp.ckpt') and 'checkpoint' not in f]
        epoch_numbers = []
        for filename in checkpoint_files:
            try:
                epoch_numbers.append(int(filename.replace('_cp.ckpt', '')))
            except ValueError:
                continue

        if not epoch_numbers:
            raise FileNotFoundError(f'No IA checkpoints found in: {ia_model_dir}')

        best_checkpoint = os.path.join(ia_model_dir, f'{max(epoch_numbers)}_cp.ckpt')
        print(f'[initialised_from_IA] Loading IA model: {best_checkpoint}')

        from MIA.models.utils import load_torch_model
        from MIA.models.models import GenericModel
        ia_model = load_torch_model(GenericModel, best_checkpoint, enforce_load=True)
        with torch.no_grad():
            trainer.model.combined_layers = ia_model.combined_layers
        trainer.model.prediction_head.initialize_from_ia(ia_model.prediction_head)
        trainer.model.pre_training = False
        del ia_model
        
    def _create_data_loader(self, dataset: Dataset, is_training: bool,
                           annotator_mapper: Optional[object] = None, sample_batches_by_annotator: bool = False) -> DataLoader:
        """
        Create appropriate data loader
        
        Args:
            dataset: The dataset to create a loader for
            is_training: Whether this is a training loader
            annotator_mapper: Optional annotator mapping object
            
        Returns:
            Configured DataLoader instance
        """
        batch_size = self.config.training_config.batch_size if is_training else 256
        print('CREATE DATALOADER CALLED WITH: ', sample_batches_by_annotator)
        return create_dataloader(
            dataset,
            batch_size=batch_size,
            model_type=self.config.model_config.model_type,
            annotator_mapper=annotator_mapper,
            is_training=is_training,
            sample_batches_by_annotator=sample_batches_by_annotator and is_training
        )
        
    def _train_model(self, model_trainer: ModelTrainer, 
                     train_dataset: Dataset, val_dataset: Dataset, start_epoch: int = 1) -> None:
        """
        Train the model
        
        Args:
            model_trainer: The model trainer instance
            train_dataset: Training dataset
            val_dataset: Validation dataset
            start_epoch: Start epoch for training
        """
        if getattr(self.config.training_config, 'all_annotators_zeroshot', False):
            raise RuntimeError('Training should not be called for zero-shot all-annotators runs!')
        
        # Get annotator mapper from the model trainer's model
        annotator_mapper = None
        if hasattr(model_trainer.model, 'prediction_head') and hasattr(model_trainer.model.prediction_head, 'act_heads'):
            annotator_mapper = model_trainer.model.prediction_head.act_heads.annotator_mapper
        
        # Setup data loaders
        train_loader = self._create_data_loader(
            train_dataset,
            is_training=True,
            annotator_mapper=annotator_mapper,
            sample_batches_by_annotator=self.config.data_config.sample_batches_by_annotator
        )
        consensus_loader = train_loader if not self.config.data_config.sample_batches_by_annotator else self._create_data_loader(
            train_dataset,
            is_training=True,
            annotator_mapper=annotator_mapper,
            sample_batches_by_annotator=False
        )
        val_loader = self._create_data_loader(
            val_dataset, 
            is_training=False,
            annotator_mapper=annotator_mapper,
            sample_batches_by_annotator=False
        )
        
        # Handle pretraining for individual annotator models
        if (model_trainer.model.args.model_type in [ModelType.INDIVIDUAL_ANNOTATOR, ModelType.INDIVIDUAL_ANNOTATOR_PLUS, ModelType.CLUSTERED, ModelType.INDIVIDUAL_ANNOTATOR_ORTHOGONAL, ModelType.INDIVIDUAL_ANNOTATOR_LOOP] 
            and hasattr(model_trainer.model, 'pre_training') and model_trainer.model.pre_training):
            if self.config.training_config.benchmarking: # If benchmarking skip pretraining as it should have no effect
                print("[PROFILING]: Skipping pretraining for benchmarking")
            else:
                print("Pretraining aggregate annotator prediction head to initialise individual annotator heads")
                
                # Create pretrain loss function
                from MIA.loss_metrics import PreTrainLoss
                model_trainer.pretrain_loss = PreTrainLoss(
                    name='Pretrain CCC loss', 
                    loss_fn='CCC', 
                    train_log_var=(model_trainer.model.args.determinism and 
                                model_trainer.model.args.determinism_type in ['default', 'include_in_output', 'shared_layer', 'utterance_variance'])
                )
                
                # Pretraining epochs
                num_epochs_pre_training = 1 if self.config.training_config.toy_dataset else 5
                # num_epochs_pre_training = 0 # PLACEHOLDER
                for epoch in range(num_epochs_pre_training):
                    print(f'Pretraining epoch {epoch}')
                    if self.config.data_config.pretrain_with_normal_sampling:
                        training_epoch(consensus_loader, model_trainer, epoch)
                    else:
                        training_epoch(train_loader, model_trainer, epoch)
                    validate_epoch(val_loader, model_trainer, self.config.model_config.prob_grid_size, epoch, pretrain=True)

            # Finish pretraining and transition to normal mode
            model_trainer.model.finish_pre_train()
            model_trainer.reset_optim()
            if hasattr(model_trainer, 'pretrain_loss'):
                del model_trainer.pretrain_loss
            print("Finished initialising individual annotator heads")
        else:
            print("Skipping pretraining - not an individual annotator model or already finished pretraining")
        
        max_epochs = self.config.training_config.max_epochs
        if self.config.training_config.toy_dataset:
            max_epochs = 2
            
        print(f"Starting main training loop for {max_epochs} epochs")

        iaes_enabled = getattr(self.config.training_config, 'individual_annotator_early_stopping', False)
        iaes_patience = model_trainer.early_stopping.patience if iaes_enabled else None
        iaes_annotators_still_training = set([annotator for annotators in train_dataset['annotators'] for annotator in annotators])
        # Track activation and valence separately: (best_act_ccc, best_val_ccc, act_bad_epochs, val_bad_epochs, act_best_epoch, val_best_epoch)
        annotator_stats = {}  
        removed_annotators = set()
        annotator_best_epochs = {}  # Track the epoch when each annotator had its best performance for each dimension
        annotator_best_weights = {}  # Store the best weights for each annotator

        benchmark_stats = {}
        import time
        if self.config.training_config.benchmarking:
            print('[PROFILING]: Running one training epoch to account for initialisation time')
            training_epoch(train_loader, model_trainer, 0, (removed_annotators, annotator_best_weights, restore_annotator_weights))

        if not self.config.data_config.training_paradigm_3: # If not training paradigm 3, then consensus loader is not and should be set to None to disable this feature
            consensus_loader = None

        for epoch in range(start_epoch, max_epochs):
            print(f'Training epoch {epoch}')
            if self.config.training_config.benchmarking:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                print('[PROFILING]: Memory reset and cache cleared before training epoch')
                benchmark_stats['start_time_training'] = time.time()
                benchmark_stats['start_memory_training'] = torch.cuda.memory_allocated()
                benchmark_stats['start_memory_peak'] = torch.cuda.max_memory_allocated()
            training_epoch(train_loader, model_trainer, epoch, (removed_annotators, annotator_best_weights, restore_annotator_weights), consensus_dataloader=consensus_loader)
            if self.config.training_config.benchmarking:
                benchmark_stats['end_time_training'] = time.time()
                benchmark_stats['end_memory_training'] = torch.cuda.memory_allocated()
                benchmark_stats['end_memory_peak'] = torch.cuda.max_memory_allocated()
                benchmark_stats['training_time'] = benchmark_stats['end_time_training'] - benchmark_stats['start_time_training']
                benchmark_stats['training_memory'] = benchmark_stats['end_memory_training'] - benchmark_stats['start_memory_training']
                benchmark_stats['training_memory_peak'] = benchmark_stats['end_memory_peak'] - benchmark_stats['start_memory_peak']
                print(f'[PROFILING]: Before Training stats: {benchmark_stats["start_time_training"]} seconds, {benchmark_stats["start_memory_training"]} bytes usage, {benchmark_stats["start_memory_peak"]} bytes peak')
                print('--------------------------------')
                print(f'[PROFILING]: After Training stats: {benchmark_stats["end_time_training"]} seconds, {benchmark_stats["end_memory_training"]} bytes usage, {benchmark_stats["end_memory_peak"]} bytes peak')
                print('--------------------------------')
                print(f'[PROFILING]: Training time: {benchmark_stats["training_time"]} seconds')
                print(f'[PROFILING]: Training memory: {benchmark_stats["training_memory"]} bytes')
                print(f'[PROFILING]: Training memory peak: {benchmark_stats["training_memory_peak"]} bytes')
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                print('[PROFILING]: Memory reset and cache cleared')
                benchmark_stats['start_time_validation'] = time.time()
                benchmark_stats['start_memory_validation'] = torch.cuda.memory_allocated()
                benchmark_stats['start_memory_peak_validation'] = torch.cuda.max_memory_allocated()
            results, continue_training = validate_epoch(
                val_loader,
                model_trainer,
                self.config.model_config.prob_grid_size,
                epoch,
                benchmarking=self.config.training_config.benchmarking
            )

            if iaes_enabled and not continue_training: # Temporary test, once model is fully trained, now perform individual annotator early stopping. LR is not reset so may be things to look into further, but this is just a quick test.
                model_trainer.model.audio_layers.freeze()
                model_trainer.model.text_layers.freeze()
                model_trainer.model.combined_layers.freeze()
                model_trainer.model.prediction_head.act_combined_layers.freeze()
                model_trainer.model.prediction_head.val_combined_layers.freeze()
                FULLY_REMOVE_ANNOTATORS = True # Temporary test, should annotators still be used to update base model while repeatedly restoring their prediction weights?
                cccs_separate = compute_annotator_ccc_separate(results, removed_annotators)
                annotators_to_remove = set()
                for ann, cccs in cccs_separate.items():
                    if ann in removed_annotators:
                        print(f'Sanity check, {ann} is in removed_annotators with values {cccs}')
                        if FULLY_REMOVE_ANNOTATORS: # Output and continue if removed from training set, if not we need to restore weights every epoch
                            continue
                    
                    if ann not in removed_annotators:
                        # Get current CCC values
                        act_ccc = cccs['activation']
                        val_ccc = cccs['valence']
                        
                        # Get previous best values and bad epoch counts
                        # Format: (best_act_ccc, best_val_ccc, act_bad_epochs, val_bad_epochs, act_best_epoch, val_best_epoch)
                        prev_stats = annotator_stats.get(ann, (-float('inf'), -float('inf'), 0, 0, -1, -1))
                        best_act, best_val, act_bad, val_bad, act_best_epoch, val_best_epoch = prev_stats
                        
                        # Check if activation improved
                        act_improved = act_ccc > best_act
                        if act_improved:
                            best_act = act_ccc
                            act_bad = 0
                            act_best_epoch = epoch
                            print(f'Annotator {ann} achieved new best activation CCC: {act_ccc.item():.4f} at epoch {epoch}')
                            # Save the current weights for this annotator (activation improved)
                            save_annotator_weights(model_trainer.model, ann, annotator_best_weights)
                        else:
                            act_bad += 1
                        
                        # Check if valence improved
                        val_improved = val_ccc > best_val
                        if val_improved:
                            best_val = val_ccc
                            val_bad = 0
                            val_best_epoch = epoch
                            print(f'Annotator {ann} achieved new best valence CCC: {val_ccc.item():.4f} at epoch {epoch}')
                            # Save the current weights for this annotator (valence improved)
                            save_annotator_weights(model_trainer.model, ann, annotator_best_weights)
                        else:
                            val_bad += 1
                        
                        # Update stats
                        annotator_stats[ann] = (best_act, best_val, act_bad, val_bad, act_best_epoch, val_best_epoch)

                    # Only stop training this annotator if BOTH activation and valence have stopped improving
                    if (ann in removed_annotators) or (act_bad >= iaes_patience and val_bad >= iaes_patience and ann not in removed_annotators):
                        print(f'Annotator {ann} triggered early stopping (activation: {act_bad} epochs without improvement, valence: {val_bad} epochs without improvement). Restoring best weights from epochs {act_best_epoch} (act) and {val_best_epoch} (val).')
                        # Restore the best weights for this annotator
                        restore_annotator_weights(model_trainer.model, ann, annotator_best_weights)
                        removed_annotators.add(ann)
                        annotators_to_remove.add(ann)

                if len(annotators_to_remove) and FULLY_REMOVE_ANNOTATORS:
                    iaes_annotators_still_training = iaes_annotators_still_training - annotators_to_remove

                    # Define function to filter annotators to only keep annotators that are still training
                    empty_set = set()
                    filter_fn = lambda x, y, z: replace_annotators_batch({'annotators': x, 'soft_act_labels': y, 'soft_val_labels': z}, iaes_annotators_still_training, empty_set)

                    # Store original format
                    original_format = train_dataset.format

                    # First calculate indices that will have no changes, there is no point running the map on these 
                    unchanged_indices = [i for i, anns in tqdm(enumerate(train_dataset['annotators']), desc='Calculating indices that need annotators removing', total=len(train_dataset)) if all(a in iaes_annotators_still_training for a in anns)]
                    changed_indices = list(set(range(len(train_dataset))) - set(unchanged_indices))

                    if len(changed_indices):
                        # Apply filtering to rows which require annotators to be removed
                        changed_rows = train_dataset.select(changed_indices).map(filter_fn, input_columns=['annotators', 'soft_act_labels', 'soft_val_labels'], batched=True, batch_size=5000)

                        # Calculate which changed rows have been left with no annotators
                        lengths = np.array([len(a) for a in changed_rows['annotators']])
                        keep_idx = np.where(lengths > 0)[0]
                        changed_rows = changed_rows.select(keep_idx)
                        print(f'{len(keep_idx)} rows changed, {len(changed_indices)-len(keep_idx)} rows removed')
                        # temp = temp.filter(has_annotators)
                        if len(unchanged_indices):
                            # No need to preserve ordering as train dataset goes into dataloader with shuffling
                            unchanged_dataset = train_dataset.select(unchanged_indices)#.add_column('orig_index', unchanged_indices)
                            # temp = temp.add_column('orig_index', changed_indices)

                            merged_dataset = concatenate_datasets([unchanged_dataset, changed_rows])
                            # merged_dataset = merged_dataset.sort('orig_index')
                            # temp = merged_dataset.remove_column('orig_index')
                            changed_rows = merged_dataset
                        train_dataset = changed_rows

                if len(train_dataset) == 0 or len(iaes_annotators_still_training) == 0:
                    print('Training dataset is empty or no more training annotators after removing annotators.', f'({len(iaes_annotators_still_training)} annotators still training.)')
                    continue_training = False
                    guaranteed_save_weights = min(model_trainer.early_stopping.previous_results)-1 if model_trainer.early_stopping.mode == 'min' else max(model_trainer.early_stopping.previous_results)+1
                    model_trainer.early_stopping.log_value(guaranteed_save_weights, model_trainer.model)
                else:
                    # Override normal early stopping and continue training until all individual annotators are removed
                    continue_training = True

                    # Restore the original format to ensure tensors remain tensors
                    train_dataset.set_format(**original_format)

                    train_loader = self._create_data_loader(
                        train_dataset,
                        is_training=True,
                        annotator_mapper=annotator_mapper
                    )

            if epoch == 1 and hasattr(self.config.training_config, 'finetune_on') and self.config.training_config.finetune_on:
                model_trainer.save_special_checkpoint('one-shot')
            # break # PLACEHOLDER
            if not continue_training:
                break
                
    def _get_base_model_path(self, seed: int, fold: int, return_attempted_path: bool = False) -> str:
        """
        Get the path to the base model for finetuning or pretraining test_only mode
        
        Args:
            seed: Random seed
            fold: Fold number
            return_attempted_path: If True, return the path even if it doesn't exist
            
        Returns:
            Path to the base model directory, or None if it doesn't exist and return_attempted_path=False
        """
        # Construct base model path (without finetune suffix and self-report suffix)
        base_experiment_name = self.config.experiment_name
        
        if hasattr(self.config.training_config, 'finetune_on') and self.config.training_config.finetune_on:
            # With the new naming format, ft_on_dataset comes first, so we can simply split on it
            finetune_suffix = f"_ft_on_{self.config.training_config.finetune_on}"
            if finetune_suffix in base_experiment_name:
                # Everything before _ft_on_dataset is the base experiment name
                base_experiment_name = base_experiment_name.split(finetune_suffix)[0]
                    
            print('For finetuning, the base model has no folds so setting fold to 0')
            fold = 0

            # Remove _closest_label suffix for self-report experiments
            base_experiment_name = base_experiment_name.replace('_closest_label', '')
            base_experiment_name = base_experiment_name.replace('_naive', '')
            base_experiment_name = base_experiment_name.replace('_only_self_report', '')
            base_experiment_name = base_experiment_name.replace('_softmax_ccc', '')
            base_experiment_name = base_experiment_name.replace('_weighted_ccc', '')
            base_experiment_name = base_experiment_name.replace('_all_pearson', '')
            base_experiment_name = base_experiment_name.replace('_monologue_level_map', '')
            base_experiment_name = base_experiment_name.replace('_ccc_ind_loss', '')
            base_experiment_name = base_experiment_name.replace('_CORAL', '')
            print(f'Removed _closest_label etc. type suffixes: {self.config.experiment_name} -> {base_experiment_name}')


        # Construct path to base model
        model_name = f'model_prob{self.config.model_config.prob_grid_size}_seed_{seed}_fold_{fold}'
        
        # The base model path needs to remove both the experiment name AND any _ft_on_dataset suffix from the directory structure
        base_model_path = self.config.model_save_path.replace(self.config.experiment_name, base_experiment_name)
        
        # Also need to remove _ft_on_dataset from the directory path structure if it exists
        if hasattr(self.config.training_config, 'finetune_on') and self.config.training_config.finetune_on:
            ft_suffix = f"_ft_on_{self.config.training_config.finetune_on}"
            if ft_suffix in base_model_path:
                base_model_path = base_model_path.replace(ft_suffix, "")
        
        full_model_path = f'{base_model_path}/{model_name}'
        
        print(f'Looking for base model at: {full_model_path}')
        print(f'  Current experiment: {self.config.experiment_name}')
        print(f'  Base experiment: {base_experiment_name}')
        print(f'  Base model path: {base_model_path}')
        
        # Check if the base model directory exists
        if os.path.exists(full_model_path):
            print(f'Found base model at: {full_model_path}')
            return full_model_path
        else:
            if return_attempted_path:
                return full_model_path
            print(f'Warning: Base model path does not exist: {full_model_path}')
            return None
