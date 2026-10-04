#!/usr/bin/env python3
"""
Benchmark data collation and model performance for the journal figures.

Two experiments provide the CSV inputs used by ``compare_models.ipynb``:
1. Natural datasets (full, 100+, 1000+)
2. Subsampled annotators (5k-30k from full dataset)

Usage:
    python benchmarking/benchmark_collation_models.py --experiment 1
    python benchmarking/benchmark_collation_models.py --experiment 2
"""

import os
import sys
import time
import argparse
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import gc

# Make the local package importable without relying on a machine-specific path.
BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from MIA.config import ModelType, ModelConfig, ExperimentConfig, TrainingConfig, DataConfig, ExperimentType
from MIA.models import GenericModel
from MIA.training import BatchCollator
from MIA.datamanager import DatasetManager
from torch.utils.data import DataLoader
from datasets import concatenate_datasets


# ==============================================================================
# CONFIGURATION
# ==============================================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
NUM_WARMUP_BATCHES = 20
NUM_BENCHMARK_BATCHES = 200
OUTPUT_DIR = BENCHMARK_DIR
SEED = 42

MODEL_TYPES = [
    ModelType.BASELINE_AGGREGATE,
    ModelType.INDIVIDUAL_ANNOTATOR,
    ModelType.INDIVIDUAL_ANNOTATOR_LOOP,
]

MODEL_DEFAULTS = {
    'dense_layers': 2,
    'layer_size': 256,
    'prob_grid_size': 4,
}

# Define which collator types each model should test
# BASELINE_AGGREGATE: without mapper (doesn't use individual annotators)
# INDIVIDUAL_ANNOTATOR: with mapper (optimized collation)
# INDIVIDUAL_ANNOTATOR_LOOP: with mapper (optimized collation)
MODEL_COLLATOR_CONFIGS = {
    ModelType.BASELINE_AGGREGATE: [False],  # Only test without mapper
    ModelType.INDIVIDUAL_ANNOTATOR: [True],  # Only test with mapper
    ModelType.INDIVIDUAL_ANNOTATOR_LOOP: [True],  # Only test with mapper
}


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def reset_memory():
    """Reset GPU memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


def get_memory_mb() -> float:
    """Get peak GPU memory in MiB since last reset."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def get_current_memory_mb() -> float:
    """Get currently allocated GPU memory in MiB."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def count_batch_annotators(batch: dict) -> Tuple[int, int]:
    """
    Count annotators in a batch.
    
    Returns:
        (num_unique_annotators, total_annotator_count)
    """
    annotators_list = batch['annotators']
    unique = set()
    total = 0
    
    for sample_annotators in annotators_list:
        if isinstance(sample_annotators, list):
            unique.update(sample_annotators)
            total += len(sample_annotators)
        else:
            unique.add(sample_annotators)
            total += 1
    
    return len(unique), total


def get_dataset_stats(dataset) -> Dict:
    """Get statistics about a dataset."""
    all_annotators = set()
    annotations_per_sample = []
    
    for sample_annotators in tqdm(dataset['annotators'], desc="Analyzing dataset", leave=True):
        if isinstance(sample_annotators, list):
            all_annotators.update(str(a) for a in sample_annotators)
            annotations_per_sample.append(len(sample_annotators))
        else:
            raise ValueError(f"Sample annotators is not a list: {sample_annotators}")
            # all_annotators.add(str(sample_annotators))
            # annotations_per_sample.append(1)
    
    return {
        'total_annotators': len(all_annotators),
        'avg_annotations_per_sample': np.mean(annotations_per_sample),
        'dataset_size': len(dataset),
        'annotators': all_annotators,
    }


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_dataset(min_annotations: Optional[int] = None):
    """Load and combine train/val/test datasets."""
    model_config = ModelConfig(
        model_type=ModelType.INDIVIDUAL_ANNOTATOR,
        prob_grid_size=4,
        determinism=False,
    )
    
    training_config = TrainingConfig(
        batch_size=BATCH_SIZE,
        max_epochs=1,
        toy_dataset=False,
        test_only=True,
        number_of_seeds=1,
        podcast_version='1.11',
    )
    
    data_config = DataConfig(
        regenerate_kde_labels=False,
        min_annotations=min_annotations,
        cluster_other_annotators=False,
    )
    
    exp_config = ExperimentConfig(
        experiment_name='benchmarking',
        experiment_type=ExperimentType.JOURNAL,
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        log_path='/tmp/benchmark_logs',
        model_save_path='/tmp/benchmark_models',
        csv_results_path='/tmp/benchmark_csv',
    )
    
    dataset_manager = DatasetManager(exp_config)
    datasets = dataset_manager.prepare_all_datasets(seed=SEED)
    
    # Unpack and ensure single datasets
    train_dataset, val_dataset, test_dataset = datasets
    if isinstance(train_dataset, list):
        train_dataset = train_dataset[0]
    if isinstance(val_dataset, list):
        val_dataset = val_dataset[0]
    if isinstance(test_dataset, list):
        test_dataset = test_dataset[0]
    
    # Apply filters if needed
    if min_annotations is not None:
        train_dataset, val_dataset, test_dataset = dataset_manager.apply_filters(
            train_dataset, val_dataset, test_dataset
        )
        if isinstance(train_dataset, list):
            train_dataset = train_dataset[0]
        if isinstance(val_dataset, list):
            val_dataset = val_dataset[0]
        if isinstance(test_dataset, list):
            test_dataset = test_dataset[0]
    
    # Combine all splits
    combined = concatenate_datasets([train_dataset, val_dataset, test_dataset])
    return combined


def subsample_annotators(dataset, num_annotators: int, all_annotators: set):
    """
    Subsample or duplicate annotators to reach target count.
    
    If num_annotators <= len(all_annotators): subsample
    If num_annotators > len(all_annotators): duplicate annotators with suffixes
    """
    rng = np.random.RandomState(SEED)
    all_annotators_list = sorted(list(all_annotators))  # Sort for reproducibility
    
    if num_annotators <= len(all_annotators):
        # Subsample: select random subset
        selected_annotators = set(rng.choice(all_annotators_list, size=num_annotators, replace=False))
        
        # Store original format to preserve tensor types (pattern from dataset_utils.py)
        original_format = dataset.format
        
        # Filter annotators using non-batched map (simpler and matches existing codebase pattern)
        def filter_annotators(sample):
            sample_annotators = sample['annotators']
            
            if isinstance(sample_annotators, list):
                # Keep only selected annotators
                indices_to_keep = [i for i, ann in enumerate(sample_annotators) if str(ann) in selected_annotators]
                
                if indices_to_keep:
                    sample['annotators'] = [sample_annotators[i] for i in indices_to_keep]
                    
                    # Filter individual annotation values (soft labels), not aggregate act/val
                    if 'soft_act_labels' in sample:
                        soft_act = sample['soft_act_labels']
                        soft_val = sample['soft_val_labels']
                        sample['soft_act_labels'] = soft_act[indices_to_keep]
                        sample['soft_val_labels'] = soft_val[indices_to_keep]
                    
                    sample['_keep'] = True
                else:
                    sample['_keep'] = False
            else:
                # Single annotator
                sample['_keep'] = str(sample_annotators) in selected_annotators
            
            return sample
        
        # Apply transformation and filter
        print("      Filtering samples...")
        transformed = dataset.map(filter_annotators, desc="      Processing")
        filtered = transformed.filter(lambda x: x['_keep'])
        filtered = filtered.remove_columns(['_keep'])
        
        # Restore original format to ensure tensors remain tensors
        filtered.set_format(**original_format)
        
        return filtered
    
    else:
        # Duplicate: create copies of annotators by adding them to each sample
        print(f"      Creating {num_annotators} annotators by expanding from {len(all_annotators)} originals...")
        print(f"      Note: This adds annotator copies to samples without duplicating samples themselves")
        
        # Calculate how many copies we need
        num_full_copies = num_annotators // len(all_annotators)
        num_partial = num_annotators % len(all_annotators)
        
        # Determine which annotators get extra copy (avoid alphabetical bias)
        extra_annotators = set()
        if num_partial > 0:
            shuffled_annotators = all_annotators_list.copy()
            rng.shuffle(shuffled_annotators)
            extra_annotators = set(shuffled_annotators[:num_partial])
        
        # Store original format
        original_format = dataset.format
        
        # Expand each sample's annotators
        def expand_annotators(sample):
            import torch
            original_annotators = sample['annotators']
            original_soft_act = sample.get('soft_act_labels')
            original_soft_val = sample.get('soft_val_labels')
            
            # Convert soft labels to list format for processing
            def tensor_to_list(val):
                if val is None:
                    return []
                if hasattr(val, 'tolist'):
                    return val.tolist()
                if isinstance(val, list):
                    return val
                return []
            
            original_soft_act = tensor_to_list(original_soft_act)
            original_soft_val = tensor_to_list(original_soft_val)
            
            new_annotators = []
            new_soft_act = []
            new_soft_val = []
            
            if isinstance(original_annotators, list):
                # For each original annotator, create copies
                for idx, ann in enumerate(original_annotators):
                    ann_str = str(ann)
                    
                    # Add original
                    new_annotators.append(ann)
                    if idx < len(original_soft_act):
                        new_soft_act.append(original_soft_act[idx])
                        new_soft_val.append(original_soft_val[idx])
                    
                    # Add full copies
                    for copy_idx in range(1, num_full_copies):
                        new_annotators.append(f"{ann}_copy{copy_idx}")
                        if idx < len(original_soft_act):
                            new_soft_act.append(original_soft_act[idx])
                            new_soft_val.append(original_soft_val[idx])
                    
                    # Add partial copy if this annotator is selected
                    if ann_str in extra_annotators:
                        new_annotators.append(f"{ann}_copy{num_full_copies}")
                        if idx < len(original_soft_act):
                            new_soft_act.append(original_soft_act[idx])
                            new_soft_val.append(original_soft_val[idx])
            else:
                # Single annotator
                ann_str = str(original_annotators)
                new_annotators.append(original_annotators)
                if len(original_soft_act) > 0:
                    new_soft_act.append(original_soft_act[0])
                    new_soft_val.append(original_soft_val[0])
                
                for copy_idx in range(1, num_full_copies):
                    new_annotators.append(f"{original_annotators}_copy{copy_idx}")
                    if len(original_soft_act) > 0:
                        new_soft_act.append(original_soft_act[0])
                        new_soft_val.append(original_soft_val[0])
                
                if ann_str in extra_annotators:
                    new_annotators.append(f"{original_annotators}_copy{num_full_copies}")
                    if len(original_soft_act) > 0:
                        new_soft_act.append(original_soft_act[0])
                        new_soft_val.append(original_soft_val[0])
            
            sample['annotators'] = new_annotators
            # Update soft labels (individual annotations), keep act/val (aggregate) unchanged
            if new_soft_act:
                sample['soft_act_labels'] = torch.tensor(new_soft_act) if isinstance(sample.get('soft_act_labels'), torch.Tensor) else new_soft_act
                sample['soft_val_labels'] = torch.tensor(new_soft_val) if isinstance(sample.get('soft_val_labels'), torch.Tensor) else new_soft_val
            
            return sample
        
        # Apply transformation
        expanded = dataset.map(expand_annotators, desc="      Expanding annotators")
        expanded.set_format(**original_format)
        
        return expanded


# ==============================================================================
# MODEL CREATION
# ==============================================================================

def create_model(model_type: ModelType, annotators: set) -> GenericModel:
    """Create and initialize a model."""
    model_config = ModelConfig(
        model_type=model_type,
        prob_grid_size=MODEL_DEFAULTS['prob_grid_size'],
        dense_layers=MODEL_DEFAULTS['dense_layers'],
        layer_size=MODEL_DEFAULTS['layer_size'],
        training_precision=torch.float32,
        determinism=False,
    )
    
    model = GenericModel(model_config=model_config, annotators=annotators)
    
    if hasattr(model, 'pre_training'):
        model.finish_pre_train()
    
    model = model.to(DEVICE)
    model.eval()
    return model


def create_collator(model: GenericModel, model_type: ModelType, use_annotator_mapper: bool) -> BatchCollator:
    """Create batch collator."""
    if use_annotator_mapper and model_type in [
        ModelType.INDIVIDUAL_ANNOTATOR,
        ModelType.INDIVIDUAL_ANNOTATOR_PLUS,
        ModelType.INDIVIDUAL_ANNOTATOR_LOOP,
    ]:
        annotator_mapper = model.prediction_head.act_heads.annotator_mapper
        return BatchCollator(annotator_mapper=annotator_mapper, include_aggregate=False)
    else:
        return BatchCollator()


# ==============================================================================
# CORE BENCHMARKING
# ==============================================================================

def benchmark_forward_pass_on_samples(
    model: GenericModel,
    sample_batches: List[List],
    collator: BatchCollator,
    model_type: ModelType,
) -> List[Dict]:
    """
    Benchmark model forward pass on pre-selected samples.
    
    Args:
        model: Model to benchmark
        sample_batches: List of batches, where each batch is a list of raw samples
        collator: Collator to use for this model
        model_type: Type of model
    
    Returns list of per-batch results.
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Collate samples with model-specific collator and track timing/memory
    print("      Collating batches...")
    all_batches = []
    collation_times = []
    collation_memory_peaks = []
    
    for samples in tqdm(sample_batches, desc="      Collating", leave=True):
        # Track collation time and memory
        reset_memory()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        batch = collator(samples)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        
        collation_time = (end - start) * 1000  # ms
        collation_memory = get_memory_mb()
        
        all_batches.append(batch)
        collation_times.append(collation_time)
        collation_memory_peaks.append(collation_memory)
    
    # Warmup
    print(f"      Warmup ({NUM_WARMUP_BATCHES} batches)...")
    with torch.no_grad():
        for batch in tqdm(all_batches[:NUM_WARMUP_BATCHES], desc="      Warmup", leave=True):
            audio = batch['audio'].to(device)
            text = batch['transcript'].to(device)
            annotator_masks = batch.get('annotator_masks')
            if annotator_masks is not None:
                annotator_masks = tuple(m.to(device) if isinstance(m, torch.Tensor) else m for m in annotator_masks)
            _ = model(audio, text, skip_kde=True, annotator_masks=annotator_masks)
    
    # Benchmark
    benchmark_batches = all_batches[NUM_WARMUP_BATCHES:NUM_WARMUP_BATCHES + NUM_BENCHMARK_BATCHES]
    print(f"      Benchmarking ({len(benchmark_batches)} batches)...")
    
    results = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(benchmark_batches, desc="      Benchmarking", leave=True)):
            # Count annotators
            num_unique, num_total = count_batch_annotators(batch)
            
            # Reset memory tracking BEFORE moving data to device
            reset_memory()
            
            # Prepare batch
            audio = batch['audio'].to(device)
            text = batch['transcript'].to(device)
            annotator_masks = batch.get('annotator_masks')
            if annotator_masks is not None:
                annotator_masks = tuple(m.to(device) if isinstance(m, torch.Tensor) else m for m in annotator_masks)
            
            # Time forward pass
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = model(audio, text, skip_kde=True, annotator_masks=annotator_masks)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            
            # Get memory stats (in MiB)
            # Peak: maximum allocated during batch (even if later freed)
            # Current: still allocated after forward pass completes
            peak_memory = get_memory_mb()
            current_memory = get_current_memory_mb()
            
            # Get collation metrics for this batch (from warmup + benchmark index)
            actual_batch_idx = NUM_WARMUP_BATCHES + batch_idx
            collation_time = collation_times[actual_batch_idx]
            collation_memory = collation_memory_peaks[actual_batch_idx]
            
            # Record results
            results.append({
                'batch_idx': batch_idx,
                'batch_size': len(batch['audio']),
                'unique_annotators': num_unique,
                'total_annotators': num_total,
                'collation_time_ms': collation_time,
                'collation_memory_peak_mb': collation_memory,
                'forward_time_ms': (end - start) * 1000,
                'forward_memory_peak_mb': peak_memory,
                'forward_memory_current_mb': current_memory,
            })
    
    return results


def load_samples_once(dataset, config_name: str, shuffle: bool = True) -> List[List]:
    """
    Load sample indices once for a dataset configuration.
    Returns batches of raw sample indices that can be collated differently for each model.
    
    Args:
        dataset: Dataset to load from
        config_name: Name for logging
        shuffle: Whether to shuffle samples (False for synthetic datasets to preserve batch structure)
    """
    print(f"    Selecting samples for {config_name}...")
    
    # Create dataloader
    indices_dataloader = DataLoader(
        range(len(dataset)),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        generator=torch.Generator().manual_seed(SEED) if shuffle else None,
    )
    
    # Load the exact number of batches needed
    total_needed_batches = NUM_WARMUP_BATCHES + NUM_BENCHMARK_BATCHES
    sample_batches = []
    for batch_indices in tqdm(indices_dataloader, desc="      Selecting", leave=True, total=total_needed_batches):
        # Convert to list and get actual samples
        indices = batch_indices.tolist() if isinstance(batch_indices, torch.Tensor) else list(batch_indices)
        samples = [dataset[i] for i in indices]
        sample_batches.append(samples)
        if len(sample_batches) >= total_needed_batches:
            break
    
    print(f"    Selected {len(sample_batches)} batches of samples")
    return sample_batches


def run_benchmark_config(
    model_type: ModelType,
    sample_batches: List[List],
    dataset_stats: Dict,
    use_annotator_mapper: bool,
    config_name: str,
) -> pd.DataFrame:
    """Run a single benchmark configuration using pre-selected samples."""
    print(f"\n    Model: {model_type.name}")
    print(f"    Collator: {'WITH' if use_annotator_mapper else 'WITHOUT'} annotator_mapper")
    print(f"    Config: {config_name}")
    
    # Create model
    model = create_model(model_type, dataset_stats['annotators'])
    model_type_actual = model.args.model_type
    
    # Create model-specific collator
    collator = create_collator(model, model_type_actual, use_annotator_mapper)
    
    # Benchmark on the pre-selected samples
    results = benchmark_forward_pass_on_samples(model, sample_batches, collator, model_type_actual)
    
    # Add metadata
    for result in results:
        result.update({
            'model_type': model_type.name,
            'collator_type': 'with_mapper' if use_annotator_mapper else 'without_mapper',
            'config_name': config_name,
            'dataset_total_annotators': dataset_stats['total_annotators'],
            'dataset_avg_annotations_per_sample': dataset_stats['avg_annotations_per_sample'],
            'dataset_size': dataset_stats['dataset_size'],
        })
    
    # Cleanup
    del model
    reset_memory()
    
    return pd.DataFrame(results)


# ==============================================================================
# EXPERIMENTS
# ==============================================================================

def experiment_1_natural_datasets():
    """
    Experiment 1: Natural datasets (full, 100+, 1000+)
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: Natural Datasets")
    print("="*70)
    
    variants = [
        ('full', None),
        ('100plus', 100),
        ('1000plus', 1000),
    ]
    
    all_results = []
    
    for variant_name, min_annotations in variants:
        print(f"\n  Loading dataset: {variant_name}...")
        dataset = load_dataset(min_annotations)
        print(f"    Computing dataset statistics...")
        stats = get_dataset_stats(dataset)
        
        print(f"    Size: {stats['dataset_size']}")
        print(f"    Total annotators: {stats['total_annotators']}")
        print(f"    Avg annotations/sample: {stats['avg_annotations_per_sample']:.2f}")
        
        # Load samples once (same for all models)
        sample_batches = load_samples_once(dataset, variant_name)
        
        # Benchmark all combinations on the same samples
        for model_type in MODEL_TYPES:
            for use_mapper in MODEL_COLLATOR_CONFIGS[model_type]:
                try:
                    df = run_benchmark_config(
                        model_type, sample_batches, stats, use_mapper, variant_name
                    )
                    all_results.append(df)
                except Exception as e:
                    print(f"    ERROR: {e}")
    
    # Save results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        output_path = os.path.join(OUTPUT_DIR, 'experiment_1_natural_datasets.csv')
        combined_df.to_csv(output_path, index=False)
        print(f"\n  Saved: {output_path}")
        print(f"  Total rows: {len(combined_df)}")


def experiment_2_subsampled_annotators():
    """
    Experiment 2: Subsample annotators from full dataset (5k-30k)
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: Subsampled Annotators")
    print("="*70)
    
    # Load full dataset
    print("\n  Loading full dataset...")
    full_dataset = load_dataset(None)
    print(f"    Computing dataset statistics...")
    full_stats = get_dataset_stats(full_dataset)
    
    print(f"    Size: {full_stats['dataset_size']}")
    print(f"    Total annotators: {full_stats['total_annotators']}")
    
    # Subsample different annotator counts
    annotator_counts = [5000, 10000, 15000, 20000, 25000, 30000]
    
    all_results = []
    
    for num_annotators in annotator_counts:
        if num_annotators <= full_stats['total_annotators']:
            print(f"\n  Subsampling {num_annotators} annotators...")
        else:
            print(f"\n  Creating {num_annotators} annotators (duplicating from {full_stats['total_annotators']})...")
        
        subsampled_dataset = subsample_annotators(
            full_dataset, num_annotators, full_stats['annotators']
        )
        print(f"    Computing dataset statistics...")
        stats = get_dataset_stats(subsampled_dataset)
        
        # For subsampled/expanded datasets, use the target count
        # (get_dataset_stats would count duplicated annotator IDs which is not what we want for the metric)
        stats['total_annotators'] = num_annotators
        
        print(f"    Resulting size: {stats['dataset_size']}")
        print(f"    Total annotators (target): {stats['total_annotators']}")
        
        config_name = f"subsample_{num_annotators}"
        
        # Load samples once (same for all models)
        sample_batches = load_samples_once(subsampled_dataset, config_name)
        
        # Benchmark all combinations on the same samples
        for model_type in MODEL_TYPES:
            for use_mapper in MODEL_COLLATOR_CONFIGS[model_type]:
                try:
                    df = run_benchmark_config(
                        model_type, sample_batches, stats, use_mapper, config_name
                    )
                    all_results.append(df)
                except Exception as e:
                    print(f"    ERROR: {e}")
    
    # Save results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        output_path = os.path.join(OUTPUT_DIR, 'experiment_2_subsampled_annotators.csv')
        combined_df.to_csv(output_path, index=False)
        print(f"\n  Saved: {output_path}")
        print(f"  Total rows: {len(combined_df)}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    global OUTPUT_DIR
    
    parser = argparse.ArgumentParser(description='Benchmark collation and model performance')
    parser.add_argument('--experiment', type=int, choices=[1, 2], default=None,
                        help='Which experiment to run (1 or 2). If omitted, runs both.')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Set output directory
    OUTPUT_DIR = Path(args.output_dir).resolve()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*70)
    print("BENCHMARKING SCRIPT")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Warmup batches: {NUM_WARMUP_BATCHES}")
    print(f"Benchmark batches: {NUM_BENCHMARK_BATCHES}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Models: {[m.name for m in MODEL_TYPES]}")
    print("="*70)
    
    # Run experiments
    if args.experiment is None:
        experiment_1_natural_datasets()
        experiment_2_subsampled_annotators()
    elif args.experiment == 1:
        experiment_1_natural_datasets()
    elif args.experiment == 2:
        experiment_2_subsampled_annotators()
    
    print("\n" + "="*70)
    print("BENCHMARKING COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
