import torch
import numpy as np
from tqdm import tqdm
from typing import List, Set, Dict, Tuple, Optional

def calculate_ordinal_values(dataset):
    """Calculate ordinal values for activation and valence from dataset."""
    all_annotators = set([x for samples in dataset['annotators'] for x in samples])
    annotator_map = {annotator: i for i, annotator in enumerate(all_annotators)}
    all_act = torch.full((len(dataset), len(annotator_map)), fill_value=torch.nan)
    all_val = torch.full((len(dataset), len(annotator_map)), fill_value=torch.nan)
    
    for i, batch in tqdm(enumerate(dataset), total=len(dataset), desc='Calculating ordinal values'):
        soft_act = batch['soft_act_labels'].to('cpu', non_blocking=True)
        soft_val = batch['soft_val_labels'].to('cpu', non_blocking=True)
        annotator_idxs = [annotator_map[x] for x in batch['annotators']]
        all_act[i, annotator_idxs] = soft_act
        all_val[i, annotator_idxs] = soft_val
        
    ordinal_act = 3*all_act + 4
    ordinal_val = 3*all_val + 4
    return ordinal_act, ordinal_val, annotator_map

def select_good_labels(train_annotators: Set[str], x: Dict) -> bool:
    """Check if any training annotators are present in the sample."""
    return any(annotator in train_annotators for annotator in x['annotators'])

def replace_annotators(row: Dict, annotators_to_keep: List[str], annotators_to_group: List[str], replace_act_val: bool = False) -> Dict:
    """Replace annotators in a row with kept and grouped annotators."""
    preserve = np.isin(np.array(row['annotators']), annotators_to_keep)
        
    new_annotators = np.array(row['annotators'])[preserve]
    new_soft_act = row['soft_act_labels'][preserve]
    new_soft_val = row['soft_val_labels'][preserve]
    
    if len(annotators_to_group):
        raise NotImplementedError("Rest-annotator is no longer supported")

    row['annotators'] = new_annotators.tolist()  # Convert numpy array back to list
    row['soft_act_labels'] = new_soft_act
    row['soft_val_labels'] = new_soft_val
    # We don't touch act/val/variances as our model may only explicitly model annotators that labeled a certain amount of samples
    # but this doesn't mean that the information from all annotators is not useful.
    # row['act'] = new_act
    # row['val'] = new_val
    if replace_act_val:
        row['act'] = new_soft_act.mean()
        row['val'] = new_soft_val.mean()
    # row['act_variance'] = new_soft_act.var()
    # row['val_variance'] = new_soft_val.var()
    
    return row

# Faster batched version for IAES annotator replacement during training
def replace_annotators_batch(batch, annotators_to_keep, annotators_to_group):
    if isinstance(annotators_to_keep, (np.ndarray, list)):
        annotators_to_keep = set(annotators_to_keep)
    if isinstance(annotators_to_group, (np.ndarray, list)):
        annotators_to_group = set(annotators_to_group)
    
    if len(annotators_to_group):
        raise NotImplementedError("Rest-annotator is no longer supported")
    
    new_annotators, new_soft_act, new_soft_val = [], [], []

    for anns, acts, vals in zip(batch["annotators"], batch["soft_act_labels"], batch["soft_val_labels"]):
        anns = np.array(anns)
        preserve = np.array([a in annotators_to_keep for a in anns])
        
        new_annotators.append(anns[preserve])
        new_soft_act.append(acts[preserve])
        new_soft_val.append(vals[preserve])

    return {
        "annotators": new_annotators,
        "soft_act_labels": new_soft_act,
        "soft_val_labels": new_soft_val,
    }

def get_filtered_annotators(filter_settings: Dict, all_annotators: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Get filtered annotators based on settings."""
    all_anns, annotation_counts = np.unique(all_annotators, return_counts=True)
    keep_annotators = all_anns[annotation_counts >= filter_settings['min']]
    group_annotators = [] if not filter_settings['rest_annotator'] else all_anns[annotation_counts < filter_settings['min']]
    return keep_annotators, group_annotators

def has_annotators(sample: Dict) -> bool:
    """Check if sample has any annotators - used for filtering."""
    return len(sample['annotators']) > 0

# All_annotators can be optionally provided
# if we want to filter the dataset for min_annotations on this dataset, then all_annotator can be none
# if instead we want to filter the dataset for min_annotations across annotations on multiple data splits, etc. then all_annotators should be provided
# if provided all_annotators is effectively a list of all annotators that appear on each sample (i.e. same annotator occurs N times for N total annotations)
def apply_annotation_filter(dataset, min_annotations: int, cluster_other_annotators: bool = False, all_annotators: List[str] = None, update_consensus: bool = False):
    """
    Apply annotation filtering to a dataset.
    
    Args:
        dataset: Dataset to filter
        min_annotations: Minimum number of annotations required
        cluster_other_annotators: Whether to cluster filtered annotators into 'rest'
        
    Returns:
        Filtered dataset
    """
    # Store the original format to restore it after map operations
    original_format = dataset.format
    
    # Determine which annotators to keep vs group
    filter_settings = {
        'min': min_annotations,
        'rest_annotator': cluster_other_annotators
    }

    if all_annotators is None:
        all_annotators = [x for annotators in dataset['annotators'] for x in annotators]

    annotators_to_keep, annotators_to_group = get_filtered_annotators(filter_settings, all_annotators)
    
    # Filter the dataset using stable functions
    filter_fn = lambda x: replace_annotators(x, annotators_to_keep, annotators_to_group, replace_act_val=update_consensus)
    
    # Apply filtering to all samples
    filtered_dataset = dataset.map(filter_fn)
    
    # Remove samples that have no remaining annotators
    filtered_dataset = filtered_dataset.filter(has_annotators)

    # Restore the original format to ensure tensors remain tensors
    filtered_dataset.set_format(**original_format)

    return filtered_dataset

def apply_self_report_filter(dataset, self_report_strategy: str):
    """
    Apply MuSE self-report filtering to a dataset.
    
    Args:
        dataset: Dataset to filter
        self_report_strategy: Strategy for self-report filtering ('naive' or 'closest_label')
        
    Returns:
        Filtered dataset
    """
    # # Remove all annotators that are not self-report annotators
    # annotators_to_keep = set([ann for ann in dataset['annotators'] if 'self_report' in ann])
    
    # # Filter the dataset using stable functions
    # filter_fn = lambda x: replace_annotators(x, annotators_to_keep, set())
    
    # # Apply filtering to all samples
    # filtered_dataset = dataset.map(filter_fn)
    
    # # Remove samples that have no remaining annotators
    # filtered_dataset = filtered_dataset.filter(has_annotators)

    if self_report_strategy == 'naive':
        return dataset # No filtering applied, just naively add self-report values
    elif self_report_strategy == 'naive_only_self_report':
        # Remove all annotators that are not self-report annotators
        annotators_to_keep = list(set([ann for anns in dataset['annotators'] for ann in anns if 'self_report' in ann]))
        print(f'Keeping {len(annotators_to_keep)} self-report annotators')
        print(f'Annotators to keep: {annotators_to_keep}')
        filter_fn = lambda x: replace_annotators(x, annotators_to_keep, [], replace_act_val=True)
        filtered_dataset = dataset.map(filter_fn)
        filtered_dataset = filtered_dataset.filter(has_annotators)
        print('Dataset size after self-report filtering:', len(filtered_dataset))
        return filtered_dataset
    elif self_report_strategy == 'closest_label':
        # All filenames have the format of [speakerID]_[session/monologue ID]_[session/monologue ID]_[session/monologue ID]_sentence_[sentence number]
        # Only want to keep the last sentence in each monologue
        filenames = dataset['FileName']
        prefix_to_sentences = {}
        for filename in filenames:
            prefix = filename.split('_')
            sentence_num = int(prefix[-1].replace('.wav', ''))
            prefix = '_'.join(prefix[:-1])
            if prefix not in prefix_to_sentences:
                prefix_to_sentences[prefix] = []
            prefix_to_sentences[prefix].append(sentence_num)
        
        # Only keep the last sentence in each monologue
        keep_filenames = set()
        for prefix in prefix_to_sentences:
            max_sentence_num = max([int(x) for x in prefix_to_sentences[prefix]])
            keep_filenames.add(f'{prefix}_{max_sentence_num}.wav')

        # Filter the dataset to only include the last sentence in each monologue
        print(dataset['FileName'], 'DEBUG FILE NAMES', keep_filenames)
        dataset = dataset.filter(lambda x: x['FileName'] in keep_filenames)

        return dataset
    else:
        raise ValueError(f"Invalid self-report strategy: {self_report_strategy}")

def downsample_dataset(dataset, fraction: float = 0.05):
    """Downsample a dataset to a given fraction."""
    indices = list(range(dataset.num_rows))
    np.random.shuffle(indices)
    return dataset.select(indices[: int(len(indices) * fraction)])

def process_single_dataset_kde(dataset, original_data_format, seed: int = 0):
    """Process KDE columns for a single dataset."""
    format_columns = original_data_format['columns'].copy()
    if 'kde_2d_probability' not in format_columns:
        format_columns.append('kde_2d_probability')

    if 'kde_2d_probability' in dataset.features:
        dataset = dataset.remove_columns('kde_2d_probability')

    generation_columns = [
        col for col in dataset.column_names if col.startswith('kde_2d_probability_generation_')
    ]

    if generation_columns:
        target_generation_col = f'kde_2d_probability_generation_{seed}'
        if target_generation_col in dataset.column_names:
            dataset = dataset.rename_column(target_generation_col, 'kde_2d_probability')
        else:
            dataset = dataset.rename_column(generation_columns[0], 'kde_2d_probability')
    else:
        print('Warning: No kde_2d_probability generation columns found in dataset')

    updated_format = original_data_format.copy()
    updated_format['columns'] = format_columns
    dataset.set_format(**updated_format)

    return dataset

def process_kde_columns(dataset, original_data_format, seed: int = 0):
    """Process KDE columns for a dataset or list of datasets."""
    if isinstance(dataset, list):
        return [process_single_dataset_kde(d, original_data_format, seed) for d in dataset]
    return process_single_dataset_kde(dataset, original_data_format, seed)

def add_aggregate_annotator(sample: Dict) -> Dict:
    """
    Add aggregate-annotator to a sample for INDIVIDUAL_ANNOTATOR_PLUS models.
    
    This artificially adds the aggregate-annotator by:
    1. Appending 'aggregate-annotator' to the annotators list
    2. Appending the aggregate act/val labels to the soft label arrays
    3. Updating the variance fields to reflect the new labels
    
    Args:
        sample: Dataset sample
        
    Returns:
        Modified sample with aggregate-annotator added
    """
    # Create new soft label arrays with aggregate annotator added
    new_soft_act_labels = torch.cat((sample['soft_act_labels'], sample['act'].view(-1)), dim=0)
    new_soft_val_labels = torch.cat((sample['soft_val_labels'], sample['val'].view(-1)), dim=0)
    
    return {
        **sample, 
        'soft_act_labels': new_soft_act_labels,
        'soft_val_labels': new_soft_val_labels,
        'annotators': sample['annotators'] + ['aggregate-annotator'],
        # 'act_variance': new_soft_act_labels.var(), # Don't want to change the variance as the included aggregate-annotator should not change the variance/consensus label on the sample
        # 'val_variance': new_soft_val_labels.var()
    }

def add_aggregate_annotator_dataset(dataset):
    """Add aggregate annotator to an entire dataset or list of datasets."""
    if not isinstance(dataset, list):
        return add_aggregate_annotator_dataset([dataset])[0]
    result = []
    for d in dataset:
        original_format = d.format
        mapped = d.map(add_aggregate_annotator)
        mapped.set_format(**original_format)
        result.append(mapped)
    return result

def create_toy_datasets(train_dataset, val_dataset, test_dataset):
    """Create smaller datasets for testing purposes"""

    # Handle both single datasets and lists of datasets
    if isinstance(train_dataset, list):
        train_dataset = [downsample_dataset(x) for x in train_dataset]
        val_dataset = [downsample_dataset(x) for x in val_dataset]
        test_dataset = [downsample_dataset(x) for x in test_dataset]
    else:
        train_dataset = downsample_dataset(train_dataset)
        val_dataset = downsample_dataset(val_dataset)
        test_dataset = downsample_dataset(test_dataset)
        
    return train_dataset, val_dataset, test_dataset


def remove_annotators_from_dataset(dataset, annotators_to_remove):
    """Remove annotators from samples without discarding the entire row."""
    if not isinstance(dataset, list):
        return remove_annotators_from_dataset([dataset], annotators_to_remove)[0]

    result = []
    for d in dataset:
        original_format = d.format

        def map_fn(row):
            keep_mask = [ann not in annotators_to_remove for ann in row["annotators"]]
            row = replace_annotators(row, np.array(row["annotators"])[keep_mask].tolist(), [])
            return row

        mapped = d.map(map_fn)
        mapped = mapped.filter(has_annotators)
        mapped.set_format(**original_format)
        result.append(mapped)

    return result
