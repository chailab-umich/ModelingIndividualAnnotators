from typing import Optional, Union, List
import numpy as np
from torch.utils.data import DataLoader, Dataset

from ..config.model_config import ModelType
from ..training import BatchCollator
from ..data.utils import map_annotators_to_sample_idxs
from .data_sampler import BatchSamplerByAnnotator
from .dataset_wrapper import DatasetWrapper

def create_dataloader(
        dataset: Dataset,
        batch_size: int,
        model_type: ModelType,
        annotator_mapper: Optional[object] = None,
        is_training: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        sample_batches_by_annotator: bool = False
    ) -> DataLoader:
    """Create a data loader with appropriate configuration"""
    collate_fn = create_collate_fn(model_type, annotator_mapper)
    sampler = None
    if sample_batches_by_annotator:
        # Need to get mapping of annotators to samples they appear on 
        annotator_to_samples = map_annotators_to_sample_idxs(dataset) 
        # These map to lists, but to enable faster selection we want to conver to numpy arrays, this way we can generate 32 random indices and select immediately
        for ann in annotator_to_samples:
            annotator_to_samples[ann] = np.array(annotator_to_samples[ann])
        sampler = BatchSamplerByAnnotator(annotator_to_samples, batch_size=batch_size, shuffle=is_training)

    if sampler is None:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_training,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    else:
        return DataLoader(
            DatasetWrapper(dataset),
            batch_sampler=sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

def create_collate_fn(model_type: ModelType, annotator_mapper: Optional[object] = None) -> BatchCollator:
    """Get appropriate collate function based on model type"""
    if model_type in [ModelType.INDIVIDUAL_ANNOTATOR, ModelType.INDIVIDUAL_ANNOTATOR_BETA, ModelType.INDIVIDUAL_ANNOTATOR_LOOP, 
                        ModelType.CLUSTERED, ModelType.INDIVIDUAL_ANNOTATOR_PLUS, ModelType.INDIVIDUAL_ANNOTATOR_CONDOR, ModelType.INDIVIDUAL_ANNOTATOR_ORTHOGONAL]:
        if annotator_mapper is None:
            raise ValueError(f"annotator_mapper cannot be None for model type {model_type}")
        return BatchCollator(annotator_mapper=annotator_mapper, include_aggregate=False) # Should always be false even with individual annotator plus due to legacy reasons
    elif model_type == ModelType.DEER:
        return BatchCollator(pad_soft_labels=True)
    else:
        return BatchCollator()