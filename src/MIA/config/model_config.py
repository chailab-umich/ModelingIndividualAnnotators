from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Union, List

class ModelType(Enum):
    # Baselines
    KDE_2D = 1
    DEER = 2
    CLUSTERED = 3
    # Non-baselines
    HUBI_MEDIUM = 4  # kinda baseline but moreso adaptation
    INDIVIDUAL_ANNOTATOR = 5
    INDIVIDUAL_ANNOTATOR_PLUS = 6
    INDIVIDUAL_ANNOTATOR_BETA = 7
    ONE_HOT_ANNOTATORS = 8
    BASELINE_AGGREGATE = 9
    AGGREGATE_GROUND_TRUTH = 10  # This is the upper bound; model predicts perfect average ground truth
    INDIVIDUAL_ANNOTATOR_CONDOR = 11
    INDIVIDUAL_ANNOTATOR_ORTHOGONAL = 12
    INDIVIDUAL_ANNOTATOR_LOOP = 13  # Loop-based implementation for benchmarking
@dataclass
class ModelArchitectureConfig:
    """Configuration for model architecture"""
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True

@dataclass
class AnnotatorConfig:
    """Configuration for annotator handling"""
    min_annotations: Optional[int] = None
    cluster_other_annotators: bool = False
    use_aggregate: bool = False

@dataclass
class ModelConfig:
    """Complete model configuration"""
    model_type: ModelType
    prob_grid_size: int
    determinism: bool = True
    determinism_type: str = "default"
    disable_kde: bool = False
    dense_layers: int = 2
    layer_size: int = 256
    kde_training_temperature: int = 8
    kde_training_grid_size: int = 4
    training_precision: type = None  # Will be set to torch.float32 by default
    condor: str = None
    basis_annotator_rank: int = None
    orthogonal_model_features: str = None
    flip_negative_values_in_validation: bool = False

    @classmethod
    def create_default(cls, model_type: ModelType, prob_grid_size: int) -> 'ModelConfig':
        """Create a default configuration for a given model type"""
        return cls(
            model_type=model_type,
            prob_grid_size=prob_grid_size,
        ) 