from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, List, Union
from .model_config import ModelConfig

class ExperimentType(Enum):
    ASRU = "asru"
    JOURNAL = "journal"
    ICASSP = "icassp"
    INTERSPEECH24 = "interspeech24"
    CONDOR = "condor"
    ORTHOGONAL = "orthogonal"

@dataclass
class FineTuningConfig:
    """Configuration for finetuning parameters"""
    num_annotators: int
    map_act_val_separately: bool
    random_map: bool
    num_samples_to_use: Union[str, int] = 'all'  # This can be 'all' or a number
    max_epochs: Optional[int] = None
    use_basis_coefficients: bool = False  # For orthogonal models: use optimal basis coefficients instead of most similar annotator

    @classmethod
    def create(cls, num_anns: int, map_separately: bool, random: bool, samples: Union[str, int] = 'all', epochs: Optional[int] = None, use_basis: bool = False):
        return cls(num_anns, map_separately, random, samples, epochs, use_basis)

@dataclass
class TrainingConfig:
    batch_size: int = 32
    max_epochs: int = 500
    toy_dataset: bool = False
    test_only: bool = False
    podcast_version: str = '2.0'
    finetune_on: Optional[str] = None
    number_of_seeds: int = 6
    finetune_config: Optional[FineTuningConfig] = None
    all_annotators_zeroshot: bool = False,
    benchmarking: Optional[bool] = False

@dataclass
class DataConfig:
    regenerate_kde_labels: bool = False
    min_annotations: Optional[int] = None
    cluster_other_annotators: bool = False
    unseen_annotator_strategy: Optional[str] = None  # 'rest', 'aggregate', 'first', or specific annotator name
    self_report_type: Optional[str] = None # 'naive' or 'closest_label'
    sample_batches_by_annotator: bool = False
    pretrain_with_normal_sampling: bool = False
    training_paradigm_3: bool = False

class ExperimentalSettings:
    """Provides default experimental settings for different experiment types"""
    
    @staticmethod
    def get_finetune_configs(experiment_type: ExperimentType) -> List[FineTuningConfig]:
        """Get default finetuning configurations for a given experiment type"""
        if experiment_type == ExperimentType.ASRU:
            # ASRU Experiment settings
            configs = [
                FineTuningConfig.create(1, False, True, 'all', None),
                FineTuningConfig.create(1, False, False, 'all', None)
            ]
            # Add configurations for different sample sizes
            configs.extend([
                FineTuningConfig.create(1, False, False, num_samples_to_use_for_map*5, None)
                for num_samples_to_use_for_map in range(1, 7)
            ])
            return configs
        elif experiment_type == ExperimentType.JOURNAL:
            # Journal experiment settings
            configs = []
            for num_anns in [1]:
                for random_map in [True, False]:
                    for map_separately in [True, False]:
                        configs.append(FineTuningConfig.create(
                            num_anns,
                            map_separately=map_separately,
                            random=random_map,
                            samples='all',
                            epochs=None
                        ))
            return configs
        elif experiment_type == ExperimentType.ICASSP or experiment_type == ExperimentType.CONDOR:
            configs = [
                FineTuningConfig.create(1, True, False, 'all', None),
            ]
            return configs
        elif experiment_type == ExperimentType.INTERSPEECH24:
            configs = [
                # 1 annotator mapped to 1 annotator, False (no random mapping), False (no separate act/val mapping), all samples used, 0 max epochs (zero-shot only)
                FineTuningConfig.create(1, False, False, 'all', 0),
            ]
            return configs
        elif experiment_type == ExperimentType.ORTHOGONAL:
            # Orthogonal experiment settings
            # Compare two approaches:
            # 1. Most similar annotator mapping (existing, use_basis=False)
            # 2. Optimal basis coefficients (new, use_basis=True)
            configs = []
            for map_separately in [False, True]:
                # Most similar annotator approach
                configs.append(FineTuningConfig.create(
                    num_anns=1,
                    map_separately=map_separately,
                    random=False,
                    samples='all',
                    epochs=None,
                    use_basis=False
                ))
                # Optimal basis coefficients approach
                configs.append(FineTuningConfig.create(
                    num_anns=None,  # Not used for basis coefficients
                    map_separately=map_separately,
                    random=False,
                    samples='all',
                    epochs=None,
                    use_basis=True
                ))
            return configs
        raise ValueError(f"Unknown experiment type: {experiment_type}")

@dataclass
class ExperimentConfig:
    experiment_type: ExperimentType
    model_config: ModelConfig
    training_config: TrainingConfig
    data_config: DataConfig
    log_path: str
    model_save_path: str
    csv_results_path: str
    task_str: str = 'task1,task2'
    experiment_name: Optional[str] = None 