from MIA.config.experiment_config import ExperimentConfig, ExperimentType
from MIA.data import make_audio_datasets

from .dataset_utils import (
    apply_annotation_filter,
    apply_self_report_filter,
    downsample_dataset,
    process_kde_columns,
    add_aggregate_annotator_dataset,
    create_toy_datasets,
)

class DatasetManager:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
    def prepare_datasets(self, seed: int = 0):
        """Prepare datasets based on configuration"""
        if self.config.training_config.finetune_on is None:
            # Train initial model on msp-podcast
            train_dataset_str = 'podcast'
            dataset_splits = make_audio_datasets(
                datasets_to_load=[train_dataset_str], 
                kde_size=4, 
                podcast_version=self.config.training_config.podcast_version, 
                prune_annotators=True,
                legacy_asru_splits=self.config.experiment_type == ExperimentType.ASRU, # Only *need* argument when cross-validation true, but it should do nothing when false
            )
        else:
            # Finetune trained model on new dataset
            train_dataset_str = self.config.training_config.finetune_on
            dataset_splits = make_audio_datasets(
                datasets_to_load=[train_dataset_str], 
                kde_size=4, 
                podcast_version=self.config.training_config.podcast_version, 
                cross_validation_folds=5, 
                prune_annotators=True,
                legacy_asru_splits=self.config.experiment_type == ExperimentType.ASRU,
                legacy_journal_splits=self.config.experiment_type == ExperimentType.JOURNAL,
            )
            
        train_dataset, val_dataset, test_dataset = dataset_splits[0][train_dataset_str], dataset_splits[1][train_dataset_str], dataset_splits[2][train_dataset_str]
        
        # Process kde_2d_probability columns
        train_dataset, val_dataset, test_dataset = self._process_kde_columns(
            train_dataset, val_dataset, test_dataset, seed=seed
        )
        
        # If we have cross-validation folds, select the appropriate fold based on seed
        if isinstance(train_dataset, list):
            fold_index = seed % len(train_dataset)  # Use seed to select fold
            train_dataset = train_dataset[fold_index]
            val_dataset = val_dataset[fold_index]
            test_dataset = test_dataset[fold_index]
        
        if self.config.training_config.toy_dataset:
            train_dataset, val_dataset, test_dataset = create_toy_datasets(train_dataset, val_dataset, test_dataset)
            
        return train_dataset, val_dataset, test_dataset
    
    def _process_kde_columns(self, train_dataset, val_dataset, test_dataset, seed: int = 0, for_training: bool = True):
        """Process kde_2d_probability generation columns"""
        # Store original format
        original_data_format = train_dataset[0].format if isinstance(train_dataset, list) else train_dataset.format

        processed_train = process_kde_columns(train_dataset, original_data_format, seed) # Since we pre-generated the kde column for up to five seeds, we use this to select current seed's KDE columns
        processed_val = process_kde_columns(val_dataset, original_data_format, seed)
        processed_test = process_kde_columns(test_dataset, original_data_format, seed)
        
        # Only add artificial annotators if this is for training (not for base testing datasets)
        if for_training:
            # Add aggregate-annotator for INDIVIDUAL_ANNOTATOR_PLUS models
            from ..config.model_config import ModelType
            if hasattr(self.config.model_config, 'model_type') and self.config.model_config.model_type == ModelType.INDIVIDUAL_ANNOTATOR_PLUS:
                print("Adding aggregate-annotator to datasets for INDIVIDUAL_ANNOTATOR_PLUS model")
                
                processed_train = add_aggregate_annotator_dataset(processed_train)
                processed_val = add_aggregate_annotator_dataset(processed_val)
                processed_test = add_aggregate_annotator_dataset(processed_test)
        else:
            print("Skipping artificial annotator addition - this is a base dataset for testing only")
        
        return processed_train, processed_val, processed_test
    
    def apply_filters(self, train_dataset, val_dataset, test_dataset):
        """Apply dataset filters based on configuration"""
        if not self.config.data_config.min_annotations:
            return train_dataset, val_dataset, test_dataset
        
        print(f"Applying filters: min_annotations={self.config.data_config.min_annotations}, cluster_other_annotators={getattr(self.config.data_config, 'cluster_other_annotators', False)}")
        
        def apply_filter_to_dataset(dataset, all_annotators):
            return apply_annotation_filter(
                dataset,
                min_annotations=self.config.data_config.min_annotations,
                cluster_other_annotators=getattr(self.config.data_config, 'cluster_other_annotators', False),
                all_annotators=all_annotators,
                update_consensus=self.config.experiment_type == ExperimentType.ASRU
            )
    
        def get_all_annotators(train_dataset, val_dataset, test_dataset):
            # Filter for min_annotations on the training dataset
            return [x for annotators in train_dataset['annotators'] for x in annotators]
        
        # Handle both single datasets and lists of datasets
        if isinstance(train_dataset, list):
            # Multiple folds case
            for i in range(len(train_dataset)):
                print(f"Filtering fold {i}")
                all_annotators = get_all_annotators(train_dataset[i], val_dataset[i], test_dataset[i])
                train_dataset[i] = apply_filter_to_dataset(train_dataset[i], all_annotators)
                val_dataset[i] = apply_filter_to_dataset(val_dataset[i], all_annotators)
                test_dataset[i] = apply_filter_to_dataset(test_dataset[i], all_annotators)
        else:
            # Single dataset case
            all_annotators = get_all_annotators(train_dataset, val_dataset, test_dataset)
            train_dataset = apply_filter_to_dataset(train_dataset, all_annotators)
            val_dataset = apply_filter_to_dataset(val_dataset, all_annotators)
            test_dataset = apply_filter_to_dataset(test_dataset, all_annotators)
            
        return train_dataset, val_dataset, test_dataset
    
    def apply_self_report_filters(self, train_dataset, val_dataset, test_dataset):
        """Apply self-report filters based on configuration"""
        if not self.config.data_config.self_report_type:
            return train_dataset, val_dataset, test_dataset
        
        print(f"Applying self-report filters: self_report_type={self.config.data_config.self_report_type}")
        
        def apply_self_report_filter_to_dataset(dataset):
            return apply_self_report_filter(
                dataset,
                self_report_strategy=self.config.data_config.self_report_type
            )
        
        # Handle both single datasets and lists of datasets
        if isinstance(train_dataset, list):
            # Multiple folds case
            for i in range(len(train_dataset)):
                print(f"Applying self-report filtering to fold {i}")
                train_dataset[i] = apply_self_report_filter_to_dataset(train_dataset[i])
                val_dataset[i] = apply_self_report_filter_to_dataset(val_dataset[i])
                test_dataset[i] = apply_self_report_filter_to_dataset(test_dataset[i])
        else:
            # Single dataset case
            train_dataset = apply_self_report_filter_to_dataset(train_dataset)
            val_dataset = apply_self_report_filter_to_dataset(val_dataset)
            test_dataset = apply_self_report_filter_to_dataset(test_dataset)
            
        return train_dataset, val_dataset, test_dataset
    
    def create_test_dataset_with_version(self, test_dataset, self_report_type):
        """
        Create a test dataset with a different self_report_type type.
        
        Args:
            test_dataset: The original test dataset
            self_report_type: The self-report type to use ('naive' or 'closest_label')
            
        Returns:
            A new test dataset with the specified self_report_type type
        """
        print(f"Creating test dataset with {self_report_type} self_report_type type...")
        
        def apply_self_report_filter_to_dataset(dataset):
            return apply_self_report_filter(
                dataset,
                self_report_strategy=self_report_type
            )
        
        # Handle both single datasets and lists of datasets
        if isinstance(test_dataset, list):
            # Multiple folds case
            filtered_test_dataset = []
            for i in range(len(test_dataset)):
                print(f"Applying {self_report_type} filtering to fold {i}")
                filtered_test_dataset.append(apply_self_report_filter_to_dataset(test_dataset[i]))
        else:
            # Single dataset case
            filtered_test_dataset = apply_self_report_filter_to_dataset(test_dataset)
            
        return filtered_test_dataset
    
    def prepare_all_datasets(self, seed: int = 0, for_training: bool = True):
        """Prepare datasets without fold selection - returns all folds if cross-validation"""
        cross_validation_folds = 5 if self.config.training_config.finetune_on is not None else False
        train_dataset_str = 'podcast' if self.config.training_config.finetune_on is None else self.config.training_config.finetune_on
        dataset_splits = make_audio_datasets(
            datasets_to_load=[train_dataset_str],
            kde_size=4,
            podcast_version=self.config.training_config.podcast_version,
            cross_validation_folds=cross_validation_folds,
            prune_annotators=True,
            legacy_asru_splits=self.config.experiment_type == ExperimentType.ASRU,
            legacy_journal_splits=self.config.experiment_type == ExperimentType.JOURNAL,
        )

        train_dataset, val_dataset, test_dataset = dataset_splits[0][train_dataset_str], dataset_splits[1][train_dataset_str], dataset_splits[2][train_dataset_str]
        
        # Process kde_2d_probability columns for all datasets/folds
        # This also handles aggregate-annotator addition for INDIVIDUAL_ANNOTATOR_PLUS models
        train_dataset, val_dataset, test_dataset = self._process_kde_columns(
            train_dataset, val_dataset, test_dataset, seed=seed, for_training=for_training
        )
        
        if self.config.training_config.toy_dataset:
            if isinstance(train_dataset, list):
                # Process each fold
                toy_train = []
                toy_val = []
                toy_test = []
                for fold in range(len(train_dataset)):
                    tr, va, te = create_toy_datasets(train_dataset[fold], val_dataset[fold], test_dataset[fold])
                    toy_train.append(tr)
                    toy_val.append(va)
                    toy_test.append(te)
                train_dataset, val_dataset, test_dataset = toy_train, toy_val, toy_test
            else:
                train_dataset, val_dataset, test_dataset = create_toy_datasets(train_dataset, val_dataset, test_dataset)
            
        return train_dataset, val_dataset, test_dataset
