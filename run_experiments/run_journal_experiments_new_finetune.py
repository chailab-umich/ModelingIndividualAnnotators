import os
import yaml
import torch
import argparse
from pathlib import Path
from itertools import product
from typing import List, Dict

from MIA.config import ExperimentType
from experiment_runner import run_experiments_from_config

def generate_finetune_experiments(config: Dict) -> List[str]:
    """Generate experiment names for all finetune combinations"""
    base_experiments = []
    for model_name in config['experiments']:
        if config['experiments'][model_name]['model_type'] in ['individual_annotator', 'individual_annotator_plus']:
            # Only include the base models (not the _var variants) for finetuning
            if not any(suffix in model_name for suffix in ['_100plus', '_1000plus']):
                base_experiments.append(model_name)
            
    finetune_experiments = []
    for model_name in base_experiments:
        for random_map, num_anns, map_separately in product(
            config['finetune_settings']['random_map_settings'],
            config['finetune_settings']['num_annotators'],
            [True, False]  # map_act_val_separately options
        ):
            exp_name = f"{model_name}_with_{num_anns}_per_new_annotator"
            if map_separately:
                exp_name += "_separate_act_val_mapping"
            if random_map:
                exp_name += "_random_map"
                
            finetune_experiments.append(exp_name)
            
    return finetune_experiments

if __name__ == '__main__':
    # Set multiprocessing strategy
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    # Load config
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / 'run_experiments' / 'configs' / 'journal_experiments.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if not Path(config['output_path']).is_absolute():
        config['output_path'] = str(project_root / config['output_path'])
    
    print("Journal Paper Pre-training Experiments")
    print("=" * 50)
    print("Running pre-training experiments for different annotator filtering criteria:")
    print("- All annotators (min_annotations: 1)")
    print("- Annotators with 100+ annotations")  
    print("- Annotators with 1000+ annotations")
    print()
    config['training']['test_only'] = True
    
    # Run ONLY base experiments (pre-training) - no finetuning
    run_experiments_from_config(
        config_path,
        ExperimentType.JOURNAL,
        # experiments_to_run=list(config['experiments'].keys()),
        config=config,
        experiments_to_run=[
            'individual_annotator',
            'individual_annotator_100plus',
            'individual_annotator_1000plus',
            'baseline_aggregate',
            'baseline_aggregate_100plus',
            'baseline_aggregate_1000plus',
            'aggregate_ground_truth',
            # 'baseline',
        ],
        finetune_on=['muse', 'iemocap', 'improv'],
        allow_crash=True,
        # use_ia_early_stopping=True, # Test individual annotator early stopping
        use_ia_early_stopping=False
    )
    
    print("Pre-training experiments completed!")
    print()
    print("If you want to run finetuning experiments later, uncomment the following section:")
    print("# Generate finetune experiment names")
    print("# finetune_experiments = generate_finetune_experiments(config)")
    print("# config['training']['finetune_on'] = ['muse', 'iemocap', 'improv']")
    print("# run_experiments_from_config(")
    print("#     config_path,")
    print("#     ExperimentType.JOURNAL,")
    print("#     experiments_to_run=finetune_experiments,")
    print("#     allow_crash=True")
    print("# )")
