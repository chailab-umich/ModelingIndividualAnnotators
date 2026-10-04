import os
import yaml
import torch
import multiprocessing as mp
from datetime import datetime
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from glob import glob
import shutil
import re

from MIA.config import (
    ExperimentConfig, 
    ModelConfig, 
    TrainingConfig, 
    DataConfig, 
    ExperimentType,
    ModelType,
    FineTuningConfig,
    ExperimentalSettings
)
from MIA.experiment import ExperimentRunner
from MIA.data import make_audio_datasets

# Global flag to track if multiprocessing has been initialized
_mp_initialized = False
# Global flag to track if datasets have been cached
_datasets_cached = False
# Global flag to track if trash directory has been cleared
_trash_cleared = False

def initialize_multiprocessing():
    """Initialize multiprocessing settings exactly once"""
    global _mp_initialized
    if not _mp_initialized:
        torch.multiprocessing.set_sharing_strategy('file_system')
        mp.set_start_method('spawn')
        _mp_initialized = True

def cache_datasets(config: dict, finetune_on: Optional[List[str]] = None):
    """Cache datasets exactly once"""
    global _datasets_cached
    if not _datasets_cached:
        print('Creating and caching all datasets before multiprocessing to prevent race conditions')
        make_audio_datasets(datasets_to_load=['podcast'], podcast_version=config['training']['podcast_version'])
        if finetune_on:
            for dataset in finetune_on:
                make_audio_datasets(datasets_to_load=[dataset])
        print('Dataset verification complete')
        _datasets_cached = True

def clear_trash_directory(output_dir: str):
    """Clear the trash directory from previous runs exactly once"""
    global _trash_cleared
    if not _trash_cleared:
        trash_dir = f'{output_dir}/trash'
        if os.path.exists(trash_dir):
            print(f'Clearing trash directory: {trash_dir}')
            shutil.rmtree(trash_dir)
            print('Trash directory cleared')
        else:
            print('No trash directory found to clear')
        _trash_cleared = True

class WorkerManager:
    """Manages multiple worker processes across available GPUs"""
    def __init__(self, gpus_to_use: Optional[List[int]] = None):
        self.num_scripts_per_gpu = 3 # How many jobs can run simultaneously per GPU
        self.num_gpus = len(gpus_to_use) if gpus_to_use is not None else torch.cuda.device_count()
        self.num_processes = self.num_gpus * self.num_scripts_per_gpu
        self.job_queue = mp.Queue()
        self.results_queue = mp.Queue()
        self.available_gpus = gpus_to_use if gpus_to_use is not None else [gpu for gpu in range(self.num_gpus) for _ in range(self.num_scripts_per_gpu)]
        # self.available_gpus = [0,2]
        self.processes = []

    def create_workers(self):
        """Create and start worker processes"""
        for gpu in self.available_gpus:
            process = Worker(self.job_queue, self.results_queue, gpu)
            self.processes.append(process)
            process.start()

    def graceful_shutdown(self):
        """Shutdown all workers gracefully"""
        # Clear queue to prevent unlaunched jobs
        while not self.job_queue.empty():
            self.job_queue.get()
        # Send shutdown signal to all workers
        for _ in range(self.num_processes):
            self.job_queue.put('shutdown worker')

    def launch_jobs(self, experiments: List[Tuple[str, ExperimentConfig]], allow_crash: bool = True):
        """Launch jobs to worker processes"""
        for exp_name, exp_config in experiments:
            self.job_queue.put((exp_name, allow_crash, exp_config))

    def wait_for_jobs(self, job_names: List[str]):
        """Wait for all jobs to complete"""
        finished_jobs = []
        failed_jobs = []
        print('Waiting for jobs to finish:', job_names)
        
        while sorted(finished_jobs) != sorted(job_names):
            finished_job = self.results_queue.get(block=True, timeout=None)
            
            if not isinstance(finished_job, str):
                print('Found a crashed job where allow_crash=False')
                script_name, exception_info = finished_job
                print(f'Crash information for {script_name}:')
                print(exception_info[0])
                print(exception_info[1])
                print('see script log for traceback information')
                print('Shutting down')
                self.graceful_shutdown()
                sys.exit(1)
            else:
                if finished_job.startswith('BAD_'):
                    finished_job = finished_job.replace('BAD_', '')
                    failed_jobs.append(finished_job)
                finished_jobs.append(finished_job)

            print('Finished job(s):', finished_job, '(', finished_jobs, ')')
            print('Remaining jobs:', set(job_names) - set(finished_jobs))
            print('Crashed jobs:', failed_jobs)

class Worker(mp.Process):
    """Worker process that runs experiments on a specific GPU"""
    def __init__(self, job_queue: mp.Queue, results_queue: mp.Queue, gpu: int):
        super(Worker, self).__init__()
        self.job_queue = job_queue
        self.results_queue = results_queue
        self.gpu = gpu
        
        # Set environment variables for GPU
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{self.gpu}'
        print(f'Worker initialised on GPU {self.gpu}')

    def run(self):
        """Main worker loop"""
        print(f'Worker ({os.getpid()}) ready for jobs on GPU {self.gpu}')
        
        for job in iter(self.job_queue.get, 'shutdown worker'):
            if job == 'shutdown worker':
                break
                
            exp_name, allow_crash, exp_config = job
            print(f'Worker ({os.getpid()}) running job {exp_name}')
            start_time = datetime.now()
            results_item = exp_name
            
            try:
                # Redirect output to log file using full experiment name
                log_file = Path(exp_config.log_path).joinpath(f'{exp_name}.log')
                log_file.parent.mkdir(parents=True, exist_ok=True)
                sys.stdout = sys.stderr = open(log_file, 'w', buffering=1)

                # Run experiment
                runner = ExperimentRunner(exp_config)
                runner.run()
                
            except Exception:
                the_type, the_value, the_traceback = sys.exc_info()
                print(f'ERROR -- {exp_name} crashed continuing with other processes...')
                print('crash information')
                print(the_type)
                print(the_value)
                traceback.print_tb(the_traceback, file=sys.stdout)
                results_item = f'BAD_{results_item}'
                
                if not allow_crash:
                    results_item = (results_item, (the_type, the_value))
                    
                temp = sys.stdout
                sys.stdout = sys.__stdout__
                print(f'ERROR -- {exp_name} crashed (see log for info) continuing with other processes...')
                sys.stdout = temp
                
            finally:
                print('Experiment start time', start_time.strftime('%D %H:%M:%S'))
                print('Experiment end time', datetime.now().strftime('%D %H:%M:%S'))
                sys.stdout = sys.__stdout__
                self.results_queue.put(results_item)

def load_config(config_path: str) -> dict:
    """Load experiment configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def check_existing_run(output_dir: str, full_experiment_name: str, config: dict) -> bool:
    """Check if experiment has already been run successfully"""
    # Use experiment_group from config as directory name
    exp_dir = config['experiment_group']
    
    # Add _ft_on_{dataset} suffix if this is a finetuning experiment
    if 'finetune_on' in config['training'] and config['training']['finetune_on']:
        dataset = config['training']['finetune_on']
        exp_dir = f"{exp_dir}_ft_on_{dataset}"
    
    # Check for exact match of full experiment name
    job_log = f'{output_dir}/logs/{exp_dir}/{full_experiment_name}.log'
    print(f"Looking for log file: {job_log}")
    
    if not os.path.exists(job_log):
        return False
        
    try:
        with open(job_log, 'r', encoding='utf-8') as f:
            log_content = f.read()
    except UnicodeDecodeError:
        print(f"Warning: Log file {job_log} has encoding issues, treating as failed run")
        return False
    except Exception as e:
        print(f"Warning: Could not read log file {job_log}: {e}, treating as failed run")
        return False
        
    if 'crash information' in log_content:
        return False
        
    # Check if results are still stored
    existing_tb_logs = glob(f'{output_dir}/tb_log_files/{exp_dir}/{full_experiment_name}_[0-9][0-9]_[0-9][0-9]_2024_[0-9][0-9]:[0-9][0-9]:[0-9][0-9]/model_prob4_seed_[0-9]/*.[0-9]*')
    tb_logs_good = len(existing_tb_logs) == 5
    
    models_exist_for_all_seeds = True
    for seed in range(5):
        existing_trained_models = glob(f'{output_dir}/TrainedModels/{exp_dir}/{full_experiment_name}/model_prob4_seed_{seed}/*.ckpt')
        models_exist_for_all_seeds = models_exist_for_all_seeds and len(existing_trained_models) > 0
        
    return models_exist_for_all_seeds and tb_logs_good

def cleanup_failed_run(output_dir: str, full_experiment_name: str, config: dict):
    """Clean up files from failed runs by moving to trash directory"""
    # Use experiment_group from config as directory name
    exp_dir = config['experiment_group']
    
    # Add _ft_on_{dataset} suffix if this is a finetuning experiment
    if 'finetune_on' in config['training'] and config['training']['finetune_on']:
        dataset = config['training']['finetune_on']
        exp_dir = f"{exp_dir}_ft_on_{dataset}"
    
    # Create trash directory if it doesn't exist
    trash_dir = f'{output_dir}/trash'
    Path(trash_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp with microseconds for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    # Clean up exact match of full experiment name
    job_log = f'{output_dir}/logs/{exp_dir}/{full_experiment_name}.log'
    print(f"Moving failed run files to trash for: {full_experiment_name}")
    
    if os.path.exists(job_log):
        # Move log file to trash with timestamp
        trash_log_dir = f'{trash_dir}/logs_{timestamp}'
        Path(trash_log_dir).mkdir(parents=True, exist_ok=True)
        shutil.move(job_log, f'{trash_log_dir}/{full_experiment_name}.log')
        
    tb_dir = glob(f'{output_dir}/tb_log_files/{exp_dir}/{full_experiment_name}_[0-9][0-9]_[0-9][0-9]_2024_[0-9][0-9]:[0-9][0-9]:[0-9][0-9]/')
    model_dir = f'{output_dir}/TrainedModels/{exp_dir}/{full_experiment_name}'
    
    if len(tb_dir) and os.path.exists(tb_dir[0]):
        # Move tensorboard logs to trash
        trash_tb_dir = f'{trash_dir}/tb_logs_{timestamp}'
        Path(trash_tb_dir).mkdir(parents=True, exist_ok=True)
        shutil.move(tb_dir[0], f'{trash_tb_dir}/{os.path.basename(tb_dir[0])}')
        
    if not config['training']['test_only'] and os.path.exists(model_dir):
        # Move model directory to trash with unique naming
        trash_model_dir = f'{trash_dir}/models_{timestamp}'
        Path(trash_model_dir).mkdir(parents=True, exist_ok=True)
        # Use a unique name for the moved directory to avoid collisions
        unique_model_name = f'{full_experiment_name}_{timestamp}'
        shutil.move(model_dir, f'{trash_model_dir}/{unique_model_name}')

def sort_experiments(experiments: List[str]) -> List[str]:
    """Sort experiments by number of tasks and alphabetically"""
    matcher = re.compile(r'baseline|_\d(?:_[a-zA-Z]+)*')
    jobs_by_num_tasks = {}
    
    for exp in experiments:
        num_tasks = len(matcher.findall(exp))
        if num_tasks not in jobs_by_num_tasks:
            jobs_by_num_tasks[num_tasks] = []
        jobs_by_num_tasks[num_tasks].append(exp)
        
    jobs_by_num_tasks = {k: sorted(jobs_by_num_tasks[k]) for k in jobs_by_num_tasks}
    
    job_order = []
    kde_tasks = []
    for num_tasks in jobs_by_num_tasks:
        for job in jobs_by_num_tasks[num_tasks]:
            if '_3' in job:
                kde_tasks.append(job)
            else:
                job_order.append(job)
                
    return job_order + kde_tasks

def create_experiment_config(
    full_experiment_name: str,
    experiment_config: dict,
    base_config: dict,
    experiment_type: ExperimentType,
    finetune_config: Optional[FineTuningConfig] = None,
    use_ia_early_stopping: bool = False
) -> ExperimentConfig:
    """Create experiment configuration from YAML config"""
    # Handle annotator filters - start with experiment-specific filters if present
    annotator_filters = experiment_config.get('annotator_filters', {}).copy()
    
    # If ASRU, then we only want to keep annotators that labeled at least 30 samples
    # Override with ASRU defaults if not already specified
    if experiment_type == ExperimentType.ASRU:
        if 'min_annotations' not in annotator_filters:
            annotator_filters['min_annotations'] = 30
        if 'cluster_other_annotators' not in annotator_filters:
            annotator_filters['cluster_other_annotators'] = False

    # Extract model configuration parameters from YAML, using defaults where needed
    model_type = ModelType[experiment_config['model_type'].upper()]
    
    print(f"Creating config for experiment '{full_experiment_name}' with model_type: {model_type}")
    if annotator_filters:
        print(f"  Annotator filters: {annotator_filters}")
    
    # Convert training_precision string to torch dtype
    training_precision_str = experiment_config.get('training_precision', base_config['model_defaults'].get('training_precision', 'float32'))
    if training_precision_str == 'float32':
        training_precision = torch.float32
    elif training_precision_str == 'float64':
        training_precision = torch.float64
    else:
        training_precision = torch.float32  # default fallback
    
    model_params = {
        'model_type': model_type,
        'prob_grid_size': base_config['model_defaults']['prob_grid_size'],
        'determinism': experiment_config.get('determinism', base_config['model_defaults'].get('determinism', False)),
        'determinism_type': experiment_config.get('determinism_type', base_config['model_defaults'].get('determinism_type', 'default')),
        'disable_kde': experiment_config.get('disable_kde', True),
        'dense_layers': experiment_config.get('dense_layers', base_config['model_defaults'].get('dense_layers', 2)),
        'layer_size': experiment_config.get('layer_size', base_config['model_defaults'].get('layer_size', 256)),
        'kde_training_temperature': experiment_config.get('kde_training_temperature', base_config['model_defaults'].get('kde_training_temperature', 8)),
        'kde_training_grid_size': experiment_config.get('kde_training_grid_size', base_config['model_defaults'].get('kde_training_grid_size', 4)),
        'training_precision': training_precision,
        'condor': experiment_config.get('condor', None),
        'basis_annotator_rank': experiment_config.get('basis_annotator_rank', None),
        'orthogonal_model_features': experiment_config.get('orthogonal_model_features', None),
        'flip_negative_values_in_validation': experiment_config.get('flip_negative_values_in_validation', False),
    }
    
    # Ensure baseline_aggregate models have correct settings
    if model_type == ModelType.BASELINE_AGGREGATE:
        print("Ensuring baseline_aggregate model has correct settings")
        model_params['determinism'] = False
        model_params['disable_kde'] = True  # Baseline aggregate doesn't need KDE

    model_config = ModelConfig(**model_params)
    
    # Create training config with finetuning configuration
    training_params = base_config['training'].copy()
    training_params['finetune_config'] = finetune_config
    training_config = TrainingConfig(**training_params)
    setattr(training_config, 'individual_annotator_early_stopping', use_ia_early_stopping)
    if 'podcast_version' in experiment_config:
        print(f"Setting podcast version to {experiment_config['podcast_version']}")
        setattr(training_config, 'podcast_version', experiment_config['podcast_version'])

    # Create data config and merge annotator filters
    data_params = base_config['data'].copy()
    if annotator_filters:
        print(f"  Applying annotator filters to data config: {annotator_filters}")
        data_params.update(annotator_filters)
    
    if 'self_report_type' in experiment_config:
        print(f"Setting self_report_type to {experiment_config['self_report_type']}")
        setattr(data_config, 'self_report_type', experiment_config['self_report_type'])
    
    sample_batches_by_annotator = experiment_config.get('sample_batches_by_annotator', False)
    pretrain_with_normal_sampling = experiment_config.get('pretrain_with_normal_sampling', False)
    if sample_batches_by_annotator:
        print(f"Setting sample_batches_by_annotator to {sample_batches_by_annotator}")
        print(f'Sample batches by annotator requires annotator filters to be set to minimum 32')
        # Set the actual DataConfig attributes - dataset_manager reads data_config.min_annotations
        # and data_config.cluster_other_annotators, NOT data_config.annotator_filters
        data_params['min_annotations'] = 32
        data_params['cluster_other_annotators'] = False
    data_params['sample_batches_by_annotator'] = sample_batches_by_annotator
    data_params['pretrain_with_normal_sampling'] = pretrain_with_normal_sampling
    data_params['training_paradigm_3'] = experiment_config.get('training_paradigm_3', False)
    # if data_config.pretrain_with_normal_sampling:
        # print(f"Setting pretrain_with_normal_sampling to {data_config.pretrain_with_normal_sampling}")
    data_config = DataConfig(**data_params)

    # Use experiment_group from config as directory name
    exp_dir = base_config['experiment_group']
    
    # If finetuning or zero-shot all-annotators, add _ft_on_{dataset} to exp_dir
    if (finetune_config is not None or base_config['training'].get('all_annotators_zeroshot', False)) and 'finetune_on' in base_config['training']:
        dataset = base_config['training']['finetune_on']
        # If dataset is a list, join with _
        if isinstance(dataset, list):
            raise ValueError(f"Should be a single dataset, not a list: {dataset}")
        exp_dir = f"{exp_dir}_ft_on_{dataset}"

    # Create log, model save, and CSV results directories using full experiment name
    log_path = f"{base_config['output_path']}/logs/{exp_dir}"
    model_save_path = f"{base_config['output_path']}/TrainedModels/{exp_dir}/{full_experiment_name}"
    csv_results_path = f"{base_config['output_path']}/csv_results/{exp_dir}/{full_experiment_name}"
    
    # Create directories
    Path(log_path).mkdir(parents=True, exist_ok=True)
    Path(model_save_path).mkdir(parents=True, exist_ok=True)
    Path(csv_results_path).mkdir(parents=True, exist_ok=True)

    task_str = experiment_config['tasks']
    if task_str == 'task1softmaxccc':
        if training_config.finetune_on == 'muse':
            task_str = 'task1softmaxccc9'
        else:
            task_str = 'task1softmaxccc5'
    if task_str == 'task1CORAL':
        if training_config.finetune_on == 'muse':
            task_str = 'task1CORAL9'
        else:
            task_str = 'task1CORAL5'
    
    return ExperimentConfig(
        experiment_type=experiment_type,
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        log_path=log_path,
        model_save_path=model_save_path,
        csv_results_path=csv_results_path,
        task_str=task_str,
        experiment_name=full_experiment_name
    )

def run_experiments_from_config(
    config_path: str,
    experiment_type: ExperimentType,
    experiments_to_run: Optional[List[str]] = None,
    allow_crash: bool = False,
    finetune_on: Optional[List[str]] = None,
    config: Optional[dict] = None,
    use_ia_early_stopping: bool = False,
    benchmarking: bool = False,
    gpus_to_use: Optional[List[int]] = None
):
    """Run experiments from configuration file using multiprocessing"""
    # Initialize multiprocessing if not already done
    initialize_multiprocessing()
    
    # Load config if not provided
    if config is None:
        config = load_config(config_path)
    
    # Clear trash directory from previous runs
    clear_trash_directory(config['output_path'])
    
    # Cache datasets if not already done
    cache_datasets(config, finetune_on)
    
    # Get experiments to run
    if experiments_to_run is None:
        experiments_to_run = list(config['experiments'].keys())
        
    # Sort experiments
    experiments_to_run = sort_experiments(experiments_to_run)
    
    # Create experiment configs for each experiment-dataset combination
    experiment_configs = []
    job_names = []
    
    if finetune_on:
        # Get finetuning configurations for the experiment type
        finetune_configs = ExperimentalSettings.get_finetune_configs(experiment_type)
        
        print(f"Creating finetuning jobs for {len(experiments_to_run)} experiments x {len(finetune_on)} datasets x {len(finetune_configs)} finetune configs")
        
        for dataset in finetune_on:
            self_report_types = ['naive', 'naive_only_self_report'] if experiment_type == ExperimentType.ICASSP else ['']
            for self_report_type in self_report_types:
                for base_exp_name in experiments_to_run:
                    # Check if this experiment should support finetuning
                    base_model_type = ModelType[config['experiments'][base_exp_name]['model_type'].upper()]
                    
                    # Get appropriate finetuning configs for this model type
                    if base_model_type in [ModelType.INDIVIDUAL_ANNOTATOR, ModelType.INDIVIDUAL_ANNOTATOR_PLUS, ModelType.CLUSTERED, ModelType.INDIVIDUAL_ANNOTATOR_ORTHOGONAL, ModelType.INDIVIDUAL_ANNOTATOR_LOOP]:
                        model_finetune_configs = finetune_configs
                        # --- Add zero-shot all-annotators config ---
                        if self_report_type == '':
                            zero_shot_exp_name = f"{base_exp_name}_ft_on_{dataset}_zero_shot_all_annotators"
                        else:
                            zero_shot_exp_name = f"{base_exp_name}_{self_report_type}_ft_on_{dataset}_zero_shot_all_annotators"
                        
                        # Apply IA early stopping suffix if needed
                        if use_ia_early_stopping and not zero_shot_exp_name.endswith('_IAES'):
                            zero_shot_exp_name += '_IAES'
                        
                        if benchmarking and not zero_shot_exp_name.startswith('benchmarking_'):
                            zero_shot_exp_name = 'benchmarking_' + zero_shot_exp_name
                        
                        # Create dataset config first
                        dataset_config = config.copy()
                        dataset_config['training'] = config['training'].copy()
                        dataset_config['training']['finetune_on'] = dataset
                        dataset_config['training']['all_annotators_zeroshot'] = True

                        # Check if already completed
                        if not check_existing_run(config['output_path'], zero_shot_exp_name, dataset_config):
                            print(f'Cleaning up failed run for {zero_shot_exp_name}')
                            cleanup_failed_run(config['output_path'], zero_shot_exp_name, dataset_config)
                            exp_config = create_experiment_config(
                                zero_shot_exp_name,
                                {**config['experiments'][base_exp_name], 'self_report_type': None if self_report_type == '' else self_report_type},
                                dataset_config,
                                experiment_type,
                                None,  # No finetune config for zero-shot
                                use_ia_early_stopping=use_ia_early_stopping
                            )
                            experiment_configs.append((zero_shot_exp_name, exp_config))
                            job_names.append(zero_shot_exp_name)
                    elif base_model_type == ModelType.BASELINE_AGGREGATE:
                        # Special case for baseline aggregate - simpler configs
                        if experiment_type == ExperimentType.ASRU:
                            model_finetune_configs = [
                                FineTuningConfig.create(None, False, False, 'all', None)
                            ]
                            model_finetune_configs.extend([
                                FineTuningConfig.create(None, False, False, num_samples_to_use_for_map*5, None)
                                for num_samples_to_use_for_map in range(1, 7)
                            ])
                        else:  # Journal settings
                            model_finetune_configs = [FineTuningConfig.create(None, False, False, 'all', None)]
                    elif base_model_type in [ModelType.AGGREGATE_GROUND_TRUTH]:
                        # These models support fine-tuning but only with basic/default config
                        model_finetune_configs = [FineTuningConfig.create(None, False, False, 'all', None)]
                    else:
                        raise ValueError(f'Model type {base_model_type} does not support finetuning')
                    
                    for finetune_config in model_finetune_configs:
                        # Generate experiment name with finetuning parameters
                        full_exp_name = _generate_finetune_experiment_name(base_exp_name, dataset, finetune_config, self_report_type)
                        finetune_config.map_settings = config['experiments'][base_exp_name]['map_settings'] if 'map_settings' in config['experiments'][base_exp_name] else None
                        print(f'Setting map_settings to {finetune_config.map_settings}')
                        
                        # Apply IA early stopping suffix if needed
                        if use_ia_early_stopping and not full_exp_name.endswith('_IAES'):
                            full_exp_name += '_IAES'
                        
                        if benchmarking and not full_exp_name.startswith('benchmarking_'):
                            full_exp_name = 'benchmarking_' + full_exp_name

                        # Update config for this specific dataset and finetuning config first
                        dataset_config = config.copy()
                        dataset_config['training'] = config['training'].copy()
                        dataset_config['training']['finetune_on'] = dataset
                        
                        # Check if already completed
                        if check_existing_run(config['output_path'], full_exp_name, dataset_config):
                            print(f'Skipping {full_exp_name} as already successfully trained')
                            continue

                        # Clean up any failed runs
                        print(f'Cleaning up failed run for {full_exp_name}')
                        cleanup_failed_run(config['output_path'], full_exp_name, dataset_config)
                        
                        # Create experiment config
                        exp_config = create_experiment_config(
                            full_exp_name,
                            {**config['experiments'][base_exp_name], 'self_report_type': self_report_type},
                            dataset_config,
                            experiment_type,
                            finetune_config,
                            use_ia_early_stopping=use_ia_early_stopping
                        )
                        experiment_configs.append((full_exp_name, exp_config))
                        job_names.append(full_exp_name)
    else:
        # No finetuning - base experiments
        print(f"Creating base experiment jobs: {experiments_to_run}")
        
        for base_exp_name in experiments_to_run:
            # Apply IA early stopping suffix if needed
            full_exp_name = base_exp_name
            if use_ia_early_stopping and not full_exp_name.endswith('_IAES'):
                full_exp_name += '_IAES'
            
            if benchmarking and not full_exp_name.startswith('benchmarking_'):
                full_exp_name = 'benchmarking_' + full_exp_name

            # Check if already completed
            if check_existing_run(config['output_path'], full_exp_name, config):
                print(f'Skipping {full_exp_name} as already successfully trained')
                continue
                
            # Clean up any failed runs
            print(f'Cleaning up failed run for {full_exp_name}')
            cleanup_failed_run(config['output_path'], full_exp_name, config)
            
            # Create experiment config
            exp_config = create_experiment_config(
                full_exp_name,
                config['experiments'][base_exp_name],
                config,
                experiment_type,
                None,  # No finetuning config for base experiments
                use_ia_early_stopping=use_ia_early_stopping
            )
            experiment_configs.append((full_exp_name, exp_config))
            job_names.append(full_exp_name)
    
    # Sort jobs by configuration suffix
    sorted_experiment_configs = _sort_jobs_by_config(experiment_configs)
    sorted_job_names = [config[0] for config in sorted_experiment_configs]
    
    # Create and run worker manager
    if sorted_experiment_configs:
        print(f"Launching {len(sorted_experiment_configs)} jobs: {sorted_job_names}")
        manager = WorkerManager(gpus_to_use)
        try:
            manager.create_workers()
            manager.launch_jobs(sorted_experiment_configs, allow_crash)
            manager.wait_for_jobs(job_names)
        finally:
            manager.graceful_shutdown()
    else:
        print("No experiments to run - all already completed or skipped")

def _extract_config_suffix(job_name: str) -> str:
    """Extract the configuration suffix from a job name for sorting purposes"""
    # Job names now have format: base_exp_name + _ft_on_dataset + config_params
    # We want to extract the config_params part for sorting
    
    # Split by '_ft_on_' to separate the base part from the dataset+config part
    if '_ft_on_' in job_name:
        dataset_and_config = job_name.split('_ft_on_', 1)[1]
        # Split by the first '_' after dataset name to get just the config part
        parts = dataset_and_config.split('_', 1)
        if len(parts) > 1:
            # Everything after the dataset name is the config
            return parts[1]
        else:
            # No config parameters, just dataset
            return ''
    
    # Fallback: return empty string for base experiments
    return ''

def _sort_jobs_by_config(experiment_configs: List[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
    """Sort jobs by configuration suffix to group similar configs together"""
    # Sort by (config_suffix, dataset, base_experiment_name) for optimal GPU utilization
    def sort_key(job_item):
        job_name = job_item[0]
        config_suffix = _extract_config_suffix(job_name)
        
        # Extract dataset and base name using the new format
        if '_ft_on_' in job_name:
            base_name, dataset_and_config = job_name.split('_ft_on_', 1)
            dataset = dataset_and_config.split('_')[0]
        else:
            base_name = job_name
            dataset = ''
        
        return (config_suffix, dataset, base_name)
    
    return sorted(experiment_configs, key=sort_key)

def _generate_finetune_experiment_name(base_exp_name: str, dataset: str, finetune_config: FineTuningConfig, self_report_type: str) -> str:
    """Generate experiment name with finetuning parameters"""
    exp_name = base_exp_name
    if self_report_type != '' and self_report_type is not None:
        exp_name += f'_{self_report_type}'
    
    # Add dataset suffix first
    exp_name += f'_ft_on_{dataset}'
    
    # For orthogonal models using basis coefficients
    if finetune_config.use_basis_coefficients:
        exp_name += '_optimal_basis_coefficients'
        if finetune_config.map_act_val_separately:
            exp_name += '_separate_act_val'
    else:
        # Traditional approach: map to most similar annotator
        # Then add all finetuning parameters for clarity and consistency
        if finetune_config.num_annotators is not None:
            exp_name += f'_with_{finetune_config.num_annotators}_per_new_annotator'
        
        # Always show mapping parameters for individual annotator models
        if finetune_config.map_act_val_separately:
            exp_name += '_separate_act_val_mapping'
            
        if finetune_config.random_map:
            exp_name += '_random_map'
    
    # Always show the number of samples used for mapping
    if finetune_config.num_samples_to_use != 'all':
        exp_name += f'_{finetune_config.num_samples_to_use}_samples_used_for_map'
    
    return exp_name
