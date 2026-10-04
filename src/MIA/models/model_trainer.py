import torch
import torch.optim as optim
from MIA.csv_logging import CSVWriter
from collections import defaultdict
from MIA.loss_metrics import *
from .models import GenericModel, ModelType
from .early_stopping import EarlyStopping
from .utils import load_torch_model
import os

class ModelTrainer:
    def __init__(self, prob_grid_size, model_name, model_config, model_save_path, training_tasks, csv_writer=None, annotators=None):
        self.training_tasks = training_tasks
        self.prob_grid_size = prob_grid_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if training_tasks == 'baseline':
            model_config.model_type = ModelType.KDE_2D
            
        # Create model config with annotators
        model_kwargs = {}
        if annotators is not None:
            model_kwargs['annotators'] = annotators
            print(f"ModelTrainer: Passing {len(annotators)} annotators to model: {sorted(list(annotators))}")
        else:
            print("ModelTrainer: No annotators provided, using empty set")
            
        self.model = GenericModel(model_config=model_config, **model_kwargs).to(self.device)

        self.name = model_name
        if model_config.model_type == ModelType.AGGREGATE_GROUND_TRUTH:
            self.optimizer = None
            self.early_stopping = None
            self.lr_scheduler = None,
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9)
            self.early_stopping = EarlyStopping(f'{model_save_path}/{model_name}')
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=5)
        self.continue_training = True
        self.current_lr = None
        self.logger = csv_writer
        self.global_batch_step = -1
        self.epoch_losses = defaultdict(list)

        # Define loss functions
        training_task_mapping = {
            'baseline': BaselineLoss,
            'baseline_aggregate': lambda **kwargs: PreTrainLoss(**{**kwargs, 'train_log_var': False, 'loss_fn': 'CCC'}),
            'task1': lambda **kwargs: Task1Loss(**kwargs, loss_fn='CCC'),
            'task1cccind': lambda **kwargs: CCCIndLoss(**kwargs, loss_fn='CCC'),
            'task1pearson': lambda **kwargs: Task1Loss(**kwargs, loss_fn='pearson'),
            'task1cccind': lambda **kwargs: CCCIndLoss(**kwargs, loss_fn='CCC'),
            'CONDOR': lambda **kwargs: CONDORLoss(**kwargs),
            'CONDORtask1combined': lambda **kwargs: CONDORTask1CombinedLoss(**kwargs),
            'task1CORAL5': lambda **kwargs: CRPS_Discrete(**kwargs, num_ordinal_bins=5),
            'task1CORAL9': lambda **kwargs: CRPS_Discrete(**kwargs, num_ordinal_bins=9),
            'task1cccweighted': lambda **kwargs: Task1Loss(**kwargs, loss_fn='weighted_ccc'),
            'task1softmaxccc5': lambda **kwargs: Task1Loss(**kwargs, loss_fn='softmax5_ccc'),
            'task1softmaxccc9': lambda **kwargs: Task1Loss(**kwargs, loss_fn='softmax9_ccc'),
            # 'task1IAES': lambda **kwargs: Task1LossIAEarlyStopping(**kwargs, loss_fn='CCC'),
            'task1-logvarloss': lambda **kwargs: LogVarSimpleLoss(**kwargs, model_determinism_type='utterance_variance'),
            'task2': lambda **kwargs: KLDivRegularisationLoss(**kwargs, model_determinism_type='utterance_variance'),
            'task2ccc': lambda **kwargs: LogVarSimpleLoss(**kwargs, model_determinism_type='utterance_variance'),
            'task3': Task3Loss,
            'deer_loss': FullDEERLoss,
            'orthogonal': lambda **kwargs: OrthogonalLoss(**kwargs),
            'categorical': CategoricalLoss,
            'orthogonal_plus_ccc': lambda **kwargs: OrthogonalPlusCCCLoss(**kwargs),
            'true_orthogonal_plus_ccc': lambda **kwargs: TrueOrthogonalPlusCCCLoss(**kwargs),
            'true_orthogonal_plus_ccc_plus_wavlm_inspired_loss': lambda **kwargs: TrueOrthogonalPlusCCCLossPlussWavLMInspiredLoss(**kwargs),
            'consensus_and_ccc_ind': lambda **kwargs: ConsensusAndCCCIndLoss(**kwargs, use_consensus=True, use_ccc_ind=True),
            'consensus_and_ccc_ind_ccc_flat_loss': lambda **kwargs: ConsensusAndCCCIndLoss(**kwargs, use_consensus=True, use_ccc_ind=True, use_ccc_flat_loss=True),
            'consensus_and_ccc_ind_with_minimum_validation_count': lambda **kwargs: ConsensusAndCCCIndLoss(**kwargs, use_consensus=True, use_ccc_ind=True, use_minimum_validation_count=32),
            'true_orthogonal_plus_ccc_ccc_ind': lambda **kwargs: ConsensusAndCCCIndLossPlusOrthogonalLoss(**kwargs, use_consensus=True, use_ccc_ind=True),
            'true_orthogonal_plus_ccc_ccc_ind_ccc_flat_loss': lambda **kwargs: ConsensusAndCCCIndLossPlusOrthogonalLoss(**kwargs, use_consensus=True, use_ccc_ind=True, use_ccc_flat_loss=True),
            'true_orthogonal_plus_ccc_ccc_ind_with_minimum_validation_count': lambda **kwargs: ConsensusAndCCCIndLossPlusOrthogonalLoss(**kwargs, use_consensus=True, use_ccc_ind=True, use_minimum_validation_count=32),
            'ccc_ind_only': lambda **kwargs: ConsensusAndCCCIndLoss(**kwargs, use_consensus=False, use_ccc_ind=True),
            'true_orthogonal_plus_ccc_ccc_ind_only': lambda **kwargs: ConsensusAndCCCIndLossPlusOrthogonalLoss(**kwargs, use_consensus=False, use_ccc_ind=True),
            # PARADIGM LOSSES:
            'ccc_ind_ccc_cons_paradigm_1': lambda **kwargs: ConsensusAndCCCIndLoss(**kwargs, use_consensus=True, use_ccc_ind=True, use_ccc_flat_loss=False),
            'ccc_ind_ccc_cons_paradigm_1_orthogonal': lambda **kwargs: ConsensusAndCCCIndLossPlusOrthogonalLoss(**kwargs, use_consensus=True, use_ccc_ind=True, use_ccc_flat_loss=False),
            'ccc_ind_ccc_cons_paradigm_1_orthogonal_add_importance_balancing': lambda **kwargs: ConsensusAndCCCIndLossPlusOrthogonalLoss(**kwargs, use_consensus=True, use_ccc_ind=True, use_ccc_flat_loss=False, add_importance_balancing=True),
            'ccc_ind_ccc_cons_paradigm_1_orthogonal_decov': lambda **kwargs: ConsensusAndCCCIndLossPlusOrthogonalLoss(**kwargs, use_consensus=True, use_ccc_ind=True, use_ccc_flat_loss=False, decov_loss_on_outputs=True), # Decov is just a shorthand term from where the idea came initially, but it is orthogonality on the output as in DeM-MoE
            'ccc_flat_ccc_cons_paradigm_2': lambda **kwargs: ConsensusAndCCCIndLoss(**kwargs, use_consensus=True, use_ccc_flat_loss=True, use_ccc_ind=False),
            'ccc_flat_ccc_cons_paradigm_2_orthogonal': lambda **kwargs: ConsensusAndCCCIndLossPlusOrthogonalLoss(**kwargs, use_consensus=True, use_ccc_flat_loss=True, use_ccc_ind=False),
            'ccc_flat_ccc_cons_paradigm_2_orthogonal_add_importance_balancing': lambda **kwargs: ConsensusAndCCCIndLossPlusOrthogonalLoss(**kwargs, use_consensus=True, use_ccc_flat_loss=True, use_ccc_ind=False, add_importance_balancing=True),
            'ccc_flat_ccc_cons_paradigm_2_orthogonal_decov': lambda **kwargs: ConsensusAndCCCIndLossPlusOrthogonalLoss(**kwargs, use_consensus=True, use_ccc_flat_loss=True, use_ccc_ind=False, decov_loss_on_outputs=True),
            'ccc_ind_only_paradigm_3': lambda **kwargs: ConsensusAndCCCIndLoss(**kwargs, use_consensus=False, use_ccc_ind=True, use_ccc_flat_loss=False),
            'ccc_conc_only_paradigm_3': lambda **kwargs: ConsensusAndCCCIndLoss(**kwargs, use_consensus=True, use_ccc_ind=False, use_ccc_flat_loss=False),
            'ccc_ind_only_paradigm_3_orthogonal': lambda **kwargs: ConsensusAndCCCIndLossPlusOrthogonalLoss(**kwargs, use_consensus=False, use_ccc_ind=True, use_ccc_flat_loss=False),
            'ccc_conc_only_paradigm_3_orthogonal': lambda **kwargs: ConsensusAndCCCIndLossPlusOrthogonalLoss(**kwargs, use_consensus=True, use_ccc_ind=False, use_ccc_flat_loss=False),
            'ccc_ind_only_paradigm_3_orthogonal_add_importance_balancing': lambda **kwargs: ConsensusAndCCCIndLossPlusOrthogonalLoss(**kwargs, use_consensus=False, use_ccc_ind=True, use_ccc_flat_loss=False, add_importance_balancing=True),
            'ccc_conc_only_paradigm_3_orthogonal_add_importance_balancing': lambda **kwargs: ConsensusAndCCCIndLossPlusOrthogonalLoss(**kwargs, use_consensus=True, use_ccc_ind=False, use_ccc_flat_loss=False, add_importance_balancing=True),
            'ccc_ind_only_paradigm_3_orthogonal_add_importance_balancing_load_balancing': lambda **kwargs: ConsensusAndCCCIndLossPlusOrthogonalLoss(**kwargs, use_consensus=False, use_ccc_ind=True, use_ccc_flat_loss=False, add_importance_balancing=True, add_load_balancing=True),
            'ccc_conc_only_paradigm_3_orthogonal_add_importance_balancing_load_balancing': lambda **kwargs: ConsensusAndCCCIndLossPlusOrthogonalLoss(**kwargs, use_consensus=True, use_ccc_ind=False, use_ccc_flat_loss=False, add_importance_balancing=True, add_load_balancing=True),
            'ccc_ind_only_paradigm_3_orthogonal_decov': lambda **kwargs: ConsensusAndCCCIndLossPlusOrthogonalLoss(**kwargs, use_consensus=False, use_ccc_ind=True, use_ccc_flat_loss=False, decov_loss_on_outputs=True),
            'ccc_conc_only_paradigm_3_orthogonal_decov': lambda **kwargs: ConsensusAndCCCIndLossPlusOrthogonalLoss(**kwargs, use_consensus=True, use_ccc_ind=False, use_ccc_flat_loss=False, decov_loss_on_outputs=True),
        }
        if 'task1cccind' in training_tasks:
            self.ringbuffer_size = 8
            self.ringbuffers = {
                'act': torch.fill(torch.empty((len(self.model.prediction_head.act_heads.annotator_mapper.map_to_idx), self.ringbuffer_size)), torch.nan).to(self.device),
                'val': torch.fill(torch.empty((len(self.model.prediction_head.act_heads.annotator_mapper.map_to_idx), self.ringbuffer_size)), torch.nan).to(self.device),
                'act_ground_truth': torch.fill(torch.empty((len(self.model.prediction_head.act_heads.annotator_mapper.map_to_idx), self.ringbuffer_size)), torch.nan).to(self.device),
                'val_ground_truth': torch.fill(torch.empty((len(self.model.prediction_head.act_heads.annotator_mapper.map_to_idx), self.ringbuffer_size)), torch.nan).to(self.device),
                'indices': torch.zeros((len(self.model.prediction_head.act_heads.annotator_mapper.map_to_idx),), dtype=torch.long).to(self.device),
            }
            for ringbuffer in self.ringbuffers.values():
                ringbuffer.requires_grad = False

        training_tasks = training_tasks.split(',')
        print(f"ModelTrainer init - Model type: {self.model.args.model_type}")
        print(f"ModelTrainer init - Training tasks: {training_tasks}")
        print(f"ModelTrainer init - Determinism: {self.model.args.determinism}")
        print(f"ModelTrainer init - Has pre_training: {hasattr(self.model, 'pre_training')}")
        if hasattr(self.model, 'pre_training'):
            print(f"ModelTrainer init - Pre_training value: {self.model.pre_training}")
        
        if self.model.args.determinism and self.model.args.model_type == ModelType.INDIVIDUAL_ANNOTATOR and 'task2' not in training_tasks:
            training_tasks.append('task1-logvarloss')
            print('Adding simple log-var loss to train log-var predictions as provided config does not train log-var parameters.')
        self.loss_fns = []
        for task in training_tasks:
            if task == '':
                continue # Empty task string
            sparse = '_sparse' in task
            later = '_later' in task
            task = task.replace('_sparse', '').replace('_later', '')
            # Sparse -> once every 10 epochs, later -> use loss after 20 epochs of training
            task_args = {'name': task, 'calculate_kde': task=='task3', 'sparsity': 10 if sparse else None, 'after_warmup': 20 if later else False}
            print(f"Creating loss function for task: {task}")
            self.loss_fns.append(training_task_mapping[task](**task_args))
        
        print(f"ModelTrainer init - Final loss functions: {[loss_fn.name for loss_fn in self.loss_fns]}")

    def save_special_checkpoint(self, checkpoint_type):
        """Save a special checkpoint for zero-shot or one-shot evaluation.
        
        Args:
            checkpoint_type: Either 'zero-shot' or 'one-shot'
        """
        if self.model.args.model_type == ModelType.AGGREGATE_GROUND_TRUTH:
            print('[INFO]: Not saving checkpoint -- Aggregate Ground Truth has no "model".')
            return
            
        save_location = os.path.join(self.early_stopping.model_save_loc, f'{checkpoint_type}-checkpoint.ckpt')
        print(f'Saving {checkpoint_type} checkpoint to {save_location}')
        from .utils import save_torch_model
        save_torch_model(self.model, save_location)

    def load_best_model(self, model_path=None, checkpoint_type=None):
        if self.model.args.model_type == ModelType.AGGREGATE_GROUND_TRUTH:
            print('[INFO]: Not loading model weights -- Aggregate Ground Truth has no "model".')
            return
            
        if model_path is None:
            # No path provided - use early stopping to get best model path
            model_path = self.early_stopping.get_best_model_path()
        elif checkpoint_type is not None:
            # Load a specific special checkpoint
            model_path = os.path.join(model_path, f'{checkpoint_type}-checkpoint.ckpt')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f'Warning: {checkpoint_type} checkpoint not found at {model_path}')
            print(f'Loading {checkpoint_type} checkpoint from {model_path}')
        elif os.path.isdir(model_path):
            # Directory path provided (test_only mode) - find checkpoint with maximum epoch number
            # Exclude special checkpoints from regular model selection
            checkpoint_files = [f for f in os.listdir(model_path) if f.endswith('_cp.ckpt') and f not in ['one_cp.ckpt', 'zero_cp.ckpt', 'zero-shot-checkpoint.ckpt', 'one-shot-checkpoint.ckpt']]
            if not checkpoint_files:
                raise FileNotFoundError(f"No checkpoint files found in directory: {model_path}")
            
            # Extract epoch numbers and find maximum
            epoch_numbers = []
            for filename in checkpoint_files:
                try:
                    epoch_num = int(filename.replace('_cp.ckpt', ''))
                    epoch_numbers.append(epoch_num)
                except ValueError:
                    continue  # Skip files that don't match the expected pattern
            
            if not epoch_numbers:
                raise FileNotFoundError(f"No valid checkpoint files found in directory: {model_path}")
                
            max_epoch = max(epoch_numbers)
            model_path = os.path.join(model_path, f'{max_epoch}_cp.ckpt')
            print(f'Loading best model from test_only mode: {model_path} (epoch {max_epoch})')
        
        # Load the model
        disable_kde = self.model.args.disable_kde # This argument makes no difference to model architecture, so it shouldn't change by loading other model weights
        self.model = load_torch_model(GenericModel, model_path, True).to(self.device)
        self.model.args.disable_kde = disable_kde
        self.model.pre_training = False # Loaded model should not be in pre-training mode

    def reset_optim(self):
        if self.model.args.model_type == ModelType.AGGREGATE_GROUND_TRUTH:
            self.optimizer = None
            self.lr_scheduler = None,
            self.current_lr = None
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9)
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=5)
            self.current_lr = None
        self.continue_training = True

    def change_prob_grid_size(self, new_size):
        print('Warning may not work depending on model parameters')
        self.model.prob_grid_size = new_size
        self.prob_grid_size = new_size

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def reset_epoch_losses(self):
        self.epoch_losses = defaultdict(list)

    def training_step(self, audio, text, targets, epoch, current_batch_type=None):
        # Skip training for aggregate ground truth model
        if self.model.args.model_type == ModelType.AGGREGATE_GROUND_TRUTH:
            return 'Aggregate ground truth: no training needed'
            
        # Check if we're in pretraining mode and have pretrain_loss
        if hasattr(self.model, 'pre_training') and self.model.pre_training and hasattr(self, 'pretrain_loss'):
            loss_fns_to_use = [self.pretrain_loss]
            if self.model.args.model_type == ModelType.INDIVIDUAL_ANNOTATOR_ORTHOGONAL and not ('learn_consensus' in self.model.args.orthogonal_model_features):
                task_args = {'name': 'orthogonal pretraining', 'calculate_kde': False, 'sparsity': None, 'after_warmup': False} # Don't use this for learn_consensus as in that case we just learn the consensus weights
                loss_fns_to_use.append(OrthogonalLoss(**task_args)) # Orthogonal loss is only used for orthogonal model
        else:
            loss_fns_to_use = self.loss_fns

        # Track batch number for logging and sparse losses 
        self.global_batch_step += 1

        # Generate masks
        # Only need to use act annotators as we are just checking for non-nan values and this will match with val annotators 
        masks = targets['annotator_masks'] if 'annotator_masks' in targets else None # Set to none if not in targets

        curr_loss_str = ''
        for i, loss_fn in enumerate(loss_fns_to_use):
            if loss_fn.sparsity is not None and self.global_batch_step % loss_fn.sparsity:
                continue # Skip this loss function as we are only calculating this every sparsity batches
            if loss_fn.after_warmup and epoch < loss_fn.after_warmup:
                continue # Skip this loss function as we are only calculating this after the model has trained that many epochs
            if loss_fn.validation_only:
                continue # Skip this loss as it is only calculated for validation set
            if loss_fn.name.startswith('ccc_ind_only_paradigm_3') and current_batch_type == 'consensus':
                continue # Skip the CCC ind only loss when using the consensus batch
            if loss_fn.name.startswith('ccc_conc_only_paradigm_3') and current_batch_type == 'ccc_ind':
                continue # Skip the CCC conc only loss when using the CCC ind batch
            model_output = self.model(audio, text, skip_kde=not loss_fn.calculate_kde, annotator_masks=masks, curr_task=loss_fn.name) # curr_task only used for logging the last seen task during model output explosion to diagnose culprit
            if loss_fn.name == 'task1cccind':
                targets['ringbuffers'] = self.ringbuffers
            # if loss_fn.name == 'task1CORAL5' or loss_fn.name == 'task1CORAL9':
            #     targets['CORALProcesses'] = self.model.coral_processes
            losses = loss_fn(model_output, targets, masks)
            for loss_name in list(losses.keys()):
                if losses[loss_name] is None:
                    del losses[loss_name]
                    continue # Brent optimisation failed so skip this batch for this loss function
                l = losses[loss_name].item()
                self.logger.log_scalar('Train', f'{loss_fn.name}-batch_loss-{loss_name}', self.name, l, step=self.global_batch_step)
                self.epoch_losses[f'{loss_fn.name}-{loss_name}'].append(l)
            if len(losses):
                loss = sum(losses.values())/len(losses)
                loss.backward()
            else:
                return f'{loss_fn.name}: failed'

            # KL-Div losses need to clip the gradients to prevent explosion
            if any(['KL-Div' in loss_name for loss_name in losses.keys()]):
                if not hasattr(self, 'has_printed_kl_div_warning'):
                    self.has_printed_kl_div_warning = True
                    print('Clipping KL-Div gradients to prevent explosion')
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, error_if_nonfinite=False)
                if not torch.isfinite(total_norm):
                    self.optimizer.zero_grad(set_to_none=True)
                    print(f'{loss_fn.name}: NaN in gradients, skipping batch')
                    continue

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            key_val_strs = [f'{k}={v.item():.3f}' for k, v in losses.items()]
            curr_loss_str += f'{loss_fn.name}: {", ".join(key_val_strs)} '

        return curr_loss_str

    def log_average_loss(self, epoch):
        for loss_type in self.epoch_losses:
            average = sum(self.epoch_losses[loss_type])/len(self.epoch_losses[loss_type])
            self.logger.log_scalar(f'train ({self.prob_grid_size}x{self.prob_grid_size})', f'batch_loss_{loss_type}', self.name, average, step=epoch)
        self.reset_epoch_losses()
    
    def get_validation_loss_display(self):
        """Get formatted string for validation loss display in tqdm using early stopping results."""
        if self.early_stopping is None or len(self.early_stopping.previous_results) == 0:
            return ""
        
        # Get last validation loss (most recent)
        last_val_loss = self.early_stopping.previous_results[-1]
        display_parts = [f"last_val: {last_val_loss:.4f}"]
        
        # Get best validation loss based on early stopping mode
        if self.early_stopping.mode == 'min':
            best_val_loss = min(self.early_stopping.previous_results)
        elif self.early_stopping.mode == 'max':
            best_val_loss = max(self.early_stopping.previous_results)
        else:
            best_val_loss = None
            
        if best_val_loss is not None:
            display_parts.append(f"best_val: {best_val_loss:.4f}")
        
        return " | " + " | ".join(display_parts)
