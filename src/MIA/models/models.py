import torch
import torch.nn as nn
import dataclasses
from .prediction_heads import KDE2DBaseline, IndividualAnnotators, DEERPrediction, AggregateHead
from transformers import AutoModel, AutoTokenizer
from MIA.config import ModelType

@dataclasses.dataclass
class ModelArguments:
    dense_layers: int
    layer_size: int
    prob_grid_size: int
    annotators: set
    training_precision: type = torch.float32
    kde_training_temperature: int = None
    kde_training_grid_size: int = None
    disable_kde: bool = False
    one_hot_annotators: bool = False
    determinism: bool = True
    determinism_type: str = 'default'
    epsilon_size: int = 200
    model_type: ModelType = ModelType.INDIVIDUAL_ANNOTATOR
    individual_annotator_output_size: int = 1
    condor: str = None
    basis_annotator_rank: int = None
    orthogonal_model_features: str = None
    flip_negative_values_in_validation: bool = False

class GenericModel(nn.Module):
    def __init__(self, model_config=None, **kwargs):
        super().__init__()
        
        # If model_config is provided, use it to initialize ModelArguments
        if model_config is not None:
            # Convert model_config to ModelArguments format
            model_args = {
                'model_type': model_config.model_type,
                'prob_grid_size': model_config.prob_grid_size,
                'dense_layers': model_config.dense_layers,
                'layer_size': model_config.layer_size,
                'annotators': kwargs.get('annotators', set()),  # Use annotators from kwargs
                'training_precision': model_config.training_precision or torch.float32,
                'determinism': model_config.determinism,
                'determinism_type': model_config.determinism_type,
                'disable_kde': model_config.disable_kde,
                'kde_training_temperature': model_config.kde_training_temperature,
                'kde_training_grid_size': model_config.kde_training_grid_size,
                'condor': model_config.condor,
                'basis_annotator_rank': model_config.basis_annotator_rank,
                'orthogonal_model_features': model_config.orthogonal_model_features,
                'flip_negative_values_in_validation': model_config.flip_negative_values_in_validation,
            }
            # Update with any additional kwargs
            model_args.update(kwargs)
            self.args = ModelArguments(**model_args)
        else:
            # Use provided kwargs directly
            self.args = ModelArguments(**kwargs)
            
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.args.model_type == ModelType.AGGREGATE_GROUND_TRUTH:
            return
        # Predict dist -> directly predicting distribution i.e. output layer is prob_grid_size*prob_grid_size
        if self.args.model_type != ModelType.KDE_2D:
            assert self.args.annotators is not None
            
            # Check if individual annotator models have annotators
            individual_annotator_models = [
                ModelType.INDIVIDUAL_ANNOTATOR, 
                ModelType.INDIVIDUAL_ANNOTATOR_PLUS, 
                ModelType.INDIVIDUAL_ANNOTATOR_BETA,
                ModelType.CLUSTERED,
                ModelType.ONE_HOT_ANNOTATORS,
                ModelType.INDIVIDUAL_ANNOTATOR_CONDOR,
                ModelType.INDIVIDUAL_ANNOTATOR_ORTHOGONAL,
                ModelType.INDIVIDUAL_ANNOTATOR_LOOP,
            ]
            
            if self.args.model_type in individual_annotator_models and len(self.args.annotators) == 0:
                raise ValueError(
                    f"Model type {self.args.model_type} requires annotators to be provided, "
                    f"but received empty annotators set. Please ensure annotators are extracted "
                    f"from the dataset and passed to the model constructor."
                )

        self.skip_random_observations = False
        self.text_layers = [
            nn.Dropout(p=0.2),
        ]
        self.audio_layers = [
            nn.Dropout(p=0.2),
        ]

        self.audio_layers = nn.ModuleList(self.audio_layers)
        self.text_layers = nn.ModuleList(self.text_layers)

        input_size = 768+768
        if self.args.model_type != ModelType.KDE_2D:
            self.number_of_annotators = len(self.args.annotators)

        if self.args.one_hot_annotators:
            input_size += self.number_of_annotators

        # In one-hot the audio and text will be concatenated at the same time as adding one-hot encoding
        self.combined_layers = [] if self.args.one_hot_annotators else [ConcatLayer()]

        # +1 to keep baseline layers the same count as the new method 
        num_fully_shared = self.args.dense_layers + 1 if self.args.model_type == ModelType.KDE_2D else self.args.dense_layers - 1 # -1 as there are 2 separate act/val layer at the end for non-baseline methods
        for _ in range(num_fully_shared):
            self.combined_layers.append(nn.Linear(input_size, self.args.layer_size))
            input_size = self.args.layer_size
            self.combined_layers.append(nn.ReLU())

        self.combined_layers = nn.Sequential(*self.combined_layers)
        if self.args.model_type == ModelType.KDE_2D:
            self.prediction_head = KDE2DBaseline(self.args)
        elif self.args.model_type in [ModelType.INDIVIDUAL_ANNOTATOR, ModelType.INDIVIDUAL_ANNOTATOR_PLUS, ModelType.CLUSTERED, ModelType.INDIVIDUAL_ANNOTATOR_LOOP]:
            # Always enable pretraining for individual annotator models, regardless of determinism
            self.pre_training = True
            self.pre_train_head_act = nn.Linear(in_features=self.args.layer_size, out_features=1)
            self.pre_train_head_val = nn.Linear(in_features=self.args.layer_size, out_features=1)
            if self.args.determinism and self.args.determinism_type == 'shared_layer':
                self.pre_train_head_act = nn.Linear(in_features=self.args.layer_size, out_features=2)
                self.pre_train_head_val = nn.Linear(in_features=self.args.layer_size, out_features=2)
            if self.args.determinism and self.args.determinism_type in ['default', 'include_in_output', 'utterance_variance']:
                self.pre_train_head_act_log_var = nn.Linear(in_features=self.args.layer_size, out_features=1)
                self.pre_train_head_val_log_var = nn.Linear(in_features=self.args.layer_size, out_features=1)
            self.prediction_head = IndividualAnnotators(self.args, use_loop_implementation=(self.args.model_type == ModelType.INDIVIDUAL_ANNOTATOR_LOOP))
        elif self.args.model_type == ModelType.INDIVIDUAL_ANNOTATOR_CONDOR:
            # For now hard code 6 for k-1 classes on seven class ordinal bins of podcast
            self.pre_training = False
            if self.args.model_type == ModelType.INDIVIDUAL_ANNOTATOR_CONDOR:
                # TODO: These maybe shouldn't just be hardcoded to 6 for podcast class size
                print(f'Condor type: {self.args.condor}')
                include_scale_parameter = self.args.condor is not None and 'with_scale_parameter' in self.args.condor
                if self.args.condor is None: # Default condor is to predict condor classes for each annotator
                    self.args.individual_annotator_output_size = 6
                    self.args.use_embeddings_for_individual_annotator = False
                elif self.args.condor == 'non_condor_categorical':
                    self.args.individual_annotator_output_size = 7 # 7 classes for categorical output, take advantage of condor code being highly similar to categorical prediction
                    self.args.use_embeddings_for_individual_annotator = False
                elif self.args.condor.replace('_with_scale_parameter', '') == 'shared_logits':
                    # When shared logits we want to predict the 6 classes using the same weights for all annotators
                    # then we use a static embedding for the bias of individual annotators
                    self.args.individual_annotator_output_size = 1
                    self.args.use_embeddings_for_individual_annotator = True
                    if include_scale_parameter:
                        self.args.individual_annotator_output_size += 1
                elif self.args.condor.replace('_with_scale_parameter', '') == 'shared_logits_input_dependent_bias':
                    # When shared logits input dependent bias we want to predict the 6 classes using the same weights for all annotators
                    # then we use an input dependent embedding for the bias of individual annotators
                    self.args.individual_annotator_output_size = 1 # Set to 1 for the bias output 
                    self.args.use_embeddings_for_individual_annotator = False
                    if include_scale_parameter:
                        self.args.individual_annotator_output_size += 1
                else:
                    raise ValueError(f'Invalid condor type: {self.args.condor}')
            self.prediction_head = IndividualAnnotators(self.args)
        elif self.args.model_type == ModelType.INDIVIDUAL_ANNOTATOR_ORTHOGONAL:
            # For orthogonal version we need to keep low rank basis annotator embeddings for the act and val heads
            num_basis_annotators = self.args.basis_annotator_rank

            # For each of the real annotators, we have an embedding that represents the weights for each of the basis annotators 
            self.args.individual_annotator_output_size = num_basis_annotators
            self.args.use_embeddings_for_individual_annotator = True
            # Now we need to keep a matrix of basis annotator embeddings for each of valence and activation
            self.basis_annotators_activation = nn.Parameter(torch.zeros(num_basis_annotators, self.args.layer_size, device=self.device))
            self.basis_annotators_valence = nn.Parameter(torch.zeros(num_basis_annotators, self.args.layer_size, device=self.device))
            # Want to initialise these to a very small and lightly orthogonal matrix 
            with torch.no_grad(): # QR decomposition of a random matrix, multiplied by 0.02 to ensure that weights start small 
                R_act = torch.randn(self.args.layer_size, num_basis_annotators, device=self.device)
                R_val = torch.randn(self.args.layer_size, num_basis_annotators, device=self.device)
                Q_act, _ = torch.linalg.qr(R_act)
                Q_val, _ = torch.linalg.qr(R_val)
                R_act = Q_act[:, :num_basis_annotators].T# * 0.02
                R_val = Q_val[:, :num_basis_annotators].T# * 0.02
            self.basis_annotators_activation.data = R_act
            self.basis_annotators_valence.data = R_val
            self.prediction_head = IndividualAnnotators(self.args, basis_annotators_activation=self.basis_annotators_activation, basis_annotators_valence=self.basis_annotators_valence)
            self.pre_training = True# if 'learn_consensus' in self.args.orthogonal_model_features else False
        elif self.args.model_type == ModelType.INDIVIDUAL_ANNOTATOR_BETA:
            self.prediction_head = IndividualAnnotators(self.args)
        elif self.args.model_type == ModelType.ONE_HOT_ANNOTATORS:
            self.prediction_head = OneHotAnnotators(self.args)
        elif self.args.model_type == ModelType.DEER:
            self.prediction_head = DEERPrediction(self.args) # Input is the concatenated 
        elif self.args.model_type == ModelType.BASELINE_AGGREGATE:
            self.prediction_head = AggregateHead(self.args)

    def finish_pre_train(self):
        # set the weights of prediction head to the weights of the pre-train head and then delete the pre-train head 
        with torch.no_grad():
            if self.args.model_type == ModelType.INDIVIDUAL_ANNOTATOR_ORTHOGONAL:
                print('No weights to copy when finishing pre-train for orthogonal model. Pre-training initialises orthogonal basis annotator embeddings')
            elif self.args.model_type == ModelType.INDIVIDUAL_ANNOTATOR_LOOP:
                # Loop-based implementation uses ModuleList of linear layers
                print(f'Copying {self.pre_train_head_act.weight.data.shape} into loop-based linear layers')
                for i, (act_layer, val_layer) in enumerate(zip(self.prediction_head.act_heads.linear_layers, 
                                                                 self.prediction_head.val_heads.linear_layers)):
                    act_layer.weight.data.copy_(self.pre_train_head_act.weight.data)
                    act_layer.bias.data.copy_(self.pre_train_head_act.bias.data)
                    val_layer.weight.data.copy_(self.pre_train_head_val.weight.data)
                    val_layer.bias.data.copy_(self.pre_train_head_val.bias.data)
            else:
                print(f'Copying {self.pre_train_head_act.weight.data.shape} into new shape {self.prediction_head.act_heads.weight.shape}')
                self.prediction_head.act_heads.weight[:,:,:] = self.pre_train_head_act.weight.data
                self.prediction_head.act_heads.bias[:,:] = self.pre_train_head_act.bias.data
                self.prediction_head.val_heads.weight[:,:,:] = self.pre_train_head_val.weight.data
                self.prediction_head.val_heads.bias[:,:] = self.pre_train_head_val.bias.data
                if self.args.determinism and self.args.determinism_type in ['default', 'include_in_output', 'utterance_variance']:
                    self.prediction_head.act_head_var.weight[:,:,:] = self.pre_train_head_act_log_var.weight.data
                    self.prediction_head.act_head_var.bias[:,:] = self.pre_train_head_act_log_var.bias.data
                    self.prediction_head.val_head_var.weight[:,:,:] = self.pre_train_head_val_log_var.weight.data
                    self.prediction_head.val_head_var.bias[:,:] = self.pre_train_head_val_log_var.bias.data

        self.pre_training = False

    def use_new_annotators(self, new_annotators, best_annotator_mapping, map_act_val_separately=False):
        if self.args.model_type not in [ModelType.INDIVIDUAL_ANNOTATOR, ModelType.INDIVIDUAL_ANNOTATOR_PLUS, ModelType.INDIVIDUAL_ANNOTATOR_CONDOR, ModelType.INDIVIDUAL_ANNOTATOR_ORTHOGONAL, ModelType.INDIVIDUAL_ANNOTATOR_LOOP]:
            return # Other model types don't track individual annotators identity so no changes needed
        # If model type has a generic annotator, use this to initialise the weights of the new heads
        # First need to find and store the generic annotator weights            
        if self.args.model_type == ModelType.INDIVIDUAL_ANNOTATOR_PLUS:
            mapper = self.prediction_head.act_heads.annotator_mapper
            mapper.set_get_idx()
            generic_annotator_idx = mapper['aggregate-annotator']
            with torch.no_grad():
                generic_weights = (
                    self.prediction_head.act_heads.weight[generic_annotator_idx, :, :],
                    self.prediction_head.act_heads.bias[generic_annotator_idx, :],
                    self.prediction_head.val_heads.weight[generic_annotator_idx, :, :],
                    self.prediction_head.val_heads.bias[generic_annotator_idx, :],
                )
        elif hasattr(self, 'pre_train_head_act'):
            with torch.no_grad():
                generic_weights = (
                    self.pre_train_head_act.weight.data,
                    self.pre_train_head_act.bias.data,
                    self.pre_train_head_val.weight.data,
                    self.pre_train_head_val.bias.data,
                )
        else:
            generic_weights = None
        # Create new prediction heads 
        self.args.annotators = new_annotators
        self.pre_training = False
        old_prediction_head = self.prediction_head
        output_size = 1
        def _is_basis_coeff_mapping(v):
            if isinstance(v, torch.Tensor):
                return True  # old joint coeffs
            if isinstance(v, dict):
                # old separate format
                if 'act' in v and isinstance(v['act'], torch.Tensor):
                    return True
                # new ridge+bias formats
                if 'act_w' in v and isinstance(v['act_w'], torch.Tensor):
                    return True
                if 'w' in v and isinstance(v['w'], torch.Tensor):
                    return True
            return False
        if best_annotator_mapping is not None:
            # Check if this is a basis coefficient mapping or traditional annotator index mapping

            first_mapping_value = next(iter(best_annotator_mapping.values()))
            is_basis_coefficient_mapping = _is_basis_coeff_mapping(first_mapping_value)
            
            # Only set output_size for traditional index-based mapping
            if not is_basis_coefficient_mapping:
                if map_act_val_separately:
                    # Can calculate lengths using only the activation mappings as valence will have the same size anyway 
                    lengths = torch.unique(torch.as_tensor([len(best_annotator_mapping[k]['act']) for k in best_annotator_mapping]))
                else:
                    lengths = torch.unique(torch.as_tensor([len(best_annotator_mapping[k]) for k in best_annotator_mapping]))
                assert len(lengths) == 1
                output_size = lengths.item()
                self.args.individual_annotator_output_size = output_size
                print(f'Mapping each 1 new annotator to {output_size} trained annotators')
            else:
                # For basis coefficients, output_size stays at default (the coefficients themselves)
                # For orthogonal models, this should be the number of basis annotators
                if self.args.model_type == ModelType.INDIVIDUAL_ANNOTATOR_ORTHOGONAL:
                    self.args.individual_annotator_output_size = self.args.basis_annotator_rank
                    print(f'Using optimal basis coefficients with {self.args.basis_annotator_rank} basis annotators')
        # For orthogonal models, pass the basis annotators from the old prediction head
        if self.args.model_type == ModelType.INDIVIDUAL_ANNOTATOR_ORTHOGONAL and hasattr(old_prediction_head, 'basis_annotators_activation'):
            self.prediction_head = IndividualAnnotators(
                self.args, 
                basis_annotators_activation=old_prediction_head.basis_annotators_activation,
                basis_annotators_valence=old_prediction_head.basis_annotators_valence
            )
        else:
            self.prediction_head = IndividualAnnotators(self.args)
        self.prediction_head = self.prediction_head.to(self.device)
        if best_annotator_mapping is not None:
            print(best_annotator_mapping.keys())
            
            # Check if this is a basis coefficient mapping (for orthogonal models)
            # by checking if the first value is a tensor (coefficients) vs int/list (indices)
            first_mapping_value = next(iter(best_annotator_mapping.values()))
            is_basis_coefficient_mapping = _is_basis_coeff_mapping(first_mapping_value)                
            if is_basis_coefficient_mapping and self.args.model_type == ModelType.INDIVIDUAL_ANNOTATOR_ORTHOGONAL:
                print("Initializing new annotators with basis coefficients (+ optional bias)")
                for ann_idx in self.prediction_head.act_heads.annotator_mapper.map_to_annotator:
                    ann_id = self.prediction_head.act_heads.annotator_mapper.map_to_annotator[ann_idx]
                    if ann_id not in best_annotator_mapping:
                        print(f'New annotator {ann_id} not mapped')
                        continue

                    m = best_annotator_mapping[ann_id]

                    with torch.no_grad():
                        if isinstance(m, torch.Tensor):
                            # OLD joint coeffs: coeffs
                            coeffs = m.to(self.device)
                            self.prediction_head.act_heads.embedding.weight[ann_idx, :] = coeffs
                            self.prediction_head.val_heads.embedding.weight[ann_idx, :] = coeffs
                            self.prediction_head.annotator_act_biases[ann_idx] = 0.0
                            self.prediction_head.annotator_val_biases[ann_idx] = 0.0

                        elif isinstance(m, dict) and 'act' in m and 'val' in m:
                            # OLD separate coeffs: {'act':..., 'val':...}
                            act_w = m['act'].to(self.device)
                            val_w = m['val'].to(self.device)
                            self.prediction_head.act_heads.embedding.weight[ann_idx, :] = act_w
                            self.prediction_head.val_heads.embedding.weight[ann_idx, :] = val_w
                            self.prediction_head.annotator_act_biases[ann_idx] = 0.0
                            self.prediction_head.annotator_val_biases[ann_idx] = 0.0

                        elif isinstance(m, dict) and 'act_w' in m and 'val_w' in m:
                            # NEW ridge+bias separate: {'act_w','act_b','val_w','val_b'}
                            act_w = m['act_w'].to(self.device)
                            val_w = m['val_w'].to(self.device)
                            act_b = float(m.get('act_b', 0.0))
                            val_b = float(m.get('val_b', 0.0))

                            self.prediction_head.act_heads.embedding.weight[ann_idx, :] = act_w
                            self.prediction_head.val_heads.embedding.weight[ann_idx, :] = val_w
                            self.prediction_head.annotator_act_biases[ann_idx] = act_b
                            self.prediction_head.annotator_val_biases[ann_idx] = val_b

                        elif isinstance(m, dict) and 'w' in m:
                            # NEW ridge+bias joint: {'w','b'}
                            coeffs = m['w'].to(self.device)
                            b = float(m.get('b', 0.0))

                            self.prediction_head.act_heads.embedding.weight[ann_idx, :] = coeffs
                            self.prediction_head.val_heads.embedding.weight[ann_idx, :] = coeffs
                            self.prediction_head.annotator_act_biases[ann_idx] = b
                            self.prediction_head.annotator_val_biases[ann_idx] = b

                        else:
                            raise ValueError(f"Unrecognized mapping format for annotator {ann_id}: keys={list(m.keys()) if isinstance(m, dict) else type(m)}")
            else:
                # Traditional approach: map to most similar annotator indices
                for ann_idx in self.prediction_head.act_heads.annotator_mapper.map_to_annotator:
                    ann_id = self.prediction_head.act_heads.annotator_mapper.map_to_annotator[ann_idx]
                    if ann_id in best_annotator_mapping:
                        if map_act_val_separately:
                            act_old_ann_idxs = best_annotator_mapping[ann_id]['act']
                            val_old_ann_idxs = best_annotator_mapping[ann_id]['val']
                            print(f'New annotator {ann_id}-activation assigned weight(s) for {act_old_ann_idxs}')
                            print(f'New annotator {ann_id}-valence assigned weight(s) for {val_old_ann_idxs}')
                            with torch.no_grad():
                                self.prediction_head.act_heads.weight[ann_idx,:,:], self.prediction_head.act_heads.bias[ann_idx,:] = stack_old_prediction_layers(old_prediction_head.act_heads.weight, old_prediction_head.act_heads.bias, act_old_ann_idxs)
                                self.prediction_head.val_heads.weight[ann_idx,:,:], self.prediction_head.val_heads.bias[ann_idx,:] = stack_old_prediction_layers(old_prediction_head.val_heads.weight, old_prediction_head.val_heads.bias, val_old_ann_idxs)
                        else:
                            old_ann_idxs = best_annotator_mapping[ann_id]
                            print(f'New annotator {ann_id} assigned weight(s) for {old_ann_idxs}')
                            with torch.no_grad():
                                self.prediction_head.act_heads.weight[ann_idx,:,:], self.prediction_head.act_heads.bias[ann_idx,:] = stack_old_prediction_layers(old_prediction_head.act_heads.weight, old_prediction_head.act_heads.bias, old_ann_idxs)
                                self.prediction_head.val_heads.weight[ann_idx,:,:], self.prediction_head.val_heads.bias[ann_idx,:] = stack_old_prediction_layers(old_prediction_head.val_heads.weight, old_prediction_head.val_heads.bias, old_ann_idxs)
                    else:
                        print(f'New annotator {ann_id} not mapped')
        elif generic_weights is not None:
            raise ValueError('Generic weights not supported for new annotators (disabled for now)')
            with torch.no_grad():
                act_weight, act_bias, val_weight, val_bias = generic_weights
                self.prediction_head.act_heads.weight[:,:,:] = act_weight
                self.prediction_head.act_heads.bias[:,:] = act_bias
                self.prediction_head.val_heads.weight[:,:,:] = val_weight
                self.prediction_head.val_heads.bias[:,:] = val_bias
        self.prediction_head.act_combined_layers = old_prediction_head.act_combined_layers
        self.prediction_head.val_combined_layers = old_prediction_head.val_combined_layers

    # def freeze_shared_layers(self):
    #     self.audio_layers.freeze()
    #     self.text_layers.freeze()
    #     self.combined_layers.freeze()
    #     self.prediction_head.act_combined_layers.freeze()
    #     self.prediction_head.val_combined_layers.freeze()
    #     self.frozen_annotator_idxs = []

    # def freeze_annotator(self, annotator_id):
    #     idx_to_freeze = self.prediction_head.act_heads.annotator_mapper.map_to_idx(annotator_id)
    #     self.frozen_annotator_idxs.append(idx_to_freeze) # Can't freeze parts of a tensor so we track these to write back the old weights after training opt step 

    def forward(self, audio, text, skip_kde=False, soft_hist=False, annotator_masks=None, curr_task=None):
        if self.args.model_type == ModelType.AGGREGATE_GROUND_TRUTH:
            # This is a fake model that should never actually be called during training
            # but if called during evaluation, it should return empty dict to be handled by evaluator
            return {}
        if self.args.disable_kde:
            skip_kde = True
        for layer in self.audio_layers:
            audio = layer(audio)
        for layer in self.text_layers:
            text = layer(text)

        if hasattr(self, 'pre_training') and self.pre_training:
            x = self.combined_layers((audio, text))
            act_x = self.prediction_head.act_combined_layers(x)
            val_x = self.prediction_head.val_combined_layers(x)
            if self.args.model_type == ModelType.INDIVIDUAL_ANNOTATOR_ORTHOGONAL:
                if 'learn_consensus' in self.args.orthogonal_model_features:
                    act = self.prediction_head.act_consensus_layer(act_x)
                    val = self.prediction_head.val_consensus_layer(val_x)
                else:
                    # Learn average basis perception -> consensus prediction to initialise the basis annotator embeddings
                    act = act_x @ self.prediction_head.basis_annotators_activation.transpose(-1,-2)
                    val = val_x @ self.prediction_head.basis_annotators_valence.transpose(-1,-2)
                    act = act.nanmean(dim=-1)
                    val = val.nanmean(dim=-1)
            else:
                act = self.pre_train_head_act(act_x)
                val = self.pre_train_head_val(val_x)
            out_values = {'act': act, 'val': val}
            if self.args.determinism and self.args.determinism_type == 'shared_layer':
                out_values = {'act': act[:,0], 'val': val[:,0], 'act_log_var': act[:,1], 'val_log_var': val[:,1]}
            if self.args.determinism and self.args.determinism_type in ['default', 'include_in_output', 'utterance_variance']:
                act_log_var = self.pre_train_head_act_log_var(act_x)
                val_log_var = self.pre_train_head_val_log_var(val_x)
                out_values['act_log_var'] = act_log_var
                out_values['val_log_var'] = val_log_var

            if self.args.model_type == ModelType.INDIVIDUAL_ANNOTATOR_ORTHOGONAL:
                out_values['basis_annotator_embeddings_activation'] = self.prediction_head.basis_annotators_activation
                out_values['basis_annotator_embeddings_valence'] = self.prediction_head.basis_annotators_valence

            return out_values

        if self.args.one_hot_annotators: # Audio and text are processed differently in this one case
            return self.prediction_head(audio, text, skip_kde, soft_hist, annotator_masks)

        x = self.combined_layers((audio, text))            
        out = self.prediction_head(x, annotator_masks, skip_kde, soft_hist)

        return out

def stack_old_prediction_layers(old_layer_weight, old_layer_bias, old_ann_idxs):
    if len(old_ann_idxs) == 1:
        return old_layer_weight[old_ann_idxs[0],:,:], old_layer_bias[old_ann_idxs[0],:]
    return torch.stack([old_layer_weight[idx,:,:] for idx in old_ann_idxs], dim=1), torch.stack([old_layer_bias[idx,:] for idx in old_ann_idxs], dim=1)

class ConcatLayer(nn.Module):
    def forward(self, x):
        # x should be a tuple of tensors to concatenate 
        return torch.cat(x, dim=1)