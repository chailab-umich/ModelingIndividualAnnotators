import torch
import torch.nn as nn
import torch.nn.functional as F
from MIA.data.kde_probability import kde_probability_bs
from MIA.AnnotatorLayer import AnnotatorOutputLayer, AnnotatorOutputLayerLoop
from MIA.config import ModelType
from .one_hot_layer import OneHotLayer
from ..loss_metrics.DEER import DenseNormalGamma

class KDE2DBaseline(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.prediction_head = nn.Linear(self.args.layer_size, self.args.prob_grid_size**2)

    # x = combined_layers((audio, text))
    def forward(self, x, *kwargs):
        output = {'probability_logits': self.prediction_head(x)}
        with torch.no_grad():
            output['probability_preds'] = torch.softmax(output['probability_logits'].view(output['probability_logits'].shape[0], -1), dim=-1)
            output['probability_preds'] = output['probability_preds'].view(output['probability_logits'].shape[0], self.args.prob_grid_size, self.args.prob_grid_size)
        return output


class CombinedLayers(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.act_combined_layers = nn.Sequential(
            nn.Linear(self.args.layer_size, self.args.layer_size),
            nn.ReLU(),
            nn.Linear(self.args.layer_size, self.args.layer_size),
            nn.ReLU(),
        )
        self.val_combined_layers = nn.Sequential(
            nn.Linear(self.args.layer_size, self.args.layer_size),
            nn.ReLU(),
            nn.Linear(self.args.layer_size, self.args.layer_size),
            nn.ReLU(),
        )

class IndividualAnnotators(CombinedLayers):
    def __init__(self, args, basis_annotators_activation=None, basis_annotators_valence=None, use_loop_implementation=False):
        super().__init__(args)
        self.use_loop_implementation = use_loop_implementation
        self.consensus_bias_only = False
        self.remove_annotator_specific_bias = False # mutually exclusive with consensus_bias_only
        self.basis_annotators_activation = basis_annotators_activation
        self.basis_annotators_valence = basis_annotators_valence
        # The journal IA_sigma^2 and IA_mu_sigma^2 checkpoints use the legacy
        # ``utterance_variance`` path below.  It is no longer used by newer
        # experiments, but it must remain constructible so published
        # checkpoints can be loaded and evaluated.
        if self.args.determinism and self.args.determinism_type != 'utterance_variance':
            raise NotImplementedError(
                'Only the publication-era utterance_variance determinism mode '
                'is supported by the restored evaluation path'
            )
        if self.args.determinism and self.args.determinism_type == 'shared_layer':
            self.args.individual_annotator_output_size = 2
        if self.args.model_type == ModelType.INDIVIDUAL_ANNOTATOR_CONDOR:
            if self.args.condor is None:
                self.args.individual_annotator_output_size = 6
                self.args.use_embeddings_for_individual_annotator = False
        self.output_size = self.args.individual_annotator_output_size
        if not hasattr(self.args, 'use_embeddings_for_individual_annotator'):
            self.args.use_embeddings_for_individual_annotator = False
        
        # Choose the appropriate layer implementation
        LayerClass = AnnotatorOutputLayerLoop if use_loop_implementation else AnnotatorOutputLayer
        
        self.act_heads = LayerClass(input_size=self.args.layer_size, output_size=self.args.individual_annotator_output_size, set_of_annotators=self.args.annotators, use_embeddings=self.args.use_embeddings_for_individual_annotator)
        self.val_heads = LayerClass(input_size=self.args.layer_size, output_size=self.args.individual_annotator_output_size, annotator_mapper=self.act_heads.annotator_mapper, use_embeddings=self.args.use_embeddings_for_individual_annotator)
        if self.args.model_type == ModelType.INDIVIDUAL_ANNOTATOR_CONDOR:
            if self.args.condor is not None and self.args.condor.replace('_with_scale_parameter', '') in ['shared_logits', 'shared_logits_input_dependent_bias']:
                self.act_condor_output = nn.Linear(in_features=self.args.layer_size, out_features=6)
                self.val_condor_output = nn.Linear(in_features=self.args.layer_size, out_features=6)
                self.include_scale_parameter = self.args.individual_annotator_output_size == 2
                if self.args.condor.replace('_with_scale_parameter', '') == 'shared_logits':
                    if self.include_scale_parameter:
                        nn.init.zeros_(self.act_heads.embedding.weight[:,0])
                        nn.init.zeros_(self.val_heads.embedding.weight[:,0])
                        nn.init.ones_(self.act_heads.embedding.weight[:,1])
                        nn.init.ones_(self.val_heads.embedding.weight[:,1])
                    else:
                        nn.init.zeros_(self.act_heads.embedding.weight)
                        nn.init.zeros_(self.val_heads.embedding.weight)
                    print('Shared logits shape:', self.act_heads.embedding.weight.shape, self.val_heads.embedding.weight.shape)
                elif self.args.condor.replace('_with_scale_parameter', '') == 'shared_logits_input_dependent_bias':
                    if self.include_scale_parameter:
                        nn.init.zeros_(self.act_heads.weight[:,0,:])
                        nn.init.zeros_(self.act_heads.bias[:,0])
                        nn.init.zeros_(self.val_heads.weight[:,0,:])
                        nn.init.zeros_(self.val_heads.bias[:,0])
                        nn.init.ones_(self.act_heads.weight[:,1,:])
                        nn.init.ones_(self.act_heads.bias[:,1])
                        nn.init.ones_(self.val_heads.weight[:,1,:])
                        nn.init.ones_(self.val_heads.bias[:,1])
                    else:
                        nn.init.zeros_(self.act_heads.weight)
                        nn.init.zeros_(self.act_heads.bias)
                        nn.init.zeros_(self.val_heads.weight)
                        nn.init.zeros_(self.val_heads.bias)
                    print('Shared logits input dependent bias shape:', self.act_heads.weight.shape, self.act_heads.bias.shape, self.val_heads.weight.shape, self.val_heads.bias.shape)
            elif self.args.condor is not None and self.args.condor == 'non_condor_categorical':
                LayerClass = AnnotatorOutputLayerLoop if self.use_loop_implementation else AnnotatorOutputLayer
                self.act_heads = LayerClass(input_size=self.args.layer_size, output_size=7, set_of_annotators=self.args.annotators)
                self.val_heads = LayerClass(input_size=self.args.layer_size, output_size=7, annotator_mapper=self.act_heads.annotator_mapper)
                if not self.use_loop_implementation:
                    print('Non condor categorical shape:', self.act_heads.weight.shape, self.act_heads.bias.shape, self.val_heads.weight.shape, self.val_heads.bias.shape)
        elif self.args.model_type == ModelType.INDIVIDUAL_ANNOTATOR_ORTHOGONAL:
            # Want to initialise the individual annotator weights for biases to zero so all annotators start from the consensus prediction
            with torch.no_grad():
                if 'init_weights_randomly' not in self.args.orthogonal_model_features: # If init weights randomly then we skip this step
                    if 'learn_consensus' in self.args.orthogonal_model_features:
                        self.act_heads.embedding.weight.data.zero_()
                        self.val_heads.embedding.weight.data.zero_()
                    else:
                        # To start from consensus prediction we actually want the output of the weights to be equivalent to nanmean
                        # So set to one then divide by the number of basis annotators
                        self.act_heads.embedding.weight.data.fill_(1.0)
                        self.val_heads.embedding.weight.data.fill_(1.0)
                        self.act_heads.embedding.weight.data /= self.args.basis_annotator_rank
                        self.val_heads.embedding.weight.data /= self.args.basis_annotator_rank

            # Biases for a linear layer are input independent, so these should always be nn.Parameter
            # Set these to zero to start with to prevent rapid changes in the biases, due to random initialisation
            self.annotator_act_biases = nn.Parameter(torch.zeros(self.act_heads.annotator_mapper.get_num_annotators(), device=self.device))
            self.annotator_val_biases = nn.Parameter(torch.zeros(self.val_heads.annotator_mapper.get_num_annotators(), device=self.device))
            # self.annotator_consensus_multiplier_act = nn.Parameter(torch.ones(self.act_heads.annotator_mapper.get_num_annotators(), device=self.device))
            # self.annotator_consensus_multiplier_val = nn.Parameter(torch.ones(self.val_heads.annotator_mapper.get_num_annotators(), device=self.device))
            # if 'learnable_consensus_multiplier' not in self.args.orthogonal_model_features:
            #     self.annotator_consensus_multiplier_act.requires_grad = False
            #     self.annotator_consensus_multiplier_val.requires_grad = False
            if 'learn_consensus' in self.args.orthogonal_model_features:
                self.act_consensus_layer = nn.Linear(in_features=self.args.layer_size, out_features=1)
                self.val_consensus_layer = nn.Linear(in_features=self.args.layer_size, out_features=1)
            if 'learn_consensus_plus_layernorm' in self.args.orthogonal_model_features:
                self.act_combined_layers.append(nn.LayerNorm(self.args.layer_size))
                self.val_combined_layers.append(nn.LayerNorm(self.args.layer_size))
            if 'initialised_from_IA' in self.args.orthogonal_model_features:
                # Initialization deferred to ExperimentRunner._initialize_model_trainer,
                # which calls self.initialize_from_ia() once the IA model path is resolved
                pass
            if 'input_dependent' in self.args.orthogonal_model_features:
                # Add the embedding of size 8 along with the multimodal input, so that foundational-annotator
                # routing is input dependent and annotator dependent. 
                self.input_dependent_act_layernorm = nn.LayerNorm(self.args.layer_size)
                self.input_dependent_val_layernorm = nn.LayerNorm(self.args.layer_size)
                self.input_dependent_act_embedding_layernorm = nn.LayerNorm(self.args.basis_annotator_rank)
                self.input_dependent_val_embedding_layernorm = nn.LayerNorm(self.args.basis_annotator_rank)
                self.model_weight_layer_act = nn.Linear(in_features=self.args.layer_size + self.args.basis_annotator_rank, out_features=self.args.basis_annotator_rank, bias=False)
                self.model_weight_layer_val = nn.Linear(in_features=self.args.layer_size + self.args.basis_annotator_rank, out_features=self.args.basis_annotator_rank, bias=False)
            if 'noisy_top_k_gating' in self.args.orthogonal_model_features:
                if 'input_dependent' not in self.args.orthogonal_model_features:
                    self.model_weight_layer_act = nn.Linear(in_features=self.args.basis_annotator_rank, out_features=self.args.basis_annotator_rank, bias=False)
                    self.model_weight_layer_val = nn.Linear(in_features=self.args.basis_annotator_rank, out_features=self.args.basis_annotator_rank, bias=False)
                self.act_noise = nn.Linear(in_features=self.args.layer_size + self.args.basis_annotator_rank, out_features=self.args.basis_annotator_rank, bias=False)
                self.val_noise = nn.Linear(in_features=self.args.layer_size + self.args.basis_annotator_rank, out_features=self.args.basis_annotator_rank, bias=False)
        if self.args.determinism:
            if self.args.determinism_type in ['default', 'include_in_output', 'utterance_variance']:
                LayerClass = AnnotatorOutputLayerLoop if self.use_loop_implementation else AnnotatorOutputLayer
                self.act_head_var = LayerClass(input_size=self.args.layer_size, output_size=args.individual_annotator_output_size, annotator_mapper=self.act_heads.annotator_mapper) # Predict a variance for the current evaluator
                self.val_head_var = LayerClass(input_size=self.args.layer_size, output_size=args.individual_annotator_output_size, annotator_mapper=self.act_heads.annotator_mapper)
                self.fixed_val_epsilon = torch.randn((self.act_heads.annotator_mapper.get_num_annotators(), self.args.epsilon_size), device=self.device)
                self.fixed_act_epsilon = torch.randn((self.act_heads.annotator_mapper.get_num_annotators(), self.args.epsilon_size), device=self.device)
            elif self.args.determinism_type == 'output_200':
                LayerClass = AnnotatorOutputLayerLoop if self.use_loop_implementation else AnnotatorOutputLayer
                self.act_heads = LayerClass(input_size=self.args.layer_size, output_size=200, set_of_annotators=self.args.annotators)
                self.val_heads = LayerClass(input_size=self.args.layer_size, output_size=200, annotator_mapper=self.act_heads.annotator_mapper)

    def initialize_from_ia(self, ia_prediction_head):
        """Initialize orthogonal model weights from a trained IA model's prediction head.

        Called by ExperimentRunner._initialize_model_trainer after model creation,
        when the IA model path can be resolved using the current seed.

        Args:
            ia_prediction_head: IndividualAnnotators prediction head from a trained IA model.
                - act_heads.weight: [num_annotators, 1, layer_size]
                - act_heads.bias:   [num_annotators, 1]
                - val_heads: same structure
                - act_combined_layers / val_combined_layers: shared dense layers
        """
        with torch.no_grad():
            self.act_combined_layers = ia_prediction_head.act_combined_layers
            self.val_combined_layers = ia_prediction_head.val_combined_layers
            act_weight = ia_prediction_head.act_heads.weight.data
            act_bias = ia_prediction_head.act_heads.bias
            val_weight = ia_prediction_head.val_heads.weight.data
            val_bias = ia_prediction_head.val_heads.bias
            # Decompose weights into basis and coefficients
            U, S, Vh = torch.linalg.svd(act_weight.squeeze(), full_matrices=False)
            orthogonal_weights = Vh[:self.args.basis_annotator_rank,:]
            W = U[:,:self.args.basis_annotator_rank] * S[:self.args.basis_annotator_rank]
            print(f'Assigning {orthogonal_weights.shape} to {self.basis_annotators_activation.shape}')
            # Need to ensure annotators are in the correct order
            this_annotator_mapper = self.act_heads.annotator_mapper
            old_annotator_mapper = ia_prediction_head.act_heads.annotator_mapper
            # Want to rearrange the old matrices to match the new order 
            new_order = [old_annotator_mapper.map_to_idx[this_annotator_mapper.map_to_annotator[i]] for i in this_annotator_mapper.map_to_annotator]
            self.basis_annotators_activation.data = orthogonal_weights
            print(f'Assigning {W.shape} to {self.act_heads.embedding.weight.shape}')
            self.act_heads.embedding.weight.data = W[new_order]
            print(f'Assigning {act_bias.shape} to {self.annotator_act_biases.shape}')
            self.annotator_act_biases.data = act_bias[new_order].squeeze()
            U, S, Vh = torch.linalg.svd(val_weight.squeeze(), full_matrices=False)
            orthogonal_weights = Vh[:self.args.basis_annotator_rank,:]
            W = U[:,:self.args.basis_annotator_rank] * S[:self.args.basis_annotator_rank]
            print(f'Assigning {orthogonal_weights.shape} to {self.basis_annotators_valence.shape}')
            self.basis_annotators_valence.data = orthogonal_weights
            print(f'Assigning {W.shape} to {self.val_heads.embedding.weight.shape}')
            self.val_heads.embedding.weight.data = W[new_order]
            print(f'Assigning {val_bias.shape} to {self.annotator_val_biases.shape}')
            self.annotator_val_biases.data = val_bias[new_order].squeeze()

    # x = combined_layers((audio, text))
    def forward(self, x, annotator_masks, skip_kde, soft_hist, *kwargs):
        act_x = self.act_combined_layers(x)
        val_x = self.val_combined_layers(x)
        if self.args.model_type == ModelType.INDIVIDUAL_ANNOTATOR and self.consensus_bias_only:
            # Skip annotator specific weights and just use the bias 
            _, weight_mask, output_mask, _ = mask
            annotator_outputs = self.act_heads.bias[weight_mask].squeeze()
            output_size = output_mask.shape if self.act_heads.output_size == 1 else output_mask.shape + (self.act_heads.output_size,)
            soft_act_labels = torch.fill(torch.empty(output_size, device=x.device), torch.nan)
            soft_act_labels[output_mask] = annotator_outputs
            annotator_outputs = self.val_heads.bias[weight_mask].squeeze()
            output_size = output_mask.shape if self.val_heads.output_size == 1 else output_mask.shape + (self.act_heads.output_size,)
            soft_val_labels = torch.fill(torch.empty(output_size, device=x.device), torch.nan)
            soft_val_labels[output_mask] = annotator_outputs
        elif self.args.model_type == ModelType.INDIVIDUAL_ANNOTATOR_ORTHOGONAL and self.remove_annotator_specific_bias:
            _, weight_mask, output_mask, _ = mask
            annotator_outputs = torch.bmm(act_x[batch_mask].unsqueeze(dim=1), self.act_heads.weight[weight_mask].transpose(1,2)).squeeze()
            output_size = output_mask.shape if self.act_heads.output_size == 1 else output_mask.shape + (self.act_heads.output_size,)
            soft_act_labels = torch.fill(torch.empty(output_size, device=x.device), torch.nan)
            soft_act_labels[output_mask] = annotator_outputs
            annotator_outputs = torch.bmm(val_x[batch_mask].unsqueeze(dim=1), self.val_heads.weight[weight_mask].transpose(1,2)).squeeze()
            output_size = output_mask.shape if self.val_heads.output_size == 1 else output_mask.shape + (self.act_heads.output_size,)
            soft_val_labels = torch.fill(torch.empty(output_size, device=x.device), torch.nan)
            soft_val_labels[output_mask] = annotator_outputs
        else:
            soft_act_labels = self.act_heads(act_x, annotator_masks).squeeze(-1)
            soft_val_labels = self.val_heads(val_x, annotator_masks).squeeze(-1)

        # Model is desired to be deterministic
        # Also want to train log variance prediction
        if self.args.determinism:
            if self.args.determinism_type in ['utterance_variance']:
                act_log_var = self.act_head_var(act_x, annotator_masks).squeeze(-1)
                val_log_var = self.val_head_var(val_x, annotator_masks).squeeze(-1)
                soft_act_z, soft_val_z = None, None
                # # During training we want to multiply the log_var with epsilon from a normal distribution of mean 0 variance 1 to introduce randomness
                # # however, we are keen to make the model deterministic, as such we use a fixed normal distribution for use in evaluation
                # # Shape will become batch x num evaluators x self.args.epsilon_size -- take epsilon_size observations of the log var
                # if act_log_var.requires_grad:
                #     act_epsilon = torch.randn(soft_act_labels.shape + (self.args.epsilon_size,), device=act_log_var.device)#if self.training else torch.ones_like(act_log_var)
                #     val_epsilon = torch.randn(soft_val_labels.shape + (self.args.epsilon_size,), device=act_log_var.device)#if self.training else torch.ones_like(act_log_var)
                # else:
                #     # In test/validation step, want to use a fixed epsilon to make model deterministic
                #     # Epsilon needs to be of shape batch x num evaluators x 200, fixed epsilon is a mapping of evaluator -> 200, so use the same approach as in prediction heads
                #     if annotator_masks is None:
                #         # Create 1 size batch dimension and then repeat to size of batch for later computation
                #         act_epsilon = self.fixed_act_epsilon[None,:,:].repeat(soft_act_labels.shape[0],1,1)
                #         val_epsilon = self.fixed_val_epsilon[None,:,:].repeat(soft_act_labels.shape[0],1,1)
                #     else:
                #         batch_mask, weight_mask, output_mask, seen_annotator_mask = annotator_masks
                #         act_epsilon = torch.fill(torch.empty(output_mask.shape + (self.args.epsilon_size,), device=x.device), torch.nan)
                #         val_epsilon = torch.fill(torch.empty(output_mask.shape + (self.args.epsilon_size,), device=x.device), torch.nan)
                #         act_epsilon[output_mask] = self.fixed_act_epsilon[weight_mask]
                #         val_epsilon[output_mask] = self.fixed_val_epsilon[weight_mask]

                # # Now we want to update soft_act_z and soft_val_z to create the observations 
                # soft_lbl_mask = soft_act_labels.isnan()
                # z_shape = soft_lbl_mask.shape
                # soft_act_z = torch.fill(torch.empty(z_shape + (self.args.epsilon_size,), device=soft_act_labels.device), torch.nan)
                # soft_val_z = torch.fill(torch.empty(z_shape + (self.args.epsilon_size,), device=soft_act_labels.device), torch.nan)
                # print(soft_act_labels.shape)
                # print(soft_lbl_mask.shape, soft_lbl_mask.any(), soft_lbl_mask.all())
                # if len(soft_lbl_mask.shape) < 2:
                #     print(soft_lbl_mask.sum(), soft_lbl_mask)
                # soft_act_z[~soft_lbl_mask] = soft_act_labels[~soft_lbl_mask].unsqueeze(dim=-1) + torch.exp(0.5*act_log_var[~soft_lbl_mask].unsqueeze(dim=-1))*act_epsilon[~soft_lbl_mask]
                # soft_val_z[~soft_lbl_mask] = soft_val_labels[~soft_lbl_mask].unsqueeze(dim=-1) + torch.exp(0.5*val_log_var[~soft_lbl_mask].unsqueeze(dim=-1))*val_epsilon[~soft_lbl_mask]
        else: # If not deterministic these values are just None
            act_log_var, val_log_var, soft_act_z, soft_val_z = None, None, None, None

        if self.args.model_type == ModelType.INDIVIDUAL_ANNOTATOR_CONDOR:
            # Don't want to do any individual annotator processing we should just return now 
            # However, we should calculate the predictions from the logits for storing in outputs
            from condor_pytorch.activations import ordinal_softmax
            # if self.args.condor is None:
                # Soft act labels are the logits already
                # No further processing needed unlike other methods that need to add bias to the shared outputs
            if self.args.condor is not None and self.args.condor.replace('_with_scale_parameter', '') in ['shared_logits', 'shared_logits_input_dependent_bias']:
                act_x = self.act_condor_output(act_x)
                val_x = self.val_condor_output(val_x)
                # Now we need to add the condor classes to the biases in the correct positions
                act_biases = soft_act_labels
                val_biases = soft_val_labels
                batch_mask, weight_mask, output_mask, seen_annotator_mask = annotator_masks
                new_output_size = output_mask.shape + (6,)
                # print('shaped_output_info', act_x.shape, act_x[batch_mask].shape, act_biases.shape, act_biases[~act_biases.isnan()].shape, output_mask.shape, output_mask.sum(), new_output_size)
                if self.include_scale_parameter:
                    act_scales = act_biases[...,1]
                    val_scales = val_biases[...,1]
                    act_biases = act_biases[...,0]
                    val_biases = val_biases[...,0]
                else:
                    act_scales = torch.ones_like(act_biases)
                    act_scales[act_biases.isnan()] = torch.nan
                    val_scales = torch.ones_like(val_biases)
                    val_scales[val_biases.isnan()] = torch.nan

                shaped_output_act = torch.full(new_output_size, device=x.device, fill_value=torch.nan)
                shaped_output_act[output_mask] = (act_x[batch_mask] / act_scales[~act_scales.isnan()].unsqueeze(dim=-1)) + act_biases[~act_biases.isnan()].unsqueeze(dim=-1)
                shaped_output_val = torch.full(new_output_size, device=x.device, fill_value=torch.nan)
                shaped_output_val[output_mask] = (val_x[batch_mask] / val_scales[~val_scales.isnan()].unsqueeze(dim=-1)) + val_biases[~val_biases.isnan()].unsqueeze(dim=-1)
                soft_act_labels = shaped_output_act
                soft_val_labels = shaped_output_val
                # soft_act_labels[~act_biases.isnan()] = act_biases[~act_biases.isnan()].unsqueeze(dim=-1) + act_x[batch_mask]
                # soft_val_labels[~val_biases.isnan()] = val_biases[~val_biases.isnan()].unsqueeze(dim=-1) + val_x[batch_mask]
                # print('After adding condor classes', soft_act_labels.shape, soft_val_labels.shape)
            mask = (~soft_act_labels.isnan()).sum(dim=-1).bool()
            soft_act_preds = torch.full(mask.shape, device=soft_act_labels.device, fill_value=torch.nan)
            soft_val_preds = torch.full(mask.shape, device=soft_val_labels.device, fill_value=torch.nan)
            
            if self.args.condor is not None and self.args.condor == 'non_condor_categorical':
                # No need to do anything for non condor categorical as the act and val heads are just predicting 7 logits at base, so we can just return now
                # Just need to skip the ordinal_softmax step 
                act_point_estimate = soft_act_labels.matmul(torch.tensor([[-1,-2/3,-1/3,0,1/3,2/3,1]], device=soft_act_labels.device).T)
                soft_act_preds[mask] = act_point_estimate[~act_point_estimate.isnan()]
                val_point_estimate = soft_val_labels.matmul(torch.tensor([[-1,-2/3,-1/3,0,1/3,2/3,1]], device=soft_val_labels.device).T)
                soft_val_preds[mask] = val_point_estimate[~val_point_estimate.isnan()]
            else:
                act_point_estimate = ordinal_softmax(soft_act_labels.view(-1, 6), soft_act_labels.device).matmul(torch.tensor([[-1,-2/3,-1/3,0,1/3,2/3,1]], device=soft_act_labels.device).T)
                soft_act_preds[mask] = act_point_estimate[~act_point_estimate.isnan()]
                val_point_estimate = ordinal_softmax(soft_val_labels.view(-1, 6), soft_val_labels.device).matmul(torch.tensor([[-1,-2/3,-1/3,0,1/3,2/3,1]], device=soft_val_labels.device).T)
                soft_val_preds[mask] = val_point_estimate[~val_point_estimate.isnan()]
            # print('act_point_estimate', soft_act_preds.shape, soft_val_preds.shape)

            return {
                'probability_logits': None,
                'mean_act_preds': soft_act_preds.nanmean(dim=1),
                'mean_val_preds': soft_val_preds.nanmean(dim=1),
                'soft_act_preds': soft_act_preds,
                'soft_val_preds': soft_val_preds,
                'act_probability_logits': soft_act_labels,
                'val_probability_logits': soft_val_labels,
                'act_z_log_var': (soft_act_z, act_log_var),
                'val_z_log_var': (soft_val_z, val_log_var),
            }

        if self.args.model_type == ModelType.INDIVIDUAL_ANNOTATOR_ORTHOGONAL:
            # Apply this step first so that if input-dependent is used it occurs before any softmax operations
            basis_weight_input_act = None
            basis_weight_input_val = None
            if 'input_dependent' in self.args.orthogonal_model_features:
                input_dependent_act = self.input_dependent_act_layernorm(act_x) # Will be [batch_size, layer_size]
                input_dependent_val = self.input_dependent_val_layernorm(val_x)
                valid_rows = ~soft_act_labels.isnan().any(dim=-1)
                if valid_rows.any():
                    input_dep_act_expanded = input_dependent_act[:,None,:].expand(-1, soft_act_labels.shape[1], -1)[valid_rows]
                    input_dep_val_expanded = input_dependent_val[:,None,:].expand(-1, soft_val_labels.shape[1], -1)[valid_rows]
                    # Concatenate the basis weights with the input dependent features
                    basis_weight_input_act = torch.cat((self.input_dependent_act_embedding_layernorm(soft_act_labels[valid_rows]), input_dep_act_expanded), dim=-1)
                    basis_weight_input_val = torch.cat((self.input_dependent_val_embedding_layernorm(soft_val_labels[valid_rows]), input_dep_val_expanded), dim=-1)
                    # Pass through linear to get the input-dependent and annotator-dependent basis weights
                    soft_act_labels[valid_rows] = self.model_weight_layer_act(basis_weight_input_act)
                    soft_val_labels[valid_rows] = self.model_weight_layer_val(basis_weight_input_val)

            fraction_selection_act = None
            fraction_selection_val = None
            if 'noisy_top_k_gating' in self.args.orthogonal_model_features:
                top_k_to_select = 2 # Based on DeM-MoE, future work can investigate this hyperparam
                valid_rows = ~soft_act_labels.isnan().any(dim=-1)
                if valid_rows.any():
                    # First add the noise from https://arxiv.org/pdf/1701.06538
                    if basis_weight_input_act is None:
                        raise ValueError('Basis weight input act and val should not be None if noisy top k gating is used. If you really want to use non-input-dependent then remove this value error check, but below code is not guaranteed to be correct.')
                        basis_weight_input_act = soft_act_labels[valid_rows].clone()
                        basis_weight_input_val = soft_val_labels[valid_rows].clone()
                        soft_act_labels[valid_rows] = self.model_weight_layer_act(soft_act_labels[valid_rows])
                        soft_val_labels[valid_rows] = self.model_weight_layer_val(soft_val_labels[valid_rows])
                    noisy_logits_act = soft_act_labels[valid_rows] + torch.randn_like(soft_act_labels[valid_rows]) * F.softplus(self.act_noise(basis_weight_input_act))
                    noisy_logits_val = soft_val_labels[valid_rows] + torch.randn_like(soft_val_labels[valid_rows]) * F.softplus(self.val_noise(basis_weight_input_val))
                    # Now I want to apply keep top k to each row of the basis weights 
                    top_k_vals, top_k_idx = torch.topk(noisy_logits_act, k=top_k_to_select, dim=-1)
                    top_k_vals_val, top_k_idx_val = torch.topk(noisy_logits_val, k=top_k_to_select, dim=-1)

                    # Now set values not in the top k to -inf
                    masked = torch.full_like(noisy_logits_act, fill_value=float('-inf'))
                    masked_val = torch.full_like(noisy_logits_val, fill_value=float('-inf'))
                    masked.scatter_(dim=-1, index=top_k_idx, src=top_k_vals)
                    masked_val.scatter_(dim=-1, index=top_k_idx_val, src=top_k_vals_val)
                    soft_act_labels[valid_rows] = masked
                    soft_val_labels[valid_rows] = masked_val
                    if 'add_load_balancing' in self.args.orthogonal_model_features:
                        K = self.args.basis_annotator_rank
                        fraction_selection_act = torch.zeros(K, device=soft_act_labels.device, dtype=soft_act_labels.dtype)
                        fraction_selection_val = torch.zeros(K, device=soft_val_labels.device, dtype=soft_val_labels.dtype)
                        
                        counts = torch.ones_like(top_k_idx.flatten(), dtype=fraction_selection_act.dtype)
                        fraction_selection_act.scatter_add_(0, top_k_idx.flatten(), counts)

                        total = counts.numel()  # N_valid * k
                        fraction_selection_act = fraction_selection_act / total
                        
                        counts = torch.ones_like(top_k_idx_val.flatten(), dtype=fraction_selection_val.dtype)
                        fraction_selection_val.scatter_add_(0, top_k_idx_val.flatten(), counts)

                        total = counts.numel()  # N_valid * k
                        fraction_selection_val = fraction_selection_val / total

            if 'add_softmax_to_weights' in self.args.orthogonal_model_features:
                # Only apply softmax to rows without any NaN values
                # If any value in a row is NaN, the entire row would become NaN after softmax
                valid_rows_act = ~soft_act_labels.isnan().any(dim=-1)
                valid_rows_val = ~soft_val_labels.isnan().any(dim=-1)
                if valid_rows_act.any():
                    if 'add_noise_to_softmax' in self.args.orthogonal_model_features:
                        noise = torch.randn_like(soft_act_labels[valid_rows_act]) * 0.5
                        soft_act_labels[valid_rows_act] = F.softmax(soft_act_labels[valid_rows_act] + noise, dim=-1)
                    else:
                        soft_act_labels[valid_rows_act] = F.softmax(soft_act_labels[valid_rows_act], dim=-1)
                if valid_rows_val.any():
                    if 'add_noise_to_softmax' in self.args.orthogonal_model_features:
                        noise = torch.randn_like(soft_val_labels[valid_rows_val]) * 0.5
                        soft_val_labels[valid_rows_val] = F.softmax(soft_val_labels[valid_rows_val] + noise, dim=-1)
                    else:
                        soft_val_labels[valid_rows_val] = F.softmax(soft_val_labels[valid_rows_val], dim=-1)

            importance_act = None
            importance_val = None
            if 'add_importance_balancing' in self.args.orthogonal_model_features:
                # Add load balancing as in https://arxiv.org/pdf/1701.06538
                # First calculate importance 
                valid_rows = ~soft_act_labels.isnan().any(dim=-1)
                importance_act = soft_act_labels[valid_rows].sum(dim=0)
                importance_val = soft_val_labels[valid_rows].sum(dim=0)
            p_selection_act = None
            p_selection_val = None
            if 'add_load_balancing' in self.args.orthogonal_model_features:
                valid_rows = ~soft_act_labels.isnan().any(dim=-1)
                p_selection_act = soft_act_labels[valid_rows].mean(dim=0)
                p_selection_val = soft_val_labels[valid_rows].mean(dim=0)

            # Need to get the masks for annotator-specific biases
            if annotator_masks is None: 
                print('ERROR THIS SHOULD NOT HAVE HAPPENED BUT AVOIDING CRASH, DEBUG THIS') # PLACEHOLDER
                # Get the basis annotator predictions
                # Basis weights are [total annotators in dataset, num_basis_annotators]
                act_basis_weights = soft_act_labels # The soft act preds and soft val preds will have returned the basis weights for each of the annotators
                val_basis_weights = soft_val_labels # Now need to make a prediction for each of the annotators using the basis weights
                if 'input_dependent' in self.args.orthogonal_model_features:
                    raise NotImplementedError('Input dependent not implemented yet')
                    #I think the below code is correct, but need to test it
                    # print('INPUT DEPENDENT SHAPES:', x.shape, x[batch_mask].shape)
                    # input_dependent_act = self.input_downsample_layer_act(x[batch_mask]) # x[batch_mask] is [num_annotators_in_batch, layer_size] -> [num_annotators_in_batch, 8]
                    # input_dependent_val = self.input_downsample_layer_val(x[batch_mask]) # x[batch_mask] is [num_annotators_in_batch, layer_size] -> [num_annotators_in_batch, 8]
                    # print('INPUT DEPENDENT SHAPES:', input_dependent_act.shape, input_dependent_val.shape)
                    # # act_basis_weights is [num_annotators_in_batch, num_basis_annotators]
                    # # Input dependent part needs to be added at the batch level
                    # # Goal: linear(8 input dependent features concatenated with 8 basis weights) -> 8 basis weights
                    # basis_weight_input_act = torch.cat((act_basis_weights, input_dependent_act), dim=-1)
                    # basis_weight_input_val = torch.cat((val_basis_weights, input_dependent_val), dim=-1)
                    # print('BASIS WEIGHT INPUT SHAPES:', basis_weight_input_act.shape, basis_weight_input_val.shape)
                    # basis_weights_act = self.model_weight_layer_act(basis_weight_input_act)
                    # basis_weights_val = self.model_weight_layer_val(basis_weight_input_val)
                    # print('BASIS WEIGHT OUTPUT SHAPES:', basis_weights_act.shape, basis_weights_val.shape)
                    # act_basis_weights = basis_weights_act
                    # val_basis_weights = basis_weights_val

                # Act predictions will be [batch_size, num_basis_annotators]
                basis_act_predictions = act_x @ self.basis_annotators_activation.transpose(-1,-2)
                basis_val_predictions = val_x @ self.basis_annotators_valence.transpose(-1,-2)
                # print('basis_act_predictions', basis_act_predictions.shape, 'basis_val_predictions', basis_val_predictions.shape, 'basis_weights', act_basis_weights.shape)
                final_act_predictions = torch.matmul(basis_act_predictions, act_basis_weights.transpose(-1,-2)).squeeze(-1)
                final_val_predictions = torch.matmul(basis_val_predictions, val_basis_weights.transpose(-1,-2)).squeeze(-1)

                if self.consensus_bias_only:
                    final_act_predictions.zero_()
                    final_val_predictions.zero_()

                # Now store the non-nan values in the correct positions in the soft_act_preds
                mean_act = None
                mean_val = None
                if 'learn_consensus' in self.args.orthogonal_model_features:
                    g0a = self.act_consensus_layer(act_x)
                    g0v = self.val_consensus_layer(val_x)
                    mean_act = g0a
                    mean_val = g0v
                    # print('g0a', g0a.shape, 'g0v', g0v.shape, 'annotator_consensus_multiplier_act', self.annotator_consensus_multiplier_act.shape, 'annotator_consensus_multiplier_val', self.annotator_consensus_multiplier_val.shape)
                    # act_consensus_addition = g0a * self.annotator_consensus_multiplier_act
                    # val_consensus_addition = g0v * self.annotator_consensus_multiplier_val
                    # print('act_consensus_addition', act_consensus_addition.shape, 'val_consensus_addition', val_consensus_addition.shape)
                    final_act_predictions = final_act_predictions + act_consensus_addition
                    final_val_predictions = final_val_predictions + val_consensus_addition
                    # print('final_act_predictions', final_act_predictions.shape, 'final_val_predictions', final_val_predictions.shape)

                # Add annotator-specific biases
                if not self.remove_annotator_specific_bias:
                    final_act_predictions = final_act_predictions + self.annotator_act_biases
                    final_val_predictions = final_val_predictions + self.annotator_val_biases

                soft_act_preds = final_act_predictions
                soft_val_preds = final_val_predictions

                # Now we can return the results, and also return the basis embeddings for orthogonality loss
                return_vals = {
                    'probability_logits': None,
                    'mean_act_preds': soft_act_preds.nanmean(dim=1) if mean_act is None else mean_act,
                    'mean_val_preds': soft_val_preds.nanmean(dim=1) if mean_val is None else mean_val,
                    'soft_act_preds': soft_act_preds,
                    'soft_val_preds': soft_val_preds,
                    'basis_annotator_embeddings_activation': self.basis_annotators_activation,
                    'basis_annotator_embeddings_valence': self.basis_annotators_valence,
                    'basis_act_predictions': basis_act_predictions,
                    'basis_val_predictions': basis_val_predictions,
                }
                if 'learn_consensus' in self.args.orthogonal_model_features:
                    return_vals['consensus_act_predictions'] = g0a
                    return_vals['consensus_val_predictions'] = g0v
                return return_vals

            batch_mask, weight_mask, output_mask, seen_annotator_mask = annotator_masks

            # act_basis_weights and val_basis_weights shapes are both [batch_size, num_annotators_in_batch, num_basis_annotators]
            act_basis_weights = soft_act_labels # The soft act preds and soft val preds will have returned the basis weights for each of the annotators
            val_basis_weights = soft_val_labels # Now need to make a prediction for each of the annotators using the basis weights

            annotator_act_biases = torch.full((act_basis_weights.shape[0], act_basis_weights.shape[1]), device=act_basis_weights.device, fill_value=torch.nan)
            annotator_val_biases = torch.full((act_basis_weights.shape[0], act_basis_weights.shape[1]), device=act_basis_weights.device, fill_value=torch.nan)
            annotator_act_biases[output_mask] = self.annotator_act_biases[weight_mask]
            annotator_val_biases[output_mask] = self.annotator_val_biases[weight_mask]

            disable_basis_weights = False
            disable_basis_matrices = False
            disable_biases = False
            if 'learn_consensus_disable_all_but_consensus' == self.args.orthogonal_model_features:
                disable_basis_weights = True
                disable_basis_matrices = True
                disable_biases = True
            if 'learn_consensus_enable_only_consensus_and_bias' == self.args.orthogonal_model_features:
                disable_basis_weights = True
                disable_basis_matrices = True
                disable_biases = False
            if 'learn_consensus_enable_only_consensus_bias_and_coefficients' == self.args.orthogonal_model_features:
                disable_basis_weights = False
                disable_basis_matrices = True
                disable_biases = False
            if 'learn_consensus_disable_only_bias' == self.args.orthogonal_model_features:
                disable_basis_weights = False
                disable_basis_matrices = False
                disable_biases = True

            if self.remove_annotator_specific_bias:
                disable_biases = True

            if disable_basis_weights:
                act_basis_weights[~act_basis_weights.isnan()] = torch.zeros_like(act_basis_weights)[~act_basis_weights.isnan()]
                val_basis_weights[~val_basis_weights.isnan()] = torch.zeros_like(val_basis_weights)[~val_basis_weights.isnan()]
            if disable_basis_matrices:
                self.basis_annotators_activation.data = torch.cat((torch.eye(self.basis_annotators_activation.shape[0], device=self.basis_annotators_activation.device), torch.zeros((self.basis_annotators_activation.shape[0], self.args.layer_size-self.basis_annotators_activation.shape[0]), device=self.basis_annotators_activation.device)), dim=-1)
                self.basis_annotators_valence.data = torch.cat((torch.eye(self.basis_annotators_valence.shape[0], device=self.basis_annotators_valence.device), torch.zeros((self.basis_annotators_valence.shape[0], self.args.layer_size-self.basis_annotators_valence.shape[0]), device=self.basis_annotators_valence.device)), dim=-1)

            if disable_biases:
                annotator_act_biases[output_mask] = torch.zeros_like(annotator_act_biases[output_mask])
                annotator_val_biases[output_mask] = torch.zeros_like(annotator_val_biases[output_mask])

            # Get prediction for all basis annotators
            # act_x is [batch_size, layer_size] and basis_annotators_activation is [num_basis_annotators, layer_size]
            # so we need to transpose the basis_annotators_activation to [layer_size, num_basis_annotators]
            # basis_act_predictions and basis_val_predictions shapes are both then [batch_size, num_basis_annotators]
            if 'detach_residuals' in self.args.orthogonal_model_features:
                if not 'learn_consensus' in self.args.orthogonal_model_features:
                    raise ValueError('detach_residuals requires learn_consensus to be enabled')
                basis_act_predictions = act_x.detach() @ self.basis_annotators_activation.transpose(-1,-2)
                basis_val_predictions = val_x.detach() @ self.basis_annotators_valence.transpose(-1,-2)
            else:
                basis_act_predictions = act_x @ self.basis_annotators_activation.transpose(-1,-2)
                basis_val_predictions = val_x @ self.basis_annotators_valence.transpose(-1,-2)

            # print('Model forward | basis annotators activation:', self.basis_annotators_activation)

            # Now we need to make a prediction for each of the annotators using the basis weights
            # act_basis_weights is [batch_size, num_annotators_in_batch, num_basis_annotators] and basis_act_predictions is [batch_size, num_basis_annotators]
            # so we need to do a batch matrix multiplication, where per item in batch shapes will be [num_annotators_in_batch, num_basis_annotators] and [num_basis_annotators, 1]
            # Can't simply do bmm as the basis weights contain NaN rows for annotators that did not annotator a given sample 
            # Easiest to flatten the first row and then do a standard matmul, before reshaping back to the original shape 
            soft_act_preds = torch.full((act_basis_weights.shape[0], act_basis_weights.shape[1]), device=act_basis_weights.device, fill_value=torch.nan)
            soft_val_preds = torch.full((act_basis_weights.shape[0], act_basis_weights.shape[1]), device=act_basis_weights.device, fill_value=torch.nan)

            # Remove NaNs to make bmm safe 
            # if 'add_softmax_to_weights' not in self.args.orthogonal_model_features: # These are already 0 in softmax case 
            act_basis_weights[act_basis_weights.isnan()] = 0
            val_basis_weights[val_basis_weights.isnan()] = 0

            # Basis weights are both of shape [batch size, num_annotators_in_batch, num_basis_annotators] so we simply matmul here to get [batch size, num_annotators_in_batch, 1]
            final_act_predictions = torch.bmm(act_basis_weights, basis_act_predictions.unsqueeze(dim=-1)).squeeze(-1)
            final_val_predictions = torch.bmm(val_basis_weights, basis_val_predictions.unsqueeze(dim=-1)).squeeze(-1)

            if self.consensus_bias_only:
                final_act_predictions.zero_()
                final_val_predictions.zero_()

            # Now store the non-nan values in the correct positions in the soft_act_preds
            mean_act_preds = None
            mean_val_preds = None
            if 'learn_consensus' in self.args.orthogonal_model_features:
                g0a = self.act_consensus_layer(act_x)
                g0v = self.val_consensus_layer(val_x)
                mean_act_preds = g0a
                mean_val_preds = g0v
                # print('Adding the consensus multiplier to the predictions:', final_act_predictions.shape, g0a.shape, self.annotator_consensus_multiplier_act[weight_mask].shape)
                act_consensus_addition = torch.full(final_act_predictions.shape, device=final_act_predictions.device, fill_value=0.0) # Unused spots don't need to be nan as these are filtered out during assignment later to soft_act_preds
                val_consensus_addition = torch.full(final_val_predictions.shape, device=final_val_predictions.device, fill_value=0.0)
                act_consensus_addition[:,:] = g0a
                val_consensus_addition[:,:] = g0v
                # act_consensus_addition[output_mask] = act_consensus_addition[output_mask] * self.annotator_consensus_multiplier_act[weight_mask]
                # val_consensus_addition[output_mask] = val_consensus_addition[output_mask] * self.annotator_consensus_multiplier_val[weight_mask]
                # print('Consensus addition shapes:', act_consensus_addition.shape, val_consensus_addition.shape)
                if 'detach_residuals' in self.args.orthogonal_model_features:
                    act_consensus_addition = act_consensus_addition.detach()
                    val_consensus_addition = val_consensus_addition.detach()
                final_act_predictions = final_act_predictions + act_consensus_addition
                final_val_predictions = final_val_predictions + val_consensus_addition

            # Now add the annotator-specific biases to the predictions
            final_act_predictions[output_mask] += annotator_act_biases[output_mask]
            final_val_predictions[output_mask] += annotator_val_biases[output_mask]

            soft_act_preds[output_mask] = final_act_predictions[output_mask]
            soft_val_preds[output_mask] = final_val_predictions[output_mask]

            # Now we can return the results, and also return the basis embeddings for orthogonality loss
            return_vals = {
                'probability_logits': None,
                'mean_act_preds': soft_act_preds.nanmean(dim=1) if mean_act_preds is None else mean_act_preds,
                'mean_val_preds': soft_val_preds.nanmean(dim=1) if mean_val_preds is None else mean_val_preds,
                'soft_act_preds': soft_act_preds,
                'soft_val_preds': soft_val_preds,
                'basis_annotator_embeddings_activation': self.basis_annotators_activation,
                'basis_annotator_embeddings_valence': self.basis_annotators_valence,
                'basis_act_predictions': basis_act_predictions,
                'basis_val_predictions': basis_val_predictions,
            }
            if soft_act_preds.requires_grad: # Don't return these during inference because we only need them for training loss and padding is a nightmare that isn't worth handling
                return_vals['act_basis_weights'] = act_basis_weights
                return_vals['val_basis_weights'] = val_basis_weights
            if 'learn_consensus' in self.args.orthogonal_model_features:
                return_vals['consensus_act_predictions'] = g0a
                return_vals['consensus_val_predictions'] = g0v
            if 'add_importance_balancing' in self.args.orthogonal_model_features:
                return_vals['importance_act'] = importance_act
                return_vals['importance_val'] = importance_val
            if 'add_load_balancing' in self.args.orthogonal_model_features:
                return_vals['fraction_selection_act'] = fraction_selection_act
                return_vals['fraction_selection_val'] = fraction_selection_val
                return_vals['p_selection_act'] = p_selection_act
                return_vals['p_selection_val'] = p_selection_val
            return return_vals

        # If model is in eval mode AND there are more than 1 output_size then we want to average the prediction into one prediction label per annotator
        if not self.training and self.output_size > 1:
            if self.args.determinism:
                raise NotImplementedError('Not yet implemented eval mode for deterministic with output_size > 1')
            soft_act_labels = soft_act_labels.mean(dim=-1)
            soft_val_labels = soft_val_labels.mean(dim=-1)

        return process_individual_annotator_output(soft_act_labels, soft_val_labels, act_log_var, val_log_var, soft_act_z, soft_val_z, self.args, self.training, x.shape[0], skip_kde, soft_hist)

class DEERPrediction(CombinedLayers):
    def __init__(self, args, separate_heads=False):
        super().__init__(args)
        self.separate_heads = separate_heads
        if self.separate_heads:
            self.act_evidential_layer = DenseNormalGamma(self.args.layer_size, 1)
            self.val_evidential_layer = DenseNormalGamma(self.args.layer_size, 1)
        else:
            self.evidential_layer = DenseNormalGamma(2*self.args.layer_size, 2)

    def forward(self, x, *kwargs):
        act_x = self.act_combined_layers(x)
        val_x = self.val_combined_layers(x)
        if self.separate_heads:
            act_out = self.act_evidential_layer(act_x)
            val_out = self.val_evidential_layer(val_x)
            gamma_a, v_a, alpha_a, beta_a = torch.split(act_out, int(act_out.shape[-1]/4), dim=-1)  #gamma.shape: [batch_size,1]
            print('separate heads activation size check:', gamma_a.shape, v_a.shape, alpha_a.shape, beta_a.shape)
            gamma_v, v_v, alpha_v, beta_v = torch.split(val_out, int(val_out.shape[-1]/4), dim=-1)  #gamma.shape: [batch_size,1]
            print('separate heads valence size check:', gamma_v.shape, v_v.shape, alpha_v.shape, beta_v.shape)
            gamma = torch.stack((gamma_a, gamma_v), dim=-1)
            v = torch.stack((v_a, v_v), dim=-1)
            alpha = torch.stack((alpha_a, alpha_v), dim=-1)
            beta = torch.stack((beta_a, beta_v), dim=-1)
        else:
            out = self.evidential_layer(torch.cat((act_x, val_x), dim=-1))
            gamma, v, alpha, beta = torch.split(out, int(out.shape[-1]/4), dim=-1)  #gamma.shape: [batch_size,1]
        return {'gamma': gamma, 'v': v, 'alpha': alpha, 'beta': beta}

class AggregateHead(CombinedLayers):
    def __init__(self, args):
        super().__init__(args)
        self.act_head = nn.Linear(self.args.layer_size, 1)
        self.val_head = nn.Linear(self.args.layer_size, 1)

    def forward(self, x, *kwargs):
        act_x = self.act_combined_layers(x)
        val_x = self.val_combined_layers(x)
        act = self.act_head(act_x)
        val = self.val_head(val_x)
        return {'act': act, 'val': val}

def process_individual_annotator_output(soft_act_labels, soft_val_labels, act_log_var, val_log_var, soft_act_z, soft_val_z, args, is_training, batch_size, skip_kde=False, use_soft_hist=False, test_precision=torch.float64):
        # For single prediction head and head per real annotator this should be the same processing steps
        # Get average of annotators for each sample in batch ignoring nan values
        # If we get a batch_size equal to the prediction shape we need to unsqueeze to the right (-1) -> 1 annotator
        # if we get a batch_size of 1 we need to squeeze to the left (0) -> 1 sample with many annotators
        squeeze_direction = 0 if batch_size == 1 else -1
        if len(soft_act_labels.shape) == 0: # If only one annotator enabled this turns into a scalar tensor during evaluation so unsqueeze twice
            soft_act_labels = soft_act_labels.unsqueeze(dim=squeeze_direction)
            soft_val_labels = soft_val_labels.unsqueeze(dim=squeeze_direction)
        if len(soft_act_labels.shape) == 1:
            soft_act_labels = soft_act_labels.unsqueeze(dim=squeeze_direction)
            soft_val_labels = soft_val_labels.unsqueeze(dim=squeeze_direction)

        mean_act = soft_act_labels.nanmean(dim=1)  # Keep on same device as soft_act_labels for training loss
        mean_val = soft_val_labels.nanmean(dim=1)

        if not skip_kde:
            # Don't use soft histogram if during validation step
            precision = args.training_precision if is_training else test_precision
            use_soft_histogram = is_training or use_soft_hist
            density_grid_size = args.kde_training_grid_size if use_soft_histogram else 512
            skip_observation = bool(args.determinism)
            skip_observation = False # Disable skipping observation to make KDE the same in all models (for now)
            if skip_observation:
                kde_act_input = soft_act_z.nanmean(dim=1).squeeze()
                kde_val_input = soft_val_z.nanmean(dim=1).squeeze()
                if torch.isnan(kde_act_input).any() or torch.isnan(kde_val_input).any():
                    print('Warning: NaN values in kde input used for skipping observation step')
            else:
                kde_act_input = soft_act_labels
                kde_val_input = soft_val_labels
            if skip_observation:
                print('skipping observation step and using raw model outputs in kde inputting z parameter of shape:', soft_act_z.shape)

            kde_probs = kde_probability_bs(kde_act_input, kde_val_input, temperature=args.kde_training_temperature, density_grid_size=density_grid_size, prob_grid_size=args.prob_grid_size, use_soft_histogram=use_soft_histogram, precision=precision, skip_observations=skip_observation) # Returns logits
            kde_2d_prob = kde_probs.view(kde_probs.shape[0],-1)# - kde_2d_prob.view(curr_bs,-1).min(dim=-1).values.unsqueeze(dim=-1)
            kde_2d_prob = kde_2d_prob / kde_2d_prob.sum(dim=-1).unsqueeze(dim=-1)
            kde_2d_prob = kde_2d_prob.view(kde_probs.shape[0],kde_probs.shape[1],kde_probs.shape[2]).float()

            return {
                'probability_logits': kde_probs,
                'probability_preds': kde_2d_prob,
                'mean_act_preds': mean_act,
                'mean_val_preds': mean_val,
                'soft_act_preds': soft_act_labels,
                'soft_val_preds': soft_val_labels,
                'act_z_log_var': (soft_act_z, act_log_var),
                'val_z_log_var': (soft_val_z, val_log_var),
            }
        return {
            'probability_logits': None,
            'mean_act_preds': mean_act,
            'mean_val_preds': mean_val,
            'soft_act_preds': soft_act_labels,
            'soft_val_preds': soft_val_labels,
            'act_z_log_var': (soft_act_z, act_log_var),
            'val_z_log_var': (soft_val_z, val_log_var),
        }
