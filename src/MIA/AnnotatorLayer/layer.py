import torch
import torch.nn as nn
import torch.nn.functional as f
import math
from MIA.AnnotatorLayer.mapper import AnnotatorMapper

class AnnotatorOutputLayerLoop(nn.Module):
    """Loop-based implementation for benchmarking comparison"""
    def __init__(self, input_size, output_size, set_of_annotators=None, annotator_mapper=None, use_embeddings=False):
        super().__init__()
        self.use_embeddings = use_embeddings
        if set_of_annotators is not None:
            self.annotator_mapper = AnnotatorMapper(set_of_annotators)
        elif annotator_mapper is not None:
            self.annotator_mapper = annotator_mapper
        else:
            raise ValueError('Must provide either set_of_annotators or annotator_mapper argument')

        self.input_size = input_size
        self.output_size = output_size
        self.num_annotators = self.annotator_mapper.get_num_annotators()

        if self.use_embeddings:
            self.embedding = nn.Embedding(self.num_annotators, output_size)
        else:
            # Create individual linear layers for each annotator
            self.linear_layers = nn.ModuleList([
                nn.Linear(input_size, output_size) for _ in range(self.num_annotators)
            ])

    def forward(self, x, mask=None):
        if mask is None:
            if self.use_embeddings:
                return self.embedding.weight.squeeze()
            # Loop through all annotators
            outputs = []
            for i in range(self.num_annotators):
                outputs.append(self.linear_layers[i](x))
            return torch.stack(outputs, dim=1).squeeze()
        else:
            # Mask should be list of boolean masks which define which annotators to enable per batch
            batch_mask, weight_mask, output_mask, seen_annotator_mask = mask

            if self.use_embeddings:
                annotator_outputs = self.embedding(weight_mask).squeeze()
            else:
                # Loop through each active annotator
                # Extract the masked inputs once (same as batched version does with x[batch_mask])
                masked_x = x[batch_mask]
                annotator_outputs = []
                for i, annotator_idx in enumerate(weight_mask):
                    # Get the i-th input and corresponding annotator
                    # linear_layers[annotator_idx] computes: masked_x[i] @ W.T + b
                    output = self.linear_layers[annotator_idx](masked_x[i:i+1])
                    # Squeeze all singleton dimensions to match batched .squeeze() behavior
                    output = output.squeeze()
                    annotator_outputs.append(output)
                # Stack along first dimension to get (N, output_size) or (N,) if output_size=1
                annotator_outputs = torch.stack(annotator_outputs, dim=0)
                # After stacking, if output_size=1, we have (N,). If output_size>1, we have (N, output_size).
                # The batched version also does final .squeeze() which doesn't change these shapes.
            
            output_size = output_mask.shape if self.output_size == 1 else output_mask.shape + (self.output_size,)
            shaped_output = torch.fill(torch.empty(output_size, device=x.device), torch.nan)
            shaped_output[output_mask] = annotator_outputs

            return shaped_output

class AnnotatorOutputLayer(nn.Module):
    def __init__(self, input_size, output_size, set_of_annotators=None, annotator_mapper=None, use_embeddings=False):
        super().__init__()
        self.use_embeddings = use_embeddings
        if set_of_annotators is not None:
            self.annotator_mapper = AnnotatorMapper(set_of_annotators)
        elif annotator_mapper is not None:
            self.annotator_mapper = annotator_mapper
        else:
            raise ValueError('Must provide either set_of_annotators or annotator_mapper argument')

        self.input_size = input_size
        self.output_size = output_size
        self.num_annotators = self.annotator_mapper.get_num_annotators()

        if self.use_embeddings:
            self.embedding = nn.Embedding(self.num_annotators, output_size)
        else:
            self.weight = nn.Parameter(torch.empty((self.num_annotators, output_size, input_size)), requires_grad=True)
            self.bias = nn.Parameter(torch.empty((self.num_annotators, output_size)), requires_grad=True)

            # Initialise weights as if many linear layers
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, mask=None):
        if mask is None:
            if self.use_embeddings:
                return self.embedding.weight.squeeze()
            return f.linear(x, self.weight.view(-1,self.input_size), self.bias.view(-1)).view(x.shape[0],self.num_annotators,self.output_size).squeeze()
        else:
            # Mask should be list of boolean masks which define which annotators to enable per batch -- loop is limited to batch size iterations
            batch_mask, weight_mask, output_mask, seen_annotator_mask = mask

            if self.use_embeddings:
                annotator_outputs = self.embedding(weight_mask).squeeze()
            else:
                annotator_outputs = torch.bmm(x[batch_mask].unsqueeze(dim=1), self.weight[weight_mask].transpose(1,2)).squeeze() + self.bias[weight_mask].squeeze()
            output_size = output_mask.shape if self.output_size == 1 else output_mask.shape + (self.output_size,)
            shaped_output = torch.fill(torch.empty(output_size, device=x.device), torch.nan)
            shaped_output[output_mask] = annotator_outputs

            return shaped_output
