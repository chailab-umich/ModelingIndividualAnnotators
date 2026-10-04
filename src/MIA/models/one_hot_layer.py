import torch
import torch.nn as nn
import torch.nn.functional as f
import math
from MIA.AnnotatorLayer.mapper import AnnotatorMapper

# This will one-hot encode the annotators and concatenate them to the layers
# duplicating the layers to make a prediction per annotator present in each batch
class OneHotLayer(nn.Module):
    def __init__(self, set_of_annotators=None, annotator_mapper=None):
        super().__init__()
        if set_of_annotators is not None:
            self.annotator_mapper = AnnotatorMapper(set_of_annotators)
        elif annotator_mapper is not None:
            self.annotator_mapper = annotator_mapper
        else:
            raise ValueError('Must provide either set_of_annotators or annotator_mapper argument')

        # Map the index of annotator -> a one hot encoding representing that index
        # this way we can use the same batch collation and tricks from predicting individual annotator layers
        # i.e. instead of indexing weight vectors by annotator id, we are indexing one-hot encodings by annotator id
        enc_length = self.annotator_mapper.get_num_annotators()
        self.one_hot_encodings = torch.nn.functional.one_hot(torch.as_tensor(list(range(enc_length))), num_classes=enc_length)
        if torch.cuda.is_available():
            self.one_hot_encodings = self.one_hot_encodings.cuda()

    # When there are many annotators the model may run out of memory
    def forward(self, x, mask=None):
        if mask is None:
            # Want to make a prediction for every annotator for every batch 
            # so need to duplicate the encodings to repeat for each item in batch
            # and duplicate each batch to repeat for each encoding
            num_annotators = self.annotator_mapper.get_num_annotators()
            batch_size = x.shape[0]
            one_hot_encodings = self.one_hot_encodings.repeat(batch_size,1)
            x = x.repeat(num_annotators, 1)
            concat_batch = torch.cat((x, one_hot_encodings), dim=1)
            act_shaped_output = torch.fill(torch.empty(batch_size, num_annotators, device='cpu'), torch.nan) # Put these on CPU as when using many annotators memory usage may be very large
            val_shaped_output = torch.fill(torch.empty(batch_size, num_annotators, device='cpu'), torch.nan)
            shaped_outputs = (act_shaped_output, val_shaped_output)
            output_mask = torch.fill(torch.empty(batch_size, num_annotators, device=x.device), True).bool()
        else:
            # Mask should be list of boolean masks which define which annotators to enable per batch -- loop is limited to batch size iterations
            batch_mask, weight_mask, output_mask, seen_annotator_mask = mask
            duplic_batch = x[batch_mask] # Will duplicate batch accordingly 
            one_hot_encodings = self.one_hot_encodings[weight_mask] # Will get corresponding number of annotator one-hot encodings 
            concat_batch = torch.cat((duplic_batch, one_hot_encodings), dim=1)
            act_shaped_output = torch.fill(torch.empty(output_mask.shape, device=x.device), torch.nan)
            val_shaped_output = torch.fill(torch.empty(output_mask.shape, device=x.device), torch.nan)
            shaped_outputs = (act_shaped_output, val_shaped_output)

        return concat_batch, shaped_outputs, output_mask # Return the shaped_output and output_mask alongside results to shape output once making predictions
