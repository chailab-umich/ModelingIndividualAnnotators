# from MIA.data import BatchCollator as SERCollator
import torch
from MIA.AnnotatorLayer import AnnotatorBatchCollator

class BatchCollator:
    def __init__(self, sample_types=None, annotator_mapper=None, pad_soft_labels=False, include_aggregate=False):
        # self.main_collator = SERCollator(sample_types)
        self.pad_soft_labels = pad_soft_labels
        self.batch_annotators = annotator_mapper is not None
        if annotator_mapper is not None:
            if self.pad_soft_labels:
                raise ValueError('Batch collator cannot pad labels and use annotator mapper as the mapper will pad labels already')
            self.annotator_collator = AnnotatorBatchCollator(annotator_mapper, include_aggregate_annotator=include_aggregate)

    def __call__(self, samples):
        # Helper function to safely calculate variance
        def safe_var(tensor):
            if tensor.numel() == 1:  # Single element
                return torch.tensor(0.0, device=tensor.device)  # Variance of single element is 0
            else:
                # Correction = 0 or else the variance will have size n dependent unbiasing during training that could be problematic for the loss 
                return tensor.var(correction=0) # Tensor with no elements will return NaN here. 
        
        main_batch = {
            'audio': torch.stack([sample['AudioFeatures'] for sample in samples]),
            'transcript': torch.stack([sample['TextFeatures'] for sample in samples]),
            'act': torch.stack([sample['act'] for sample in samples]),
            'val': torch.stack([sample['val'] for sample in samples]),
            'kde_2d_probability': torch.stack([sample['kde_2d_probability'] for sample in samples]),
            'act_variance': torch.stack([safe_var(sample['soft_act_labels']) for sample in samples]),
            'val_variance': torch.stack([safe_var(sample['soft_val_labels']) for sample in samples]),
            'soft_act_labels': [sample['soft_act_labels'] for sample in samples],
            'soft_val_labels': [sample['soft_val_labels'] for sample in samples],
            'annotators': [sample['annotators'] for sample in samples],
            'FileName': [sample['FileName'] for sample in samples],
        }

        if 'target_annotator' in samples[0]:
            assert all(sample['target_annotator'] == samples[0]['target_annotator'] for sample in samples), "Target annotator must be the same for all samples in a batch"
            main_batch['target_annotator'] = samples[0]['target_annotator']
        if self.pad_soft_labels:
            main_batch['padded_individual_annotators_act'] = torch.nn.utils.rnn.pad_sequence(main_batch['soft_act_labels'], batch_first=True, padding_value=torch.nan)
            main_batch['padded_individual_annotators_val'] = torch.nn.utils.rnn.pad_sequence(main_batch['soft_val_labels'], batch_first=True, padding_value=torch.nan)
        if hasattr(self, 'annotator_collator') and self.batch_annotators:
            annotator_batch = self.annotator_collator(samples)
            main_batch = {**main_batch, **annotator_batch}
            # print('has annotator mapping', annotator_batch.keys())
        return main_batch