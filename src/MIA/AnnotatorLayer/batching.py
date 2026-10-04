import torch

class AnnotatorBatchCollator:
    def __init__(self, annotator_mapper=None, include_aggregate_annotator=False, include_self_report=False, include_unseen_annotators=True):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.initialised = False
        self.annotator_mapper = annotator_mapper
        self.include_agg = include_aggregate_annotator
        self.include_self = include_self_report
        self.include_unseen_annotators = include_unseen_annotators
        if self.include_unseen_annotators:
            self.set_of_annotators = self.annotator_mapper.get_annotators()
        if include_aggregate_annotator or include_self_report:
            self.annotator_mapper.add_special_annotators(include_aggregate_annotator, include_self_report)
        self.batch_annotators = True

    def __call__(self, samples):
        batched_samples = {}

        # In this case we want to batch these with NaN values the same as the model will in training and generate a list of masks for use in model
        # This should ensure that individual annotator models from output line up with individual annotator target labels
        soft_act_labels = []
        soft_val_labels = []
        annotator_mask = []
        batch_mask = []
        seen_annotator_mask = []

        # Set annotator_mapper to map from annotator id (string) -> annotator index (int)
        self.annotator_mapper.set_get_idx()

        for i, sample in enumerate(samples):
            annotators = sample['annotators']
            soft_act = sample['soft_act_labels']
            soft_val = sample['soft_val_labels']
            if self.include_agg:
                soft_act = torch.cat((soft_act, sample['act'].view(-1)), dim=0)
                soft_val = torch.cat((soft_val, sample['val'].view(-1)), dim=0)
                annotators.append('aggregate-annotator')
            if self.include_self:
                soft_act = torch.cat((soft_act, sample['self-report-act']), dim=0)
                soft_val = torch.cat((soft_act, sample['self-report-val']), dim=0)
                annotators.append('self-report')

            # Get indexes for annotators
            if self.include_unseen_annotators:
                # Default unseen annotator behaviour is not to predict it 
                # Since the annotator indexes are used to build masks for prediction we can just remove these
                batch_seen_mask = [a in self.set_of_annotators for a in annotators]
                seen_annotators = [a for a in annotators if a in self.set_of_annotators]
                annotators_to_predict = self.annotator_mapper[seen_annotators] if len(seen_annotators) else []
            else:
                annotators_to_predict = self.annotator_mapper[annotators]
                batch_seen_mask = [True]*len(annotators) # All annotators must be seen 

            annotator_mask.extend(annotators_to_predict)
            seen_annotator_mask.append(torch.as_tensor(batch_seen_mask).bool())

            # Generate batch mask (for duplicating batch input for each annotator to be used)
            batch_mask.append(torch.as_tensor([i]*len(annotators_to_predict)).long())

            # Create corresponding soft-labels for training
            soft_act_labels.append(soft_act)
            soft_val_labels.append(soft_val)

        # There are variable number of annotators in each sample, so pad these using NaN for values with no annotator
        seen_annotator_mask = torch.nn.utils.rnn.pad_sequence(seen_annotator_mask, batch_first=True, padding_value=False)

        # Now create a mask for filling the output, if there is a corresponding annotator, then there should be an output (true) if NaN then we don't use this for output mask
        output_mask = seen_annotator_mask

        # This will now give us a list of all non-nan annotator indexs
        annotator_mask = torch.as_tensor(annotator_mask, dtype=torch.long, device=self.device)

        # Now pad and concatenate remaining values
        batch_mask = torch.cat(batch_mask)
        soft_act_labels = torch.nn.utils.rnn.pad_sequence(soft_act_labels, batch_first=True, padding_value=torch.nan)
        soft_val_labels = torch.nn.utils.rnn.pad_sequence(soft_val_labels, batch_first=True, padding_value=torch.nan)
        # Warning -- masks will not correspond to padded_individual_annotators_act in the event that unseen annotators were included
        batched_samples['annotator_masks'] = (batch_mask, annotator_mask, output_mask, seen_annotator_mask)
        batched_samples['padded_individual_annotators_act'] = soft_act_labels
        batched_samples['padded_individual_annotators_val'] = soft_val_labels

        # Return batch
        return batched_samples