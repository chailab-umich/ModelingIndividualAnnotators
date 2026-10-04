from torch.utils.data import Sampler
import numpy as np

class BatchSamplerByAnnotator(Sampler):
    def __init__(self, annotator_to_samples, shuffle: bool = True, batch_size: int = 32):
        self.annotator_to_samples = annotator_to_samples
        self.shuffle = shuffle
        self.annotator_order = list(annotator_to_samples.keys())
        self.batch_size = batch_size

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.annotator_order)
        for annotator in self.annotator_order:
            # Choose 32 random samples from this annotator and yield them 
            sample_idxs = np.random.choice(self.annotator_to_samples[annotator], size=self.batch_size, replace=False)
            yield [(idx.item(), annotator) for idx in sample_idxs]

    def __len__(self):
        return len(self.annotator_order)