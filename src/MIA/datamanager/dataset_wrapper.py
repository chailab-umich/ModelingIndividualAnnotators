from torch.utils.data import Dataset

class DatasetWrapper(Dataset):
    def __init__(self, dataset):
        print('Created DatasetWrapper for dataset', dataset)
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return_dict = {}
        if isinstance(idx, tuple):
            idx, target_annotator = idx
            return_dict['target_annotator'] = target_annotator

        # Explicitly access each key from the row - HuggingFace Row/LazyDict applies
        # format (including device placement) only on __getitem__, not when unpacking
        # with **. Using {**row} can bypass format for some column types.
        row = self.dataset[idx]
        for key in row.keys():
            return_dict[key] = row[key]
        return return_dict