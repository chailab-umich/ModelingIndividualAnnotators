import os
import pathlib
from .utils import save_torch_model

def mkdirp(dir_path):
    if not os.path.isdir(dir_path):
        pathlib.Path(dir_path).mkdir(parents=True)

class EarlyStopping:
    def __init__(self, model_save_loc, patience=10, mode='min'):
        mkdirp(model_save_loc)
        self.model_save_loc = model_save_loc
        self.patience = patience
        self.mode = mode
        self.previous_results = []

    def reset(self):
        mkdirp(self.model_save_loc)
        self.previous_results = []

    def compare(self, value):
        if len(self.previous_results) == 0:
            return True # Always the best seen result if there are no previously seen results 
        if self.mode == 'min':
            return value < min(self.previous_results)
        elif self.mode == 'max':
            return value > max(self.previous_results)
        else:
            raise ValueError(f'Unknown early stopping mode: {self.mode}')

    def get_best_idx(self):
        comp_fn = None
        if self.mode == 'min':
            comp_fn = min
        elif self.mode == 'max':
            comp_fn = max
        else:
            raise ValueError(f'Unknown early stopping mode: {self.mode}')
        
        return self.previous_results.index(comp_fn(self.previous_results))

    # Returns false if the model should stop training 
    def log_value(self, value, model):
        if self.compare(value):
            # Value is best we have seen so far! Save model and log 
            save_location = os.path.join(self.model_save_loc, f'{len(self.previous_results)}_cp.ckpt')
            print(f'Best seen early stopping value, saving model to {save_location}')
            save_torch_model(model, save_location)
            if len(self.previous_results): # If there are no values in the previous results then this is undefined
                old_location = self.get_best_model_path()
                if os.path.exists(old_location):
                    print(f'Removing old best model checkpoint from {old_location}')
                    os.remove(old_location)

        self.previous_results.append(value)

        # Check if we should stop training 
        best_seen_it_num = self.get_best_idx()
        current_it_num = len(self.previous_results) - 1 # Since the indexing is from 0 this needs minus 1. E.g. patience 1 would always stop immediately otherwise
        return current_it_num - best_seen_it_num < self.patience # Returns true if we should stop early

    def get_best_model_path(self):
        idx = self.get_best_idx()
        return os.path.join(self.model_save_loc, f'{idx}_cp.ckpt')