import re
# from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from datetime import datetime
from MIA.utils import Singleton
import pandas as pd 
import pathlib
import os
import torch

# Convert logger to use pandas instead of tensorboard 
class CSVWriter:
    def __init__(self, csv_results_path='csv_results', experiment_name=None):
        print('creating CSV writer with results directory', csv_results_path)
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
        
        # Create nested directory structure: csv_results_path/experiment_name_timestamp/
        if experiment_name:
            self.base_csv_dir = f'{csv_results_path}/{experiment_name}_{dt_string}'
        else:
            # Fallback to old behavior for backward compatibility
            self.base_csv_dir = f'{csv_results_path}_{dt_string}'
            
        self.csv_data = {} # Store dataframes for each model
        self.columns = ['log_type', 'group', 'model_name', 'scalar_value', 'step']
        if not os.path.exists(self.base_csv_dir):
            os.makedirs(self.base_csv_dir)

    # log_type defines the test (e.g. test zero shot, test all annotators, etc.)
    # group is the name of the metric
    # model_name is the name of the model being tested 
    # scalar_value is the value of the metric
    # step is just the epoch number usually, for testing I think this is always 0
    def log_scalar(self, log_type, group, model_name, scalar_value, step):
        # If we have been asked to plot a scalar that addresses individual annotators, we want to group this by model rather than by metric
        # Brackets mess up the custom scalar logging. Remove all here.
        # if 'train' in log_type.lower() and step % 100 and 'batch_loss' not in group.lower():
            # return # Only log every 100 steps
        if 'test' not in log_type.lower() and 'mapping information' not in log_type.lower():
            return # Only log test and mapping information now we are storing as a csv 

        log_type = log_type.replace('(', '').replace(')', '')
        group = group.replace('(', '').replace(')', '')

        if model_name not in self.csv_data:
            print('CREATING CSV data for', model_name)
            self.csv_data[model_name] = pd.DataFrame(columns=self.columns)
        if torch.is_tensor(scalar_value):
            scalar_value = scalar_value.item()

        self.csv_data[model_name] = pd.concat([pd.DataFrame([[log_type, group, model_name, scalar_value, step]], columns=self.columns), self.csv_data[model_name]], ignore_index=True)

    def write_csvs(self):
        print(f'Attempting to write CSVs to directory: {self.base_csv_dir}')
        
        # Ensure directory exists
        try:
            pathlib.Path(self.base_csv_dir).mkdir(parents=True, exist_ok=True)
            # print(f'Successfully created/verified directory: {self.base_csv_dir}')
        except Exception as e:
            raise IOError(f'Error creating directory {self.base_csv_dir}: {e}')
            
        print('SAVING CSVs', self.csv_data.keys())
        
        if not self.csv_data:
            print('No CSV data to save - no data was logged')
            return

        for name in self.csv_data:
            csv_path = f'{self.base_csv_dir}/{name}.csv'
            # print(f'Writing CSV for {name} to {csv_path}')
            try:
                # print(f'CSV data for {name} has {len(self.csv_data[name])} rows')
                self.csv_data[name].to_csv(csv_path, index=False)
                # print(f'Successfully wrote {csv_path}')
            except Exception as e:
                print(f'Error writing CSV {csv_path}: {e}')