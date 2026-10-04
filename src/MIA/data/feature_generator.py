import os
import time
import torch
import math
import shutil
import pathlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from threading import Thread
from multiprocessing import Process, SimpleQueue as Queue
from transformers import AutoModel, AutoFeatureExtractor, AutoTokenizer
from datasets import Dataset, concatenate_datasets

class AudioReader(Thread):
    # Pass full dataset in as this will run on exactly 1 thread
    def __init__(self, dataset, chunk_queue, update_queue, CHUNK_SIZE, chunk_id):
        self.dataset = dataset
        self.chunk_queue = chunk_queue
        self.update_queue = update_queue
        self.CHUNK_SIZE = CHUNK_SIZE
        self.chunk_id = chunk_id
        super().__init__()

    def run(self):
        curr_chunk = []
        chunk_idx = 0
        for i in range(len(self.dataset)):
            curr_chunk.append((self.dataset[i]['Audio'], self.dataset[i]['FileName'], self.dataset[i]['Text']))
            self.update_queue.put((1,0,0))
            if len(curr_chunk) >= self.CHUNK_SIZE:
                self.chunk_queue.put((self.chunk_id, chunk_idx, curr_chunk))
                curr_chunk = []
                chunk_idx += 1
        if len(curr_chunk):
            self.chunk_queue.put((self.chunk_id, chunk_idx, curr_chunk))

class ChunkWriter(Thread):
    def __init__(self, chunk_ds_queue, update_queue, temp_path):
        self.chunk_ds_queue = chunk_ds_queue
        self.update_queue = update_queue
        self.temp_path = temp_path
        super().__init__()

    def run(self):
        for (chunk_id, chunk_ds) in iter(self.chunk_ds_queue.get, 'shutdown worker'):
            Dataset.from_dict(chunk_ds).save_to_disk(os.path.join(self.temp_path, chunk_id))
            self.update_queue.put((0,0,1))

class ChunkProcessor(Process):
    def __init__(self, chunk_queue, chunk_ds_queue, update_queue, ds_df, gpu, audio_features, text_features, audio_feature_layer):
        self.chunk_queue = chunk_queue
        self.chunk_ds_queue = chunk_ds_queue
        self.update_queue = update_queue
        self.ds_df = ds_df
        self.device = torch.device(f'cuda:{gpu}')
        self.audio_features = audio_features
        self.text_features = text_features
        self.audio_feature_layer = audio_feature_layer
        self.initialised = False
        super().__init__()

    def initialise(self):
        from transformers.utils import logging # Silence warnings as model loads result in repeated warnings per process
        logging.set_verbosity_error() 
        if self.audio_features != 'raw' and self.audio_features != 'mfb' and type(self.audio_features) == str:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.audio_features)
            self.transformer_model = AutoModel.from_pretrained(self.audio_features)
            if torch.cuda.is_available():
                self.transformer_model = self.transformer_model.to(self.device)
            self.transformer_model.eval()

        if self.text_features != 'raw' and type(self.text_features) == str:
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_features)
            self.bert_model = AutoModel.from_pretrained(self.text_features)
            if torch.cuda.is_available():
                self.bert_model = self.bert_model.to(self.device)
        self.initialised = True

    def run(self):
        if not self.initialised:
            self.initialise()
        with torch.no_grad():
            for (chunk_id_prefix, chunk_idx, chunk) in iter(self.chunk_queue.get, 'shutdown worker'):
                chunk_id = f'{chunk_id_prefix}_temp_chunk_{chunk_idx}'
                chunk_results = {'FileName': [], 'AudioFeatures': [], 'TextFeatures': []}
                for (audio, key, text) in chunk:
                    if audio['sampling_rate'] != 16000:
                        raise ValueError('Sampling rate should be 16000')
                    if len(audio['array']):
                        audio_features = self.feature_extractor(audio['array'], sampling_rate=16000, return_tensors='pt')
                        if torch.cuda.is_available():
                            audio_features = audio_features.to(self.device)
                        # assert len(audio_features) == 1 
                        if self.audio_feature_layer == 'last_hidden_state':
                            audio_features = self.transformer_model(**audio_features)['last_hidden_state']
                            audio_features = torch.mean(torch.as_tensor(audio_features), dim=1)
                        elif self.audio_feature_layer == 'last_three_layers':
                            # print('output shape:', len(self.transformer_model(**audio_features, output_hidden_states=True)['hidden_states']))
                            audio_features = self.transformer_model(**audio_features, output_hidden_states=True)['hidden_states'][-3]

                        audio_features = audio_features.squeeze(dim=0).cpu().numpy()
                    else:
                        print('ERROR: Empty wav file -- returning empty tensor for w2v2 features')
                        audio_features = torch.tensor([]).squeeze()

                    bert_tokens = self.tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')
                    # if torch.cuda.is_available():
                        # bert_tokens = bert_tokens.to('cuda')
                    bert_tokens = bert_tokens.to(self.bert_model.device)
                    out = self.bert_model(**bert_tokens).last_hidden_state
                    cls_tok = out.squeeze()[0].cpu().numpy()
                    
                    # Store results in chunk
                    chunk_results['FileName'].append(key)
                    chunk_results['AudioFeatures'].append(audio_features)
                    chunk_results['TextFeatures'].append(cls_tok)
                chunk_results = pd.DataFrame(chunk_results)
                chunk_results = self.ds_df.merge(chunk_results, on='FileName')
                self.chunk_ds_queue.put((chunk_id, chunk_results))
                self.update_queue.put((0,1,0))

def generate_features(dataset, dataset_name, audio_features, text_features, audio_feature_layer, temp_path, chunk_id_type):
    dataset.set_format(type='torch', columns=['Audio'], output_all_columns=True)
    CHUNK_SIZE = 1000
    NUM_PROCESSES_PER_GPU = 1
    CUDA_GPUS = [0,1,2]
    
    # audio_and_keys = [(dataset[i]['Audio'], dataset[i]['FileName'], dataset[i]['Text']) for i in tqdm(range(len(dataset)), desc='Reading Audio Files')]
    num_chunks = math.ceil(len(dataset)/CHUNK_SIZE)
    chunk_queue = Queue() # Where chunks will be placed in order to be processed 
    chunk_ds_queue = Queue() # Where processed chunks will be placed
    update_queue = Queue() # Queue for updating the progress bars
    read_pbar = tqdm([], total=len(dataset), desc='Reading wavs')
    process_pbar = tqdm([], total=num_chunks, desc='Enhancing wavs')
    write_pbar = tqdm([], total=num_chunks, desc='Reading wavs')

    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)

    pathlib.Path(temp_path).mkdir(parents=True, exist_ok=True)

    # Create processes and threads
    chunk_id_prefix = f'{dataset_name}_{chunk_id_type}'
    read_thread = AudioReader(dataset, chunk_queue, update_queue, CHUNK_SIZE, chunk_id_prefix)
    read_thread.start()

    ds_df = dataset.to_pandas()
    processes = [ChunkProcessor(chunk_queue, chunk_ds_queue, update_queue, ds_df, gpu, audio_features, text_features, audio_feature_layer) for gpu in CUDA_GPUS for _ in range(NUM_PROCESSES_PER_GPU)]

    for p in processes:
        p.start()

    write_threads = [ChunkWriter(chunk_ds_queue, update_queue, temp_path)]
    for t in write_threads:
        t.start()

    # for chunk_idx in read_pbar:
    #     start = chunk_idx*CHUNK_SIZE
    #     end = min((chunk_idx+1)*CHUNK_SIZE, len(dataset))
    #     audio_and_keys = [(dataset[i]['Audio'], dataset[i]['FileName'], dataset[i]['Text']) for i in range(start, end)]
    #     chunk_queue.put((dataset_name, chunk_idx, audio_and_keys))

    # finished_reading = False
    finished_processing = False
    finished_writing = False
    while not (not read_thread.is_alive() and finished_processing and finished_writing):
        time.sleep(1)
        new_reads, new_processes, new_writes = 0, 0, 0
        while not update_queue.empty():
            update = update_queue.get()
            countr, countp, countw = update
            new_reads += countr
            new_processes += countp
            new_writes += countw
        if read_thread.is_alive():
            read_pbar.update(new_reads)
            read_pbar.set_description(f'Wavs read')
            read_pbar.refresh()
        else:
            for _ in range(len(processes)):
                chunk_queue.put('shutdown worker')
        if not finished_processing:
            process_pbar.update(new_processes)
            process_pbar.set_description(f'Chunks Processed (alive threads: {[t.is_alive() for t in processes]})')
            process_pbar.refresh()
        if not finished_writing:
            write_pbar.update(new_writes)
            write_pbar.set_description(f'Chunks Written (alive threads: {[t.is_alive() for t in write_threads]})')
            write_pbar.refresh()
        if not finished_processing and (not np.array([t.is_alive() for t in processes]).any()):
            for _ in range(len(write_threads)):
                chunk_ds_queue.put('shutdown worker')
            finished_processing = True
        if not finished_writing and (not np.array([t.is_alive() for t in write_threads]).any()):
            finished_writing = True

    # Load full dataset and concatenate into one dataset
    all = [Dataset.load_from_disk(os.path.join(temp_path, f'{chunk_id_prefix}_temp_chunk_{chunk_idx}')) for chunk_idx in range(num_chunks)]
    full = concatenate_datasets(all)
    return full
