import os
import torch
import pandas as pd
import numpy as np
import torch
import yaml as pyyaml
import pathlib
import re
import json
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoFeatureExtractor, AutoProcessor
from datasets import load_dataset, concatenate_datasets, Dataset, Audio
from .podcast import read_podcast
from .improv import read_improv
from .iemocap import read_iemocap
from .muse import read_muse
from .config import Config
from .kde_probability import kde_probability_bs
from .utils import scale_dataset, prune_annotators_fn, add_muse_annotators_NO_SELF_REPORT, add_muse_annotators
from .feature_generator import generate_features

conf = Config()

# Define directories and paths for use in loading
dataset_paths = {
    'podcast': conf['podcast_directory'],
    'improv': conf['improv_directory'],
    'muse': conf['muse_directory'],
    'iemocap': conf['iemocap_directory'],
}
label_paths = {
    'podcast': os.path.join(conf['podcast_directory'], 'Labels', 'labels_consensus.csv'),
    'improv': os.path.join(conf['improv_directory'], 'Evaluation.txt'),
    'muse': os.path.join(conf['muse_directory'], 'SurveyInformation', 'Emotion data from crowdsourcing (C is when annotators had access to all previous sentences).csv'),
    'iemocap': os.path.join(conf['iemocap_directory'], 'IEMOCAP_EmoEvaluation.txt'),
}
# Define the min and maximum values in the data prior to scaling
dataset_scale_parameters = {
    'podcast': (1, 7),
    'improv': (1, 5),
    'iemocap': (1, 5),
    'muse': (1, 9),
}

# MuSE not named consistently so check if path correct
if not os.path.exists(label_paths['muse']):
    label_paths['muse'] = os.path.join(conf['muse_directory'], 'Survey Information (Questions, Data etc)', 'Emotion data from crowdsourcing (C is when annotators had access to all previous sentences).csv')

file_reader = {
    'podcast': read_podcast,
    'improv': read_improv,
    'muse': read_muse,
    'iemocap': read_iemocap,
}

# Method for calculating length from the audio column in huggingface dataset
def add_length(sample):
    sample['AudioLength'] = sample['Audio']['array'].shape[0]/sample['Audio']['sampling_rate']
    return sample

def format_datasets(type_to_columns, column_masks, *datasets):
    # Only the torch type needs to be formatted, the None types are still left as none
    # only one formatting can be applied so it has to be left implicit for the None types
    column_type = 'torch'
    columns_to_set, kwargs = type_to_columns[column_type]
    columns_to_set = columns_to_set[column_masks[column_type]]
    for dataset in datasets:
        # Dataset may not have all available columns -- remove them at this point
        ds_cols_to_set = [col for col in columns_to_set if col in dataset.features.keys()]
        if len(ds_cols_to_set):
            dataset.set_format(column_type, columns=ds_cols_to_set, output_all_columns=True, **kwargs)

def make_audio_datasets(
    datasets_to_load=['improv', 'iemocap', 'muse', 'podcast'],
    kde_size=4,
    add_enhanced_wavs=False,
    podcast_version='1.12',
    prune_annotators=False,
    cross_validation_folds=False,
    legacy_asru_splits=False,
    legacy_journal_splits=False,
): # legacy ASRU splits (see readme in ./ASRULegacySplits)
    """
    Creates audio datasets for training, development, and testing from labeled audio files.

    Code requires setting multiprocessing method to spawn prior to call to create features correctly
    from multiprocess import set_start_method
    set_start_method('spawn')
    
    Args:
        audio_dir (str): The directory containing the audio files.
        labels_path (str): The path to the CSV file containing the labels and file information.
        cached_layer (int, optional): If provided then use the cached w2v2 outputs from this layer. Defaults to None, which does not use cached features.

    Returns:
        train_dataset: A huggingface Dataset object containing the training data.
        dev_dataset: A huggingface Dataset object containing the development data.
        test_dataset: A huggingface Dataset object containing the testing data.
    """
    cache_read_only = os.environ.get('MIA_DATASET_CACHE_READ_ONLY', '').lower() in {
        '1', 'true', 'yes', 'on'
    }
    trusted_cache_datasets = {
        item.strip()
        for item in os.environ.get(
            'MIA_DATASET_CACHE_TRUSTED_DATASETS', ''
        ).split(',')
        if item.strip()
    }

    # Update file_reader to function for requested podcast version
    file_reader['podcast'] = lambda *args, **kwargs: read_podcast(*args, **kwargs, podcast_v=podcast_version)
    # First load config and calculate which columns will be used based on the config file 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    type_to_columns = { # None returns just plain python objects -- use for dictionaries and strings
        None: (np.array(['FileName', 'Split_Set', 'annotators']), {}),
        'torch': (np.array(['Audio', 'Text', 'AudioFeatures', 'TextFeatures', 'AudioEnhanced', 'act', 'val', 'soft_act_labels', 'soft_val_labels', 'self-report-act', 'self-report-val']), {'dtype': torch.float32, 'device': device})
    }
    column_masks = {
        None: [True, True, conf['return_annotator_info']],
        'torch': [True, True, True, True, add_enhanced_wavs, conf['return_activation'], conf['return_valence'], conf['return_activation'] and conf['return_soft_labels'], conf['return_valence'] and conf['return_soft_labels'], conf['return_activation'] and conf['return_self_report'], conf['return_valence'] and conf['return_self_report']]
    }
    # Apply masks along configs to keep only columns where mask is True
    columns = [x for coltype in type_to_columns for x in type_to_columns[coltype][0][column_masks[coltype]]]
    if 'Dataset' not in columns:
        columns.append('Dataset')
    print('Using columns:', columns)
    
    train_datasets, dev_datasets, test_datasets = {}, {}, {}
    loaded_train_datasets, loaded_dev_datasets, loaded_test_datasets = {}, {}, {}
    was_change_to_cache = {}
    loaded_keys = []

    if trusted_cache_datasets and not conf['cache_datasets']:
        raise IOError('Trusted cache datasets require cache_datasets to be enabled')

    # Load datasets from cache
    if conf['cache_datasets']:
        # Ensure path exists
        cache_path = pathlib.Path(conf['cache_dataset_path'])
        if cache_read_only and not cache_path.is_dir():
            raise IOError(f'Read-only dataset cache does not exist: {cache_path}')
        if not cache_read_only:
            cache_path.mkdir(parents=True, exist_ok=True)
        config_path = os.path.join(conf['cache_dataset_path'], 'config_used_for_cache.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as config_file:
                old_config = pyyaml.safe_load(config_file)
        else:
            if cache_read_only:
                raise IOError(f'Read-only dataset cache has no metadata file: {config_path}')
            old_config = conf
            with open(config_path, 'w') as config_file:
                config_file.write(pyyaml.dump(conf))

        # Model-identification runs validate the feature-generating fields and
        # parquet schemas before importing this module.  Their historical
        # caches can legitimately differ in lightweight label/path settings.
        # Non-trusted datasets reconstruct those below; trusted historical
        # datasets preserve the cached values.  Neither may be overwritten.
        config_changed = False if cache_read_only else conf != old_config

        for key in datasets_to_load:
            podcast_v_str = f'v{podcast_version}' if podcast_version != '1.12' and key == 'podcast' else ''
            load_paths = {
                'train': os.path.join(conf['cache_dataset_path'], f'{key}{podcast_v_str}_train.parquet'),
                'dev': os.path.join(conf['cache_dataset_path'], f'{key}{podcast_v_str}_dev.parquet'),
                'test': os.path.join(conf['cache_dataset_path'], f'{key}{podcast_v_str}_test.parquet'),
            }
            train_exists = os.path.exists(load_paths['train'])
            dev_exists = os.path.exists(load_paths['dev'])
            test_exists = os.path.exists(load_paths['test'])
            all_exist = all([train_exists, dev_exists, test_exists])
            if key == 'podcast' and podcast_version != '1.12' and not all_exist:
                # Since the requested version of podcast doesnt exist, check if 1.12 exists as this can be used to prevent recalculating the majority of values
                load_paths = {
                    'train': os.path.join(conf['cache_dataset_path'], f'{key}_train.parquet'),
                    'dev': os.path.join(conf['cache_dataset_path'], f'{key}_dev.parquet'),
                    'test': os.path.join(conf['cache_dataset_path'], f'{key}_test.parquet'),
                }
                train_exists = os.path.exists(load_paths['train'])
                dev_exists = os.path.exists(load_paths['dev'])
                test_exists = os.path.exists(load_paths['test'])
                all_exist = all([train_exists, dev_exists, test_exists])

            if all_exist:
                # hfdataset = load_dataset('parquet', data_files=load_paths)
                loaded_train_datasets[key] = load_dataset('parquet', data_files={'train': load_paths['train']})['train']
                loaded_dev_datasets[key] = load_dataset('parquet', data_files={'dev': load_paths['dev']})['dev']
                loaded_test_datasets[key] = load_dataset('parquet', data_files={'test': load_paths['test']})['test']
                # print(f'Loaded from cache train:{len(loaded_train_datasets[key])} dev:{len(loaded_dev_datasets[key])} test:{len(loaded_test_datasets[key])}')
                was_change_to_cache[key] = False
                loaded_keys.append(key)
            if (not all_exist and any([train_exists, dev_exists, test_exists])) or config_changed:
                if config_changed:
                    print('Config changed:')
                    for key in conf:
                        old_value = old_config.get(key, '<missing>')
                        if conf[key] != old_value:
                            print(f'Expected {key} to be {old_value} from cached config but got {conf[key]} in current run settings')
                raise IOError(f'Found partial files of dataset or cached dataset not matching current config settings. Either remove all files to cause recalculation, or correct paths of missing files. Checked paths: {load_paths}')

        missing_trusted = trusted_cache_datasets.intersection(datasets_to_load) - set(loaded_keys)
        if missing_trusted:
            raise IOError(
                'Trusted datasets were not completely loaded from cache: '
                + ', '.join(sorted(missing_trusted))
            )

    # Remove any datasets that were loaded from cache
    # Rather than caching entire dataset, we instead should verify that all labels have the correct generated features
    # This allows lightweight changes i.e. to file_reader labels returned (for example maybe adding new speaker labels that were not originally present)
    # without regenerating the transformer features that are very slow to generate
    some_datasets_loaded = len(loaded_keys) > 0
    for key in datasets_to_load:
        if key in trusted_cache_datasets:
            # A validated historical cache is the source of truth for this
            # dataset.  In particular, old MSP-Podcast releases may no longer
            # have every raw audio/transcript needed to reconstruct labels.
            train_datasets[key] = pd.DataFrame({'FileName': []})
            dev_datasets[key] = pd.DataFrame({'FileName': []})
            test_datasets[key] = pd.DataFrame({'FileName': []})
            continue
        # Load labels for each dataset
        train_datasets[key], dev_datasets[key], test_datasets[key] = file_reader[key](dataset_paths[key], label_paths[key], columns=columns)
        if key == 'muse':
            if conf['return_self_report']:
                train_datasets[key], dev_datasets[key], test_datasets[key] = add_muse_annotators(
                    train_datasets[key],
                    dev_datasets[key],
                    test_datasets[key],
                    conf.get('muse_annotator_info_path'),
                )
            else:
                train_datasets[key], dev_datasets[key], test_datasets[key] = add_muse_annotators_NO_SELF_REPORT(
                    train_datasets[key],
                    dev_datasets[key],
                    test_datasets[key],
                    conf.get('muse_annotator_info_path'),
                )
        # print(f'Read train:{len(train_datasets[key])} dev:{len(dev_datasets[key])} test:{len(test_datasets[key])}')
        min_v, max_v = dataset_scale_parameters[key]

        train_datasets[key] = scale_dataset(train_datasets[key], min_v, max_v)
        dev_datasets[key] = scale_dataset(dev_datasets[key], min_v, max_v)
        test_datasets[key] = scale_dataset(test_datasets[key], min_v, max_v)

    # Now remove any extra samples in the loaded dataset
    paired_iterator = [(loaded_train_datasets, train_datasets), (loaded_dev_datasets, dev_datasets), (loaded_test_datasets, test_datasets)]
    if some_datasets_loaded:
        # Since all datasets were loaded we should quickly check if feature generation is required at all by checking if the datasets loaded from cache all had a feature
        for key in loaded_keys:
            if key in trusted_cache_datasets:
                continue
            extra_count = 0
            for loaded_data, new_data in paired_iterator:
                loaded_filenames = set(loaded_data[key]['FileName'])
                new_filenames = set(new_data[key]['FileName'])
                extra_data = loaded_filenames - new_filenames
                extra_count += len(extra_data)
                loaded_data[key] = loaded_data[key].select([i for i, fname in enumerate(loaded_data[key]['FileName']) if fname not in extra_data])

            print(f'{extra_count} Extra files removed for {key}')


    # Now overwrite the lightweight labels in the loaded datasets if they were loaded
    for key in loaded_keys:
        if key in trusted_cache_datasets:
            continue
        for loaded_data, new_data in paired_iterator:
            order = loaded_data[key]['FileName']
            ordered_train = new_data[key].set_index('FileName').loc[order].reset_index()
            print(ordered_train.columns)
            for column in ordered_train.columns:
                if column in ['FileName', 'Audio', 'Text']:
                    continue
                old_column = None
                if column in loaded_data[key].column_names:
                    old_column = loaded_data[key][column]
                    loaded_data[key] = loaded_data[key].remove_columns(column)
                loaded_data[key] = loaded_data[key].add_column(column, ordered_train[column])
                if old_column != loaded_data[key][column]:
                    was_change_to_cache[key] = True
                    print('Data changed for', key, column)

    # Check that features need to be generated for audio and text 
    audio_features, text_features = conf['audio_feature_type'], conf['text_feature_type']
    if some_datasets_loaded:
        # Since all datasets were loaded we should quickly check if feature generation is required at all by checking if the datasets loaded from cache all had a feature
        for key in loaded_keys:
            missing_count = 0
            for loaded_data, new_data in paired_iterator:
                loaded_filenames = set(loaded_data[key]['FileName'])
                new_filenames = set(new_data[key]['FileName'])
                missing_data = new_filenames - loaded_filenames
                missing_count += len(missing_data)
                new_data[key] = new_data[key][new_data[key]['FileName'].isin(missing_data)]

            print(f'{missing_count} Files need audio features generating for {key}')

    # Now, for any keys that were *not* loaded convert to huggingface datasets
    # Datasets that were loaded will already be huggingface datasets that have had the lightweight labels overwritten
    datasets_to_generate_features = []
    for key in datasets_to_load:
        if len(train_datasets[key]) or len(dev_datasets[key]) or len(test_datasets[key]):
            print('converting to hf dataset', key)
            train_datasets[key] = Dataset.from_pandas(train_datasets[key]).cast_column('Audio', Audio(sampling_rate=16000, mono=True))
            dev_datasets[key] = Dataset.from_pandas(dev_datasets[key]).cast_column('Audio', Audio(sampling_rate=16000, mono=True))
            test_datasets[key] = Dataset.from_pandas(test_datasets[key]).cast_column('Audio', Audio(sampling_rate=16000, mono=True))
            datasets_to_generate_features.append(key)
            was_change_to_cache[key] = True
        else:
            # Remove any datasets that don't need any new features as the loaded dataset is sufficient
            del train_datasets[key], dev_datasets[key], test_datasets[key]

    feature_generation = len(datasets_to_generate_features) and (audio_features != 'raw' or text_features != 'raw')

    # Now create features 
    for key in datasets_to_generate_features:
        if prune_annotators:
            print('Pruning annotators before KDE calculation to ensure consistent labels were used in the KDE generation')
            print('later pruning should therefore return that no annotators were removed or changed')
            train_datasets[key], dev_datasets[key], test_datasets[key] = prune_annotators_fn(train_datasets[key], dev_datasets[key], test_datasets[key], min_n=1)

        if len(train_datasets[key]) or len(dev_datasets[key]) or len(test_datasets[key]): # If an item was fully removed previously then this will crash, so check that there is any data not pruned
            # Calculate audio and text features 
            if feature_generation:
                audio_feature_layer = conf['audio_feature_layer']
                base_path = conf['cache_dataset_path']
                temp_path = os.path.join(base_path, 'temp')
                if len(train_datasets[key]):
                    train_datasets[key] = generate_features(train_datasets[key], key, audio_features, text_features, audio_feature_layer, temp_path, 'train')
                if len(dev_datasets[key]):
                    dev_datasets[key] = generate_features(dev_datasets[key], key, audio_features, text_features, audio_feature_layer, temp_path, 'dev')
                if len(test_datasets[key]):
                    test_datasets[key] = generate_features(test_datasets[key], key, audio_features, text_features, audio_feature_layer, temp_path, 'test')

            # Set the correct format on the dataset -- has to be done prior to calculation of KDE labels
            format_datasets(type_to_columns, column_masks, train_datasets[key], dev_datasets[key], test_datasets[key])

            # Calculate KDE 2D labels
            if conf['calculate_kde']:
                allow_failure = key == 'podcast' and podcast_version == '1.8'
                train_datasets[key] = train_datasets[key].map(lambda x: create_kde_labels_map(x, kde_size=kde_size, num_calculations=conf['num_kde_calculations'], allow_failure=False), batched=True, batch_size=256)
                dev_datasets[key] = dev_datasets[key].map(lambda x: create_kde_labels_map(x, kde_size=kde_size, num_calculations=conf['num_kde_calculations'], allow_failure=allow_failure), batched=True, batch_size=256)
                test_datasets[key] = test_datasets[key].map(lambda x: create_kde_labels_map(x, kde_size=kde_size, num_calculations=conf['num_kde_calculations'], allow_failure=allow_failure), batched=True, batch_size=256)

            # If datasets were partially loaded previously then now concatenate the new values to the old values
            if key in loaded_keys:
                train_datasets[key] = concatenate_datasets([loaded_train_datasets[key], train_datasets[key]])
                dev_datasets[key] = concatenate_datasets([loaded_dev_datasets[key], dev_datasets[key]])
                test_datasets[key] = concatenate_datasets([loaded_test_datasets[key], test_datasets[key]])
        else:
            train_datasets[key] = loaded_train_datasets[key]
            dev_datasets[key] = loaded_dev_datasets[key]
            test_datasets[key] = loaded_test_datasets[key]

    # Now insert the updated datasets into the loaded datasets dictionary
    for key in datasets_to_generate_features:
        loaded_train_datasets[key] = train_datasets[key]
        loaded_dev_datasets[key] = dev_datasets[key]
        loaded_test_datasets[key] = test_datasets[key]

    # Now complete updated datasets are in the loaded datasets dictionary, overwrite the return dictionary with the new updated ones 
    train_datasets = loaded_train_datasets
    dev_datasets = loaded_dev_datasets
    test_datasets = loaded_test_datasets
    for types, typeg in [('train', train_datasets), ('val', dev_datasets), ('test', test_datasets)]:
        if len(typeg[key]['act']):
            print(key, types, 'Activation min and max:', min(typeg[key]['act']), max(typeg[key]['act']))
        if len(typeg[key]['val']):
            print(key, types, 'Valence min and max:', min(typeg[key]['val']), max(typeg[key]['val']))

    # Now save any new changes to dataset
    for key in datasets_to_load:
        # Store datasets
        if conf['cache_datasets'] and was_change_to_cache[key] and not cache_read_only:
            podcast_v_str = f'v{podcast_version}' if podcast_version != '1.12' and key == 'podcast' else ''
            train_datasets[key].to_parquet(os.path.join(conf['cache_dataset_path'], f'{key}{podcast_v_str}_train.parquet'))
            dev_datasets[key].to_parquet(os.path.join(conf['cache_dataset_path'], f'{key}{podcast_v_str}_dev.parquet'))
            test_datasets[key].to_parquet(os.path.join(conf['cache_dataset_path'], f'{key}{podcast_v_str}_test.parquet'))

            # For some reason after storing datasets to disk the below audio filtering will hang indefinitely
            # not sure if the underlying huggingface code is trying to write later changes to disk as well 
            # so we just reload these datasets immediately to resolve this problem 
            train_datasets[key] = load_dataset('parquet', data_files={'train': os.path.join(conf['cache_dataset_path'], f'{key}{podcast_v_str}_train.parquet')})['train']
            dev_datasets[key] = load_dataset('parquet', data_files={'dev': os.path.join(conf['cache_dataset_path'], f'{key}{podcast_v_str}_dev.parquet')})['dev']
            test_datasets[key] = load_dataset('parquet', data_files={'test': os.path.join(conf['cache_dataset_path'], f'{key}{podcast_v_str}_test.parquet')})['test']

    # if add_enhanced_wavs:
    #     print('Datasets loaded, adding enhanced audio paths')
    #     for key in datasets_to_load:
    #         for ds_type in [train_datasets, dev_datasets, test_datasets]:
    #             dataset = ds_type[key]
    #             file_names = dataset['FileName']
    #             # TODO: should not hardcode just for podcast, this is temporary
    #             new_paths = [f'/z/public/data/SpeechEnhancedWavs/MSP-Podcast-1.12/Audios/{file_name}' for file_name in file_names]
    #             dataset = dataset.add_column('AudioEnhanced', new_paths)
    #             dataset = dataset.cast_column('Audio', Audio(sampling_rate=16000, mono=True))
    #             ds_type[key] = dataset.cast_column('AudioEnhanced', Audio(sampling_rate=16000, mono=True))

    print('Datasets loaded, filtering on audio length...')
    # Now that datasets are loaded (and possibly cached to disk) apply filtering on audio length
    # Load csv containing audio lengths for all utterances in each dataset
    len_csv_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'audiolengths.csv')
    if os.path.exists(len_csv_path):
        lengths = pd.read_csv(len_csv_path, index_col=0)
    else: # If audio length csv doesn't exist then create it
        print('Concatenating datasets to calculate audio lengths and generate audiolengths.csv for filtering')
        lengths = concatenate_datasets(list(train_datasets.values()) + list(dev_datasets.values()) + list(test_datasets.values()))
        lengths = lengths.map(add_length, num_proc=16).to_pandas()
        lengths = lengths[['Dataset', 'FileName', 'AudioLength']]
        lengths.to_csv(len_csv_path)

    # Separate lengths by dataset utterance came from 
    lengths_by_dataset = {
        'podcast': lengths[lengths['Dataset'] == 'MSP-Podcast'].set_index('FileName'),
        'improv': lengths[lengths['Dataset'] == 'MSP-Improv'].set_index('FileName'),
        'muse': lengths[lengths['Dataset'] == 'MuSE'].set_index('FileName'),
        'iemocap': lengths[lengths['Dataset'] == 'IEMOCAP'].set_index('FileName'),
    }

    # If no limit provided then we want to not filter on this so set max/min appropriately
    min_audio_len = None if conf['min_len'] == -1 else conf['min_len']
    max_audio_len = None if conf['max_len'] == -1 else conf['max_len']

    if max_audio_len is None:
        max_audio_len = lengths['AudioLength'].max()+1 # Don't want to filter any so set it higher than the max 
    if min_audio_len is None:
        min_audio_len = lengths['AudioLength'].min()-1 # Don't want to filter any so set it lower than the min

    # Now filter dataset and create new
    def filter_len(x):
        lengths = dataset_lengths.loc[x['FileName']]['AudioLength']
        not_too_short = min_audio_len <= lengths
        not_too_long = lengths <= max_audio_len
        audio_exists = lengths > 0
        return np.logical_and(np.logical_and(not_too_long, not_too_short), audio_exists)

    for key in datasets_to_load:
        dataset_lengths = lengths_by_dataset[key]
        missing_lengths = set(train_datasets[key]['FileName']) | set(dev_datasets[key]['FileName']) | set(test_datasets[key]['FileName'])
        missing_lengths -= set(dataset_lengths.index)
        if len(missing_lengths):
            print(f'Missing samples: {len(missing_lengths)}')
            lengths_new = concatenate_datasets(list(train_datasets.values()) + list(dev_datasets.values()) + list(test_datasets.values()))
            lengths_new = lengths_new.select([i for i, x in enumerate(lengths_new['FileName']) if x in missing_lengths])
            lengths_new = lengths_new.cast_column('Audio', Audio(sampling_rate=16000, mono=True))
            lengths_new = lengths_new.map(add_length, num_proc=16).to_pandas()
            lengths_new = lengths_new[['Dataset', 'FileName', 'AudioLength']]
            lengths = pd.concat([lengths, lengths_new], axis=0, ignore_index=True)
            lengths.to_csv(len_csv_path)
            lengths_by_dataset = {
                'podcast': lengths[lengths['Dataset'] == 'MSP-Podcast'].set_index('FileName'),
                'improv': lengths[lengths['Dataset'] == 'MSP-Improv'].set_index('FileName'),
                'muse': lengths[lengths['Dataset'] == 'MuSE'].set_index('FileName'),
                'iemocap': lengths[lengths['Dataset'] == 'IEMOCAP'].set_index('FileName'),
            }
            dataset_lengths = lengths_by_dataset[key]

        # Remove samples not in the min/max audio length
        train_datasets[key] = train_datasets[key].filter(filter_len, batched=True)
        dev_datasets[key] = dev_datasets[key].filter(filter_len, batched=True)
        test_datasets[key] = test_datasets[key].filter(filter_len, batched=True)

        # Now make sure dataset is in the correct format 
        format_datasets(type_to_columns, column_masks, train_datasets[key], dev_datasets[key], test_datasets[key])

    if cross_validation_folds:
        for key in datasets_to_load:
            # Concatenate datasets back into one large dataset 
            full_dataset = concatenate_datasets([train_datasets[key], dev_datasets[key], test_datasets[key]])
            train_datasets[key], dev_datasets[key], test_datasets[key] = [], [], [] # Change into list to add folds to 
            if legacy_asru_splits:
                # JSON_to_save = {f'fold_{i}': {'train_fnames': [], 'val_fnames': [], 'test_fnames': []} for i in range(5)}
                current_file_path = '/'.join(os.path.abspath(__file__).split('/')[:-1])
                fold_path = os.path.join(current_file_path, f'LegacyASRUSplits/{key}_folds.json')
                if not os.path.exists(fold_path):
                    raise IOError(f'ASRU legacy split data missing ({fold_path} does not exist)')
                with open(fold_path, 'r') as f:
                    JSON_to_save = json.load(f)
                for i in range(5):
                    train_fnames = JSON_to_save[f'fold_{i}']['train_fnames']
                    val_fnames = JSON_to_save[f'fold_{i}']['val_fnames']
                    test_fnames = JSON_to_save[f'fold_{i}']['test_fnames']
                    train_fold = full_dataset.filter(lambda x: x['FileName'] in train_fnames)
                    val_fold = full_dataset.filter(lambda x: x['FileName'] in val_fnames)
                    test_fold = full_dataset.filter(lambda x: x['FileName'] in test_fnames)
                    train_datasets[key].append(train_fold)
                    dev_datasets[key].append(val_fold)
                    test_datasets[key].append(test_fold)
            else:
                all_file_names = full_dataset['FileName']
                if key == 'iemocap':
                    if conf['return_self_report']:
                        # Create splits such that there is limited linguistic overlap, but should still be speaker-dependent
                        # In this case we need to stratify as well according to the self-report label
                        def iemocap_map_to_speaker(fname):
                            session_speaker = re.match(fr'^Ses(?P<session>\d\d)[MF]_(?P<type>impro\d\d|script\d\d)[ab]?_.*(?P<speaker_gender>[MF])\d+.wav$', fname)
                            return session_speaker.group('type')
                        speakers = [iemocap_map_to_speaker(fname) for fname in all_file_names]
                        groups = speakers
                    else:
                        # Create 5 splits according to the 5 sessions 
                        def iemocap_map_to_session(fname):
                            label_id = fname.replace('.wav', '')
                            session = re.match(r'^Ses(?P<session>\d\d).*$', label_id).group('session')
                            return int(session)-1 # Session will be 0-4 then 
                        sessions = [iemocap_map_to_session(fname) for fname in all_file_names]
                        groups = sessions
                else:
                    # Create 5 random SPEAKER INDEPENDENT splits for muse and improv 
                    if key == 'improv':
                        # Calculate the speakers from improv
                        utterance_matcher = re.compile(r'MSP-IMPROV-S(?P<sentence>\d\d)(?P<intended_emotion>[AHSN])-(?P<speaker>(?P<gender>[MF])\d\d)-(?P<scenario>[PRST])-(?P<listener>[FM])(?P<dyadic_speaker>[FM])(?P<turn_number>\d\d)')
                        speakers = [utterance_matcher.match(fname.replace('.wav', '')).group('speaker') for fname in all_file_names]
                        if conf['return_self_report']:
                            raise ValueError('Self report not supported for MSP-Improv')
                    elif key == 'muse':
                        # Calculate the speakers from MuSE
                        if conf['return_self_report']:
                            # The file naming is: SubjectID_(Audio_File_ID)_(Monologue_Number*2-1)
                            # We want to make the folds monologue-independent and speaker-dependent
                            get_monologue = lambda fname: '_'.join(fname.split('_')[1:-2])
                            speakers = [get_monologue(fname) for fname in all_file_names]
                        else:
                            speakers = [fname[:2] for fname in all_file_names]

                    groups = speakers
                if conf['return_self_report']:
                    stratify_by = []
                    for annotators in full_dataset['annotators']:
                        found_self_report = 'None'
                        for ann in annotators:
                            if 'self_report' in ann:
                                found_self_report = ann
                                break
                        stratify_by.append(found_self_report)
                    print('Stratifying group k fold by self-report label (unique values):', np.unique(stratify_by))
                    group_k_fold = StratifiedGroupKFold(n_splits=5, shuffle=False)
                    stratify_by = np.array(stratify_by)
                else:
                    stratify_by = None
                    group_k_fold = GroupKFold(n_splits=5) # Group k fold is not randomised so no need to worry about reproducibility here 
                all_file_names = np.array(all_file_names)
                groups = np.array(groups)
                fold_splits = list(
                    group_k_fold.split(
                        all_file_names,
                        y=stratify_by,
                        groups=groups,
                    )
                )
                random_generator = np.random.default_rng(seed=0)

                for i, (train_val_index, test_index) in enumerate(fold_splits):
                    if legacy_journal_splits:
                        # Journal checkpoints were trained with the original
                        # SERDatasets split logic: GroupKFold selected the test
                        # speakers and a seeded permutation selected one quarter
                        # of the remaining speaker groups for validation.
                        train_val_groups = groups[train_val_index]
                        unique_train_val_groups = np.unique(train_val_groups)
                        num_val_groups = len(unique_train_val_groups) // 4
                        val_groups = set(
                            random_generator.permutation(unique_train_val_groups)[
                                :num_val_groups
                            ]
                        )
                        val_mask = np.isin(train_val_groups, list(val_groups))
                        val_index = train_val_index[val_mask]
                        train_index = train_val_index[~val_mask]
                    else:
                        # Later experiments use the next held-out fold as the
                        # validation fold.
                        j = (i + 1) % len(fold_splits)
                        val_index = fold_splits[j][1]
                        train_index = np.concatenate([
                            fold_splits[k][1]
                            for k in range(len(fold_splits))
                            if k != i and k != j
                        ])
                    train_groups = np.unique(groups[train_index])
                    val_groups = np.unique(groups[val_index])
                    test_groups = np.unique(groups[test_index])
                    train_fnames = all_file_names[train_index]
                    val_fnames = all_file_names[val_index]
                    test_fnames = all_file_names[test_index]

                    test_fnames = set(test_fnames)
                    val_fnames = set(val_fnames)
                    train_fnames = set(train_fnames)
                    # Assert there is no overlap between splits
                    assert not (train_fnames & test_fnames) and not (train_fnames & val_fnames) and not (val_fnames & test_fnames)
                    # Assert all samples are used 
                    assert len(test_fnames) + len(val_fnames) + len(train_fnames) == len(full_dataset)
                    print(f'Fold {i} group info:\n\t{train_groups=}\n\t\tNum train samples:{len(train_fnames)}\n\t{val_groups=}\n\t\tNum val samples:{len(val_fnames)}\n\t{test_groups=}\n\t\tNum test samples:{len(test_fnames)}')
                    if key == 'muse' and conf['return_self_report']:
                        tr = set([fname[:2] for fname in train_fnames])
                        va = set([fname[:2] for fname in val_fnames])
                        te = set([fname[:2] for fname in test_fnames])
                        print(f'Fold {i} speaker info:\n\t{tr=}\n\t\tNum train samples:{len(tr)}\n\t{va=}\n\t\tNum val samples:{len(va)}\n\t{te=}\n\t\tNum test samples:{len(te)}')
                        tr = np.unique(stratify_by[train_index], return_counts=True)
                        va = np.unique(stratify_by[val_index], return_counts=True)
                        te = np.unique(stratify_by[test_index], return_counts=True)
                        print(f'Fold {i} self-report info:\n\t{tr=}\n\t\tNum train samples:{len(tr)}\n\t{va=}\n\t\tNum val samples:{len(va)}\n\t{te=}\n\t\tNum test samples:{len(te)}')

                    if key == 'iemocap' and conf['return_self_report']:
                        utterance_matcher = re.compile(r'MSP-IMPROV-S(?P<sentence>\d\d)(?P<intended_emotion>[AHSN])-(?P<speaker>(?P<gender>[MF])\d\d)-(?P<scenario>[PRST])-(?P<listener>[FM])(?P<dyadic_speaker>[FM])(?P<turn_number>\d\d)')
                        map_to_speaker = lambda match: f"{match.group('session')}{match.group('speaker_gender')}"
                        # for fname in train_fnames:
                        #     print(fname)
                        #     print(re.match(fr'^Ses(?P<session>\d\d)[MF]_(?P<type>impro\d\d|script\d\d)[ab]?_.*(?P<speaker_gender>[MF])\d+.wav$', fname))
                        #     print(map_to_speaker(re.match(fr'^Ses(?P<session>\d\d)[MF]_(?P<type>impro\d\d|script\d\d)[ab]?_.*(?P<speaker_gender>[MF])\d+.wav$', fname)))
                        #     print('-'*100)
                        tr = set([map_to_speaker(re.match(fr'^Ses(?P<session>\d\d)[MF]_(?P<type>impro\d\d|script\d\d)[ab]?_.*(?P<speaker_gender>[MF])\d+.wav$', fname)) for fname in train_fnames])
                        va = set([map_to_speaker(re.match(fr'^Ses(?P<session>\d\d)[MF]_(?P<type>impro\d\d|script\d\d)[ab]?_.*(?P<speaker_gender>[MF])\d+.wav$', fname)) for fname in val_fnames])
                        te = set([map_to_speaker(re.match(fr'^Ses(?P<session>\d\d)[MF]_(?P<type>impro\d\d|script\d\d)[ab]?_.*(?P<speaker_gender>[MF])\d+.wav$', fname)) for fname in test_fnames])
                        print(f'Fold {i} speaker info:\n\t{tr=}\n\t\tNum train samples:{len(tr)}\n\t{va=}\n\t\tNum val samples:{len(va)}\n\t{te=}\n\t\tNum test samples:{len(te)}')
                        tr = np.unique(stratify_by[train_index], return_counts=True)
                        va = np.unique(stratify_by[val_index], return_counts=True)
                        te = np.unique(stratify_by[test_index], return_counts=True)
                        print(f'Fold {i} self-report info:\n\t{tr=}\n\t\tNum train samples:{len(tr)}\n\t{va=}\n\t\tNum val samples:{len(va)}\n\t{te=}\n\t\tNum test samples:{len(te)}')
                    train_fold = full_dataset.filter(lambda x: x['FileName'] in train_fnames)
                    val_fold = full_dataset.filter(lambda x: x['FileName'] in val_fnames)
                    test_fold = full_dataset.filter(lambda x: x['FileName'] in test_fnames)
                    train_datasets[key].append(train_fold)
                    dev_datasets[key].append(val_fold)
                    test_datasets[key].append(test_fold)
    else:
        for key in datasets_to_load:
            train_datasets[key] = [train_datasets[key]]
            dev_datasets[key] = [dev_datasets[key]]
            test_datasets[key] = [test_datasets[key]]

    if prune_annotators:
        for key in datasets_to_load:
            for i in range(len(train_datasets[key])):
                train_datasets[key][i], dev_datasets[key][i], test_datasets[key][i] = prune_annotators_fn(train_datasets[key][i], dev_datasets[key][i], test_datasets[key][i], min_n=1)

    return train_datasets, dev_datasets, test_datasets

def create_kde_labels_map(batched_examples, kde_size, num_calculations=1, allow_failure=False):
    for gen in range(num_calculations):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        soft_act = torch.nn.utils.rnn.pad_sequence(batched_examples['soft_act_labels'], batch_first=True, padding_value=torch.nan).to(device, non_blocking=True)
        soft_val = torch.nn.utils.rnn.pad_sequence(batched_examples['soft_val_labels'], batch_first=True, padding_value=torch.nan).to(device, non_blocking=True)
        batch_size = soft_act.shape[0]
        annotators_per_sample = (~soft_act.isnan()).sum(dim=-1)
        if (annotators_per_sample < 2).any() and allow_failure:
            print(f'Less than 2 annotators per sample for batch {batched_examples["FileName"]}, {annotators_per_sample=}')
            print(f'SETTING UNIFORM PROBABILITY THIS IS VERY BAD BUT WE DONT USE THIS TEST SET (podcast 1.8)')
            kde_2d_prob = torch.ones(batch_size, kde_size, kde_size).to(device)
        else:
            kde_2d_prob = kde_probability_bs(soft_act, soft_val, use_soft_histogram=False, prob_grid_size=kde_size, temperature=512, density_grid_size=512, precision=torch.float64)
        if kde_2d_prob is None and allow_failure:
            print(f'KDE probability is None for batch {batched_examples["FileName"]}, {soft_act=}, {soft_val=}')
            print(f'SETTING UNIFORM PROBABILITY THIS IS VERY BAD BUT WE DONT USE THIS TEST SET (podcast 1.8)')
            kde_2d_prob = torch.ones(batch_size, kde_size, kde_size).to(device)
        negs = kde_2d_prob < 0
        if negs.any():
            raise ValueError(f'Negative values in KDE probability. Largest negative:-{kde_2d_prob[negs].abs().max()}')
        kde_2d_prob = kde_2d_prob.view(batch_size,-1)# - kde_2d_prob.view(curr_bs,-1).min(dim=-1).values.unsqueeze(dim=-1)
        kde_2d_prob = kde_2d_prob / kde_2d_prob.sum(dim=-1).unsqueeze(dim=-1)
        kde_2d_prob = kde_2d_prob.view(batch_size,kde_size,kde_size).float()
        batched_examples[f'kde_2d_probability_generation_{gen}'] = kde_2d_prob.cpu()
    if num_calculations == 1:
        batched_examples['kde_2d_probability'] = batched_examples['kde_2d_probability_generation_0']
        del batched_examples['kde_2d_probability_generation_0']
    return batched_examples

class Collator:
    def __init__(self, processor):
        self.processor = processor
        self.dataset_to_id = {'MSP-Podcast': 0, 'MSP-Improv': 1, 'MuSE': 2, 'IEMOCAP': 3}

    def __call__(self, batch):
        if not hasattr(self, 'using_cache'):
            self.using_cache = 'AudioFeatures' in batch[0]
        labels_act = [sample['act'] for sample in batch]
        labels_val = [sample['val'] for sample in batch]
        transcripts = [sample['Text'] for sample in batch]
        dataset_ids = [self.dataset_to_id[sample['Dataset']] for sample in batch]

        if not self.using_cache:
            audios = [torch.from_numpy(sample['Audio']['array']) for sample in batch]
            # Pad to longest seq length in the batch
            max_len = max([len(a) for a in audios])
            audios = [torch.nn.functional.pad(a, (0, max_len - len(a))) for a in audios]
            inputs = self.processor(audios, sampling_rate=16000, padding=True, return_tensors='pt').input_values[0]
        else:
            inputs = torch.nn.utils.rnn.pad_sequence([sample['AudioFeatures'].squeeze() for sample in batch], batch_first=True)

        labels_act = torch.tensor(labels_act)
        labels_val = torch.tensor(labels_val)
        # labels = torch.stack([labels_act, labels_val], dim=1)
        return {'inputs': inputs, 'text': transcripts, 'dataset_ids': dataset_ids, 'act': labels_act, 'val': labels_val}
