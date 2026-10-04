import os
import torch
import numpy as np
import pandas as pd
# min_v, max_v: min and max value present in original data
# new_min, new_max: new min and max to be min-max scaled to
# x: value to scale
def scale(min_v, max_v, new_min, new_max, x):
    return new_min + (x-min_v)*(new_max-new_min)/(max_v-min_v)

def scale_dataset(dataset, minv=1, maxv=7):
    dataset['act'] = scale(minv,maxv,-1,1,dataset['act'])
    dataset['val'] = scale(minv,maxv,-1,1,dataset['val'])
    if 'soft_act_labels' in dataset:
        dataset['soft_act_labels'] = [scale(minv, maxv, -1, 1, np.array(x)) for x in dataset['soft_act_labels']]
        # dataset['soft_act_labels'] = [scale(minv, maxv, -1, 1, x) for x in dataset['soft_act_labels']]
        dataset['soft_val_labels'] = [scale(minv, maxv, -1, 1, np.array(x)) for x in dataset['soft_val_labels']]
        # dataset['soft_val_labels'] = [scale(minv, maxv, -1, 1, x) for x in dataset['soft_val_labels']]
    return dataset

def map_annotators_to_sample_idxs(dataset):
    annotator_to_idxs = {}
    for i, annotators in enumerate(dataset['annotators']):
        for ann in annotators:
            if ann not in annotator_to_idxs:
                annotator_to_idxs[ann] = []
            annotator_to_idxs[ann].append(i)
    for ann in annotator_to_idxs:
        annotator_to_idxs[ann] = list(set(annotator_to_idxs[ann]))
    return annotator_to_idxs

# Remove all annotators with less than n annotations in the train dataset
# also remove all annotators not seen in the train dataset from val and test dataset
def prune_annotators_fn(train_dataset, val_dataset, test_dataset, min_n):
    annotator_to_idxs = map_annotators_to_sample_idxs(train_dataset)
    pre_remove = len(annotator_to_idxs)
    for ann in list(annotator_to_idxs.keys()):
        if len(annotator_to_idxs[ann]) < min_n:
            del annotator_to_idxs[ann]
    num_removed = pre_remove - len(annotator_to_idxs)
    print(f'Removing {num_removed} annotators from training set ({pre_remove} -> {len(annotator_to_idxs)})')
    # We now have a list of all annotators that are present in the training set with > 30 annotators
    # So we can use this to filter all datasets
    # First we need to remove samples that will not contain any valid annotators
    def filter_labels(sample):
        annotators_in_train_set = [ann in annotator_to_idxs for ann in sample['annotators']]
        return any(annotators_in_train_set)
    if num_removed: # If we didn't remove any training data then no operation required for train
        pruned_train = train_dataset.filter(filter_labels)
    else:
        pruned_train = train_dataset
    # If no training annotators were removed it is possible that no modifications are required
    # quickly check if there are any annotators in validation/test set that actually need removing
    # should only occur if there are any annotators not present in training set
    val_anns = set([ann for sample in val_dataset['annotators'] for ann in sample])
    test_anns = set([ann for sample in test_dataset['annotators'] for ann in sample])
    val_in_train = np.array([ann in annotator_to_idxs for ann in val_anns])
    test_in_train = np.array([ann in annotator_to_idxs for ann in test_anns])
    if not num_removed and val_in_train.all() and test_in_train.all():
        print('Did not filter datasets; no changes required')
        return train_dataset, val_dataset, test_dataset
    
    print(f'Removing {len(val_in_train)-val_in_train.sum()} annotators from validation set ({len(val_in_train)} -> {val_in_train.sum()})')
    print(f'Removing {len(test_in_train)-test_in_train.sum()} annotators from test set ({len(test_in_train)} -> {test_in_train.sum()})')
    
    pruned_val = val_dataset.filter(filter_labels)
    pruned_test = test_dataset.filter(filter_labels)
    print(f'Filtered datasets (train, val, test) from size: ({len(train_dataset)}, {len(val_dataset)}, {len(test_dataset)}) -> ({len(pruned_train)}, {len(pruned_val)}, {len(pruned_test)})')
    def prune_labels(sample):
        new_soft_act = []
        new_soft_val = []
        new_annotators = []
        for i, ann in enumerate(sample['annotators']):
            if ann in annotator_to_idxs:
                new_soft_act.append(torch.as_tensor(sample['soft_act_labels'][i]))
                new_soft_val.append(torch.as_tensor(sample['soft_val_labels'][i]))
                new_annotators.append(ann)
        if len(new_soft_act) == 0:
            raise ValueError('0 Annotators')
        # print(new_soft_act)
        new_soft_act = torch.stack(new_soft_act)
        new_soft_val = torch.stack(new_soft_val)
        return {**sample, 'soft_act_labels': new_soft_act, 'soft_val_labels': new_soft_val, 'act': new_soft_act.mean(), 'val': new_soft_val.mean(), 'annotators': new_annotators}
    if num_removed:
        pruned_train = pruned_train.map(prune_labels)
    pruned_val = pruned_val.map(prune_labels)
    pruned_test = pruned_test.map(prune_labels)

    return pruned_train, pruned_val, pruned_test

def add_muse_annotators_NO_SELF_REPORT(train_dataset, val_dataset, test_dataset, muse_annotator_info_path):
    train_dataset, val_dataset, test_dataset = add_muse_helper(
        train_dataset,
        val_dataset,
        test_dataset,
        drop_last=True,
        muse_annotator_info_path=muse_annotator_info_path,
    )

    all_ann = [a for ann in train_dataset['annotators'].to_list() + val_dataset['annotators'].to_list() + test_dataset['annotators'].to_list() for a in ann]
    if any([a.startswith('self-report') for a in all_ann]):
        raise ValueError('Failed to remove self report')

    return train_dataset, val_dataset, test_dataset

def add_muse_annotators(train_dataset, val_dataset, test_dataset, muse_annotator_info_path):
    return add_muse_helper(
        train_dataset,
        val_dataset,
        test_dataset,
        drop_last=False,
        muse_annotator_info_path=muse_annotator_info_path,
    )

def add_muse_helper(train_dataset, val_dataset, test_dataset, drop_last, muse_annotator_info_path):
    if not muse_annotator_info_path or not os.path.isfile(muse_annotator_info_path):
        raise FileNotFoundError(
            'MuSE annotator metadata is restricted and is not distributed with this repository. '
            'Set muse_annotator_info_path in data_config.yaml to an authorized local copy.'
        )
    muse_ann_df = pd.read_csv(muse_annotator_info_path)
    muse_ann_df = muse_ann_df.set_index('FileName')
    muse_ann_df.soft_act_labels = muse_ann_df.soft_act_labels.apply(lambda s: [float(x.strip(' []')) for x in s.split(',')])
    muse_ann_df.soft_val_labels = muse_ann_df.soft_val_labels.apply(lambda s: [float(x.strip(' []')) for x in s.split(',')])
    muse_ann_df.annotators = muse_ann_df.annotators.apply(lambda s: [x.strip(' []\'') for x in s.split(',')])
    def convert_dataset(sample):
        new_values = muse_ann_df.loc[sample['FileName']].to_dict()
        if drop_last:
            sample['soft_act_labels'] = new_values['soft_act_labels'][:-1]
            sample['soft_val_labels'] = new_values['soft_val_labels'][:-1]
            sample['annotators'] = new_values['annotators'][:-1]
        else:
            sample['soft_act_labels'] = new_values['soft_act_labels']
            sample['soft_val_labels'] = new_values['soft_val_labels']
            sample['annotators'] = new_values['annotators']
        sample['act'] = np.mean(sample['soft_act_labels'])
        sample['val'] = np.mean(sample['soft_val_labels'])
        return sample

    train_dataset = train_dataset.apply(convert_dataset, axis=1)
    val_dataset = val_dataset.apply(convert_dataset, axis=1)
    test_dataset = test_dataset.apply(convert_dataset, axis=1)

    return train_dataset, val_dataset, test_dataset
