### ASRU Cross-Validation Data Splits
Cross-validation generation code has been improved since the ASRU code project. For reproducibility purposes we provide the splits calculated by the different runs of the ASRU code and the original cross validation generation code. 

### Original Code
```python3
if cross_validation_folds:
    random_generator = np.random.default_rng(seed=0)
    for key in datasets_to_load:
        # Concatenate datasets back into one large dataset 
        full_dataset = concatenate_datasets([train_datasets[key], dev_datasets[key], test_datasets[key]])
        train_datasets[key], dev_datasets[key], test_datasets[key] = [], [], [] # Change into list to add folds to 
        all_file_names = full_dataset['FileName']
        if key == 'iemocap':
            # Create 5 splits according to the 5 sessions 
            def iemocap_map_to_session(fname):
                label_id = fname.replace('.wav', '')
                session = re.match(r'^Ses(?P<session>\d\d).*$', label_id).group('session')
                return int(session)-1 # Session will be 0-4 then 
            sessions = [iemocap_map_to_session(fname) for fname in all_file_names]
            groups = sessions
            if conf['return_self_report']:
                raise ValueError('Not yet supported to do self-report cross-validation on IEMOCAP')
        else:
            # Create 5 random SPEAKER INDEPENDENT splits for muse and improv 
            if key == 'improv':
                # Calculate the speakers from improv
                utterance_matcher = re.compile(r'MSP-IMPROV-S(?P<sentence>\d\d)(?P<intended_emotion>[AHSN])-(?P<speaker>(?P<gender>[MF])\d\d)-(?P<scenario>[PRST])-(?P<listener>[FM])(?P<dyadic_speaker>[FM])(?P<turn_number>\d\d)')
                speakers = [utterance_matcher.match(fname.replace('.wav', '')).group('speaker')[1:] for fname in all_file_names]
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
        group_k_fold = GroupKFold(n_splits=5) # Group k fold is not randomised so no need to worry about reproducibility here 
        all_file_names = np.array(all_file_names)
        groups = np.array(groups)
        for i, (train_index, test_index) in enumerate(group_k_fold.split(all_file_names, y=None, groups=groups)):
            train_val_groups = groups[train_index]
            # Want to convert about 1/4 of the train index to val index, but still needs to be speaker independent
            num_val_groups = len(np.unique(train_val_groups))//4
            # Use permutation to create a randomly shuffled copy for selection of validation set
            val_groups = set(random_generator.permutation(np.unique(train_val_groups))[:num_val_groups]) # This is the only part of the fold generation
            train_groups = np.unique([g for g in train_val_groups if g not in val_groups])
            test_groups = np.unique(groups[test_index])
            train_val_fnames = all_file_names[train_index]
            train_fnames = [fname for i, fname in enumerate(train_val_fnames) if train_val_groups[i] not in val_groups]
            val_fnames = [fname for i, fname in enumerate(train_val_fnames) if train_val_groups[i] in val_groups]
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
                tr = set([fname[:2] for fname in test_fnames])
                va = set([fname[:2] for fname in test_fnames])
                te = set([fname[:2] for fname in test_fnames])
                print(f'Fold {i} speaker info:\n\t{tr=}\n\t\tNum train samples:{len(tr)}\n\t{va=}\n\t\tNum val samples:{len(va)}\n\t{te=}\n\t\tNum test samples:{len(te)}')
            train_fold = full_dataset.filter(lambda x: x['FileName'] in train_fnames)
            val_fold = full_dataset.filter(lambda x: x['FileName'] in val_fnames)
            test_fold = full_dataset.filter(lambda x: x['FileName'] in test_fnames)
            train_datasets[key].append(train_fold)
            dev_datasets[key].append(val_fold)
            test_datasets[key].append(test_fold)

```