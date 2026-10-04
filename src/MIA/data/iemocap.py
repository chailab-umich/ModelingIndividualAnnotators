import os
import re
import pandas as pd
from glob import glob
from datasets import Dataset, Audio

def read_iemocap(dataset_dir, labels_path, columns):
    get_annotator_scores = re.compile(r'(?P<annotator>[^:]+)?.*val\s+(?P<val>[1-5]\.?\d?);\s+act\s+(?P<act>[1-5]\.?\d?);\s+dom\s+(?P<dom>[1-5]\.?\d?);.*')
    label_info = {}
    labels = {}
    with open(labels_path, 'r') as file:
        section = []
        for line in file:
            # Skip the first line
            if 'TURN_NAME' in line: continue
            line = line.rstrip()
            if line == '' and len(section):
                # Empty line - end of this section, store in the label info
                session_id = section[0].split('\t')[1]
                label_info[session_id] = section
                section = []
            elif line != '':
                section.append(line)
    
    # We now just need to process each of the sections and also load the transcript for each id
    for label_id in label_info:
        if label_id in labels:
            raise IOError(f'Multiple labels for the same speech {label_id}')
        labels[label_id] = {'soft_act_labels': [], 'soft_val_labels': [], 'soft_dom_labels': [], 'self-report-annotators': [], 'annotators': []}
        for line in label_info[label_id]:
            # First line contains averaged labels
            if line.startswith('[') and line.endswith(']'):
                cat_lbl = line.split("\t")[2]
                val_lbl = float(line.split("\t")[3][1:-1].split(", ")[0])
                act_lbl = float(line.split("\t")[3][1:-1].split(", ")[1])
                dom_lbl = float(line.split("\t")[3][1:-1].split(", ")[2])
                labels[label_id]['act'] = act_lbl
                labels[label_id]['val'] = val_lbl
                labels[label_id]['dom'] = dom_lbl
                labels[label_id]['categorical'] = cat_lbl
                labels[label_id]['gender'] = re.match(r'.*(?P<gender>[FM])\d+$', label_id).group('gender')
                session = re.match(r'^Ses(?P<session>\d\d).*$', label_id).group('session')
                labels[label_id]['speaker_id'] = f'{session}{labels[label_id]["gender"]}'
            elif line.startswith('A-E'):
                # Attribute perception of other annotator
                regex_match = get_annotator_scores.match(line)
                if regex_match:
                    labels[label_id]['soft_val_labels'].append(int(regex_match.group('val')))
                    labels[label_id]['soft_act_labels'].append(int(regex_match.group('act')))
                    labels[label_id]['soft_dom_labels'].append(int(regex_match.group('dom')))
                    annotator = regex_match.group('annotator')
                    # Warning: Individual annotator votes can be reconstructed as the soft label variables above will align with annotators list, so we can pair up readings
                    # the individual annotators stored in huggingface format becomes an insanely large dictionary and takes over a minute per batch to read from disk 
                    # probably best to delete these when loading the dataset and recalculate it at a later date
                    labels[label_id]['annotators'].append(annotator)
                else:
                    # Bad label that is not in the range 1-5 or is just blank 
                    print(f'Bad label {label_id}: {line}')
                    del labels[label_id]
                    break
            elif line.startswith('A-F') or line.startswith('A-M'):
                # Attribute perception of self annotator
                if 'self-report' in labels[label_id]:
                    raise IOError(f'Found multiple self-report scores for {label_id}.')
                regex_match = get_annotator_scores.match(line)
                if regex_match:
                    labels[label_id]['self-report-val'] = float(regex_match.group('val'))
                    labels[label_id]['self-report-act'] = float(regex_match.group('act'))
                    labels[label_id]['self-report-dom'] = float(regex_match.group('dom'))
                    annotator = regex_match.group('annotator')
                    labels[label_id]['self-report-annotators'].append(annotator)
                else:
                    # Bad label that is not in the range 1-5 or is just blank 
                    print(f'Bad label {label_id}: {line}')
                    del labels[label_id]
            elif line.startswith('C-'):
                # All categorical based evaluations
                pass
            else:
                raise IOError('Unknown line in IEMOCAP label file - cannot process:', line)

    # Now read the transcripts
    transcript_files = glob(os.path.join(dataset_dir, '**/dialog/transcriptions/*.txt'))
    for transcript_file in transcript_files:
        with open(transcript_file, 'r') as file:
            for line in file:
                line = line.rstrip()
                split_line = line.split()
                utt_id, transcript = split_line[0], ' '.join(split_line[2:])
                if utt_id in labels:
                    labels[utt_id]['Text'] = transcript
                else:
                    print(f'No label found for transcript {utt_id} in file {transcript_file}')

    labels_dict = []
    for label_id in labels:
        filename = f'{label_id.strip()}.wav'
        # Now calculate the audio path 
        session_number = filename[4]
        session_num_type = '_'.join(filename.split('_')[:-1])
        audio_path = os.path.join(dataset_dir, f'Session{session_number}', 'sentences', 'wav', session_num_type, filename)
        if filename.startswith('Ses01') or filename.startswith('Ses02') or filename.startswith('Ses03'):
            split = 'Train'
        elif filename.startswith('Ses04'):
            split = 'Validation'
        elif filename.startswith('Ses05'):
            split = 'Test'
        else:
            raise ValueError(f'Unknown split for {filename}')
        labels_dict.append({
            'Audio': audio_path, 'Split_Set': split, 'FileName': filename, 'Dataset': 'IEMOCAP',
            **labels[label_id]
        })

    labels_df = pd.DataFrame(labels_dict)
    # labels_df['Audio'] = labels_df['FileName'].apply(lambda x: os.path.join(audio_dir, x))
    missing = [col for col in columns if col not in labels_df.columns]
    columns = [col for col in columns if col in labels_df.columns]
    if len(missing): # TODO: Better way of doing this rather than copying from muse
        # Missing are due to new columns generated during caching (AudioFeatures) not present in pre-processed dataset
        print('Warning IEMOCAP returning empty columns for:', missing)
    labels_df = labels_df[columns]

    train_df = labels_df[labels_df['Split_Set'] == 'Train']
    dev_df = labels_df[labels_df['Split_Set'] == 'Validation']
    test_df = labels_df[labels_df['Split_Set'] == 'Test']

    return train_df, dev_df, test_df
