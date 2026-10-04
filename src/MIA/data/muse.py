import os
import re
import pandas as pd
from datasets import Dataset, Audio

def map_to_split(utterance_id):
    speaker = utterance_id[:2]
    train_f_speakers = ['20', '17', '25', '22', '08'] # 5 f 11 m speakers
    val_f_speakers = ['27', '11'] # 2 f 4 m speakers 
    test_f_speakers = ['19', '23'] # 2 f 4 m speakers 
    train_m_speakers = ['24', '12', '26', '21', '01', '04', '07', '03', '16', '06', '13']
    val_m_speakers = ['15', '05', '09', '10']
    test_m_speakers = ['14', '02', '18', '28']
    if speaker in train_f_speakers + train_m_speakers:
        return 'Train'
    elif speaker in val_f_speakers + val_m_speakers:
        return 'Validation'
    elif speaker in test_f_speakers + test_m_speakers:
        return 'Test'
    else:
        raise ValueError(f'Unknown data split for utterance: {utterance_id}')

def map_to_filepath(row):
    stress_type = row['Type']
    if stress_type == 'NS':
        stress_path = os.path.join('NotStressed', 'Audio', 'Non Stressed Question Monologue Audio [Sentence Split]')
    elif stress_type == 'S':
        stress_path = os.path.join('Stressed', 'Audio', 'Stressed Question Monologue Audio')
    else:
        raise ValueError('Unknown stress type', stress_type, 'for file:', row['FileName'])
    return os.path.join(stress_path, row['FileName'])

def map_to_transcript(row):
    stress_type = row['Type']
    if stress_type == 'NS':
        stress_path = os.path.join('Whisper-Transcriptions', 'Nonstressed_Segments')
    elif stress_type == 'S':
        stress_path = os.path.join('Whisper-Transcriptions', 'Stressed_Segments')
    else:
        raise ValueError('Unknown stress type', stress_type, 'for file:', row['FileName'])
    return os.path.join(stress_path, row['FileName'].replace('.wav', '.txt'))

def read_transcript(path):
    with open(path, 'r') as f:
        text = f.read()
    return text

def read_muse(dataset_dir, labels_path, columns):
    labels_df = pd.read_csv(labels_path)
    labels_df = labels_df.rename(columns={'Sentence_name': 'FileName', 'Activation_Mean': 'act', 'Valence_Mean': 'val',
                                          'S_Activation': 'self-report-act', 'S_Valence': 'self-report-val','Gender': 'gender'})
    labels_df['soft_act_labels'] = labels_df['Activation_Annotation'].map(lambda values: [int(x) for x in values.split(';')])
    labels_df['soft_val_labels'] = labels_df['Valence_Annotation'].map(lambda values: [int(x) for x in values.split(';')])
    labels_df['Split_Set'] = labels_df['FileName'].apply(lambda x: map_to_split(x))
    labels_df['Audio'] = labels_df.apply(lambda x: os.path.join(dataset_dir, map_to_filepath(x)), axis=1)
    labels_df['Text'] = labels_df.apply(lambda x: read_transcript(os.path.join(dataset_dir, map_to_transcript(x))), axis=1)
    labels_df['Dataset'] = 'MuSE'
    # MuSE is missing many columns present in other datasets so ensure columns don't contain any unknown columns
    missing = [col for col in columns if col not in labels_df.columns]
    columns = [col for col in columns if col in labels_df.columns]
    if len(missing):
        print('Warning MuSE returning empty columns for:', missing)
    labels_df = labels_df[columns]

    train_df = labels_df[labels_df['Split_Set'] == 'Train']
    dev_df = labels_df[labels_df['Split_Set'] == 'Validation']
    test_df = labels_df[labels_df['Split_Set'] == 'Test']

    return train_df, dev_df, test_df
