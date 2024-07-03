import re
import os
import copy
import tqdm
import glob
import json
import math
import shutil
import random
import pickle
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

import config

# t: ms
def convert_time(t):
    t = int(t)
    ms = t % 1000
    t = math.floor(t / 1000)
    h = math.floor(t / 3600)
    m = math.floor((t - h * 3600) / 60)
    s = t - 3600 * h - 60 * m
    return '%02d:%02d:%02d.%03d' % (h, m, s, ms)

def select_videos_for_cmumosei():
    data_root = '../emotion-data/CMUMOSEI'
    trans_file = os.path.join(data_root, 'transcription.csv')
    video_root = os.path.join(data_root, 'whole_video')
    save_root = os.path.join(data_root, 'subvideo')
    if not os.path.exists(save_root): os.makedirs(save_root)

    error_lines = []
    df = pd.read_csv(trans_file)
    for idx, row in tqdm.tqdm(df.iterrows()):
        name = row['name']
        sentence = row['sentence']
        video_path = os.path.join(video_root, name + '.mp4')
        if not os.path.exists(video_path):
            error_lines.append(name)
        else:
            save_path = os.path.join(save_root, name + '.mp4')
            cmd = f'cp {video_path} {save_path}'
            os.system(cmd)
    print(f'error samples: {len(error_lines)}')


######################################################
######################################################
## gain name2features [only one speaker]
def feature_compressed(feature_root, save_root):
    names = os.listdir(feature_root)

    ## (names, speakers) => features
    features = []
    feature_dim = -1
    for ii, name in enumerate(names):
        print(f'Process {name}: {ii + 1}/{len(names)}')
        feature = []
        feature_dir = os.path.join(feature_root, name)
        facenames = os.listdir(feature_dir)
        for facename in sorted(facenames):
            facefeat = np.load(os.path.join(feature_dir, facename))
            feature_dim = max(feature_dim, facefeat.shape[-1])
            feature.append(facefeat)

        single_feature = np.array(feature).squeeze()
        if len(single_feature) == 0:
            single_feature = np.zeros((feature_dim,))
        elif len(single_feature.shape) == 2:  # [seqlen, featdim]
            single_feature = np.mean(single_feature, axis=0)
        features.append(single_feature)

    ## save (names, features)
    if not os.path.exists(save_root): os.makedirs(save_root)
    for ii in range(len(names)):
        save_path = os.path.join(save_root, names[ii] + '.npy')
        np.save(save_path, features[ii])


## gain name2features [two speakers]
def feature_compressed_iemocap(feature_root, save_root):
    names = os.listdir(feature_root)

    ## (names, speakers) => features
    features = []
    feature_dim = -1
    for ii, name in enumerate(names):
        print(f'Process {name}: {ii + 1}/{len(names)}')
        feature = {'F': [], 'M': []}
        feature_dir = os.path.join(feature_root, name)
        facenames = os.listdir(feature_dir)
        for facename in sorted(facenames):
            assert facename.find('F') >= 0 or facename.find('M') >= 0
            facefeat = np.load(os.path.join(feature_dir, facename))
            feature_dim = max(feature_dim, facefeat.shape[-1])
            if facename.find('F') >= 0:
                feature['F'].append(facefeat)
            else:
                feature['M'].append(facefeat)

        for speaker in feature:
            single_feature = np.array(feature[speaker]).squeeze()
            if len(single_feature) == 0:
                single_feature = np.zeros((feature_dim,))
            elif len(single_feature.shape) == 2:
                single_feature = np.mean(single_feature, axis=0)
            feature[speaker] = single_feature
        features.append(feature)

    ## save (names, features)
    if not os.path.exists(save_root): os.makedirs(save_root)
    for ii in range(len(names)):
        save_subroot = os.path.join(save_root, names[ii])
        if not os.path.exists(save_subroot): os.makedirs(save_subroot)
        feature = features[ii]
        for speaker in feature:
            save_path = os.path.join(save_subroot, f'compress_{speaker}.npy')
            np.save(save_path, feature[speaker])


######################################################
######################################################
def generate_transcription_files_IEMOCAP():
    ## gain all transcriptions
    names = []
    sentences = []
    data_root = '../emotion-data/IEMOCAP'
    for session_name in ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']:
        transcription_root = os.path.join(data_root, session_name, 'dialog/transcriptions')
        for trans_path in glob.glob(transcription_root + '/S*.txt'):
            with open(trans_path, encoding='utf8') as f:
                lines = [line.strip() for line in f]
            lines = [line for line in lines if len(line) != 0]
            for line in lines:  # line: Ses05F_script03_1_F033 [241.6700-243.4048]: You knew there was nothing.
                try:  # some line cannot be processed
                    subname = line.split(' [')[0]
                    start = line.split('[')[1].split('-')[0]
                    end = line.split('-')[1].split(']')[0]
                    start = convert_time(float(start) * 1000)
                    end = convert_time(float(end) * 1000)
                    sentence = line.split(']:')[1].strip()
                    names.append(subname)
                    sentences.append(sentence)
                except:
                    continue

    ## write to csv file
    csv_file = 'dataset/IEMOCAP/transcription.csv'
    columns = ['name', 'sentence']
    data = np.column_stack([names, sentences])
    df = pd.DataFrame(data=data, columns=columns)
    df[columns] = df[columns].astype(str)
    df.to_csv(csv_file, index=False)


def generate_transcription_files_CMUMOSI():
    ## read pkl file
    # pkl_path = 'dataset/CMUMOSI/CMUMOSI_features_raw_7way.pkl'
    pkl_path = 'dataset/CMUMOSI/CMUMOSI_features_raw_2way.pkl'
    names, sentences = [], []
    videoIDs, videoLabels, videoSpeakers, videoSentences, trainVid, valVid, testVid = pickle.load(open(pkl_path, "rb"),
                                                                                                  encoding='latin1')
    for vid in videoIDs:
        uids = videoIDs[vid]
        sens = videoSentences[vid]
        assert len(uids) == len(sens)
        for ii in range(len(uids)):
            names.append(uids[ii])
            sentences.append(sens[ii])

    ## write to csv file
    csv_file = 'dataset/CMUMOSI/transcription.csv'
    columns = ['name', 'sentence']
    data = np.column_stack([names, sentences])
    df = pd.DataFrame(data=data, columns=columns)
    df[columns] = df[columns].astype(str)
    df.to_csv(csv_file, index=False)


def generate_transcription_files_CMUMOSEI():
    ## read pkl file
    pkl_path = 'dataset/CMUMOSEI/CMUMOSEI_features_raw_2way.pkl'
    names, sentences = [], []
    videoIDs, videoLabels, videoSpeakers, videoSentences, trainVid, valVid, testVid = pickle.load(open(pkl_path, "rb"),
                                                                                                  encoding='latin1')
    for vid in videoIDs:
        uids = videoIDs[vid]
        sens = videoSentences[vid]
        assert len(uids) == len(sens)
        for ii in range(len(uids)):
            names.append(uids[ii])
            sentences.append(sens[ii])

    ## write to csv file
    csv_file = 'dataset/CMUMOSEI/transcription.csv'
    columns = ['name', 'sentence']
    data = np.column_stack([names, sentences])
    df = pd.DataFrame(data=data, columns=columns)
    df[columns] = df[columns].astype(str)
    df.to_csv(csv_file, index=False)


if __name__ == '__main__':
    import fire

    fire.Fire()