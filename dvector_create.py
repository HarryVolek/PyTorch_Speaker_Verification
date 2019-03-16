#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 14:34:01 2018

@author: Harry

Creates "segment level d vector embeddings" compatible with
https://github.com/google/uis-rnn

"""

import glob
import librosa
import numpy as np
import os
import torch

from hparam import hparam as hp
from speech_embedder_net import SpeechEmbedder
from VAD_segments import VAD_chunk


def concat_segs(times, segs):
    #Concatenate continuous voiced segments
    concat_seg = []
    seg_concat = segs[0]
    for i in range(0, len(times)-1):
        if times[i][1] == times[i+1][0]:
            seg_concat = np.concatenate((seg_concat, segs[i+1]))
        else:
            concat_seg.append(seg_concat)
            seg_concat = segs[i+1]
    else:
        concat_seg.append(seg_concat)
    return concat_seg

def get_STFTs(segs):
    #Get 240ms STFT windows with 50% overlap
    sr = hp.data.sr
    STFT_frames = []
    for seg in segs:
        S = librosa.core.stft(y=seg, n_fft=hp.data.nfft,
                              win_length=int(hp.data.window * sr), hop_length=int(hp.data.hop * sr))
        S = np.abs(S)**2
        mel_basis = librosa.filters.mel(sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
        S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances
        for j in range(0, S.shape[1], int(.12/hp.data.hop)):
            if j + 24 < S.shape[1]:
                STFT_frames.append(S[:,j:j+24])
            else:
                break
    return STFT_frames

def align_embeddings(embeddings):
    partitions = []
    start = 0
    end = 0
    j = 1
    for i, embedding in enumerate(embeddings):
        if (i*.12)+.24 < j*.401:
            end = end + 1
        else:
            partitions.append((start,end))
            start = end
            end = end + 1
            j += 1
    else:
        partitions.append((start,end))
    avg_embeddings = np.zeros((len(partitions),256))
    for i, partition in enumerate(partitions):
        avg_embeddings[i] = np.average(embeddings[partition[0]:partition[1]],axis=0) 
    return avg_embeddings
#dataset path
audio_path = glob.glob(os.path.dirname(hp.unprocessed_data))  

total_speaker_num = len(audio_path)
train_speaker_num= (total_speaker_num//10)*9            # split total data 90% train and 10% test

embedder_net = SpeechEmbedder()
embedder_net.load_state_dict(torch.load(hp.model.model_path))
embedder_net.eval()

train_sequence = []
train_cluster_id = []
label = 0
count = 0
train_saved = False
for i, folder in enumerate(audio_path):
    for file in os.listdir(folder):
        if file[-4:] == '.wav':
            times, segs = VAD_chunk(2, folder+'/'+file)
            if segs == []:
                print('No voice activity detected')
                continue
            concat_seg = concat_segs(times, segs)
            STFT_frames = get_STFTs(concat_seg)
            STFT_frames = np.stack(STFT_frames, axis=2)
            STFT_frames = torch.tensor(np.transpose(STFT_frames, axes=(2,1,0)))
            embeddings = embedder_net(STFT_frames)
            aligned_embeddings = align_embeddings(embeddings.detach().numpy())
            train_sequence.append(aligned_embeddings)
            for embedding in aligned_embeddings:
                train_cluster_id.append(str(label))
            count = count + 1
            if count % 100 == 0:
                print('Processed {0}/{1} files'.format(count, len(audio_path)))
    label = label + 1
    
    if not train_saved and i > train_speaker_num:
        train_sequence = np.concatenate(train_sequence,axis=0)
        train_cluster_id = np.asarray(train_cluster_id)
        np.save('train_sequence',train_sequence)
        np.save('train_cluster_id',train_cluster_id)
        train_saved = True
        train_sequence = []
        train_cluster_id = []
        
train_sequence = np.concatenate(train_sequence,axis=0)
train_cluster_id = np.asarray(train_cluster_id)
np.save('test_sequence',train_sequence)
np.save('test_cluster_id',train_cluster_id)
