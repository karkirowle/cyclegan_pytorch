from typing import Callable, List
from nnmnkwii.datasets import FileDataSource

import librosa
import numpy as np
import matplotlib.pyplot as plt

from utils import world_decompose, world_encode_spectral_envelop, wav_padding

import os
import re

from os.path import join, splitext, isdir, basename
from os import listdir

available_speakers = ['F02', 'F03', 'F04', 'F05', 'M01', 'M04',
                      'M05', 'M07', 'M08', 'M09', 'M10', 'M11', 'M12', 'M14', 'M16']

def wav_filter(x):
    match = re.match(
        r'(?P<speaker>[\w]+)\_(?P<rep>[\w]+)\_(?P<file_id>\w+)\_(?P<mic>\w+)\.wav', x)
    if match:
        name_dict = match.groupdict()
        criterion = name_dict['rep'] == 'B2' or (
            name_dict['rep'] == 'B1' or name_dict['rep'] == 'B3') and name_dict['file_id'].startswith('UW')
        return criterion and name_dict['mic'] == 'M3'
    else:
        return False

class UASpeechDataSource(FileDataSource):

    def __init__(self, data_root, cache_dir, speakers, labelmap=None, max_files=None,
                 training=True,
                 validating=False,
                 num_features=24,
                 preprocess_function: Callable[[List[float]], np.ndarray]=None):

        for speaker in speakers:
            if speaker not in available_speakers:
                raise ValueError(
                    "Unknown speaker '{}'. It should be one of {}".format(
                        speaker, available_speakers))

        self.data_root = data_root
        self.cache_dir = cache_dir or "./preprocessed/UASpeech"
        self.training = training
        self.speakers = speakers
        self.validating = validating
        if labelmap is None:
            labelmap = {}
            for i, speaker in enumerate(self.speakers):
                labelmap[speaker] = i
        self.labelmap = labelmap
        self.max_files = max_files
        self.labels = None
        self.num_features = num_features
        self.preprocess_function  = preprocess_function

        if not os.path.exists(os.path.join(self.cache_dir)):
            os.makedirs(self.cache_dir)


    def collect_files(self):
        speaker_dirs = [join(self.data_root, x)
                        for x in self.speakers] if self.training else [join(self.data_root, 'C' + x) for x in self.speakers]
        paths = []
        labels = []

        if self.max_files is None:
            max_files_per_speaker = None
        else:
            max_files_per_speaker = self.max_files // len(self.speakers)
        for (i, d) in enumerate(speaker_dirs):
            if not isdir(d):
                raise RuntimeError("{} doesn't exist.".format(d))
            files = [join(speaker_dirs[i], f) for f in listdir(d)]
            # files = list(filter(lambda x: splitext(x)[1] == ".wav", files))
            files = list(filter(lambda x: wav_filter(basename(x)), files))
            files = sorted(files)
            files = files[: max_files_per_speaker]

            for f in files[: max_files_per_speaker]:
                paths.append(f)
                labels.append(self.labelmap[self.speakers[i]])
                
        self.labels = np.array(labels, dtype=np.int32)
        return paths

    def collect_features(self, file_path):

        sr = 16000
        save_path = os.path.join(
            self.cache_dir, basename(file_path))

        if os.path.exists(save_path):
            features = np.load(save_path, allow_pickle=True)
        else:
            if self.preprocess_function is not None:
                wav, _ = librosa.load(file_path, sr=sr, mono=True)
                features = self.preprocess_function(wav)
                features.dump(save_path)
            else:
                raise Exception("Data has not been preprocessed yet, please preprocess data first.")
            
            # f0, _, sp, ap = world_decompose(wav, sr)
            # mcep = world_encode_spectral_envelop(sp, sr, dim=self.num_features)

            # features = np.hstack((f0[:, None], mcep, ap))
            # features.dump(save_path)

        return features
