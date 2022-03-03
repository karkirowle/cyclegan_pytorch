from typing import Callable, List
from nnmnkwii.datasets import FileDataSource
import librosa
import numpy as np

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
        """

        Class constructor for management of the UASpeech dataset

        :param data_root: Location of the UASpeech root on your disk
        :param cache_dir: Place to save the features
        :param speakers: List of speakers to include in the data source
        :param labelmap: Dictionary assigning an interger label to speaker (TODO: Not sure if this used/needed)
        :param max_files: Limits the number of files to use (works on a single speaker)
        :param training: (TODO: Does not seem to actually correspond to training/testing, rather control/validation)
        :param validating: (TODO: Does not seem to be useful, should be deprecated)
        :param num_features: number of MCEP features to extract using the WORLD vocoder, recommended to leave at 24
        :param preprocess_function: Python function which takes an audio and returns a feature
        """

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
        """
        Collects the files eligible based on the public members of the class constructor
        :return:
        """
        speaker_dirs = [join(self.data_root, x)
                        for x in self.speakers] \
            if self.training else [join(self.data_root, 'C' + x) for x in self.speakers]
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
            files = list(filter(lambda x: wav_filter(basename(x)), files))
            files = sorted(files)
            files = files[: max_files_per_speaker]

            for f in files[: max_files_per_speaker]:
                paths.append(f)
                labels.append(self.labelmap[self.speakers[i]])
                
        self.labels = np.array(labels, dtype=np.int32)
        return paths

    def collect_features(self, file_path):
        """
        Extracts the MCEP features or loads the MCEP features from a file

        :param file_path: full file path to the audio wave
        :return:
        """

        sr = 16000
        save_path = os.path.join(self.cache_dir, basename(file_path))

        if os.path.exists(save_path):
            features = np.load(save_path, allow_pickle=True)
        else:
            if self.preprocess_function is not None:
                wav, _ = librosa.load(file_path, sr=sr, mono=True)
                features = self.preprocess_function(wav)
                features.dump(save_path)
            else:
                raise Exception("Data has not been preprocessed yet, please preprocess data first.")
            
        return features
