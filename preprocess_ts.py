from nnmnkwii.datasets import FileSourceDataset, MemoryCacheDataset
from nnmnkwii.preprocessing import meanstd
from uaspeech import available_speakers
from uaspeech import UASpeechDataSource
from mcep_wrapper import MCEPWrapper
from os.path import join, splitext, isdir, basename
from utils import world_decompose, world_encode_spectral_envelop

import argparse
import pickle
import itertools
import os
import numpy as np

DATA_ROOT = "/tudelft.net/staff-bulk/ewi/insy/SpeechLab/mlrprananta/fast_uaspeech"
CACHE_ROOT = './preprocessed/UASpeech'

def preprocess(cache_dir, speakers, num_features):

    if not isdir(cache_dir):
        os.mkdir(cache_dir)

    def preprocess(wav):
        f0, _, sp, ap = world_decompose(wav, 16000)
        mcep = world_encode_spectral_envelop(sp, 16000, dim=num_features)
        features = np.hstack((f0[:, None], mcep, ap))
        return features

    speakers.sort()

    # for speaker in speakers:
    input_set = MemoryCacheDataset(FileSourceDataset(UASpeechDataSource(DATA_ROOT, cache_dir, speakers, num_features=num_features, preprocess_function=preprocess)))
    for _ in input_set:
        pass

    # for fold in itertools.combinations(speakers, len(speakers) - 1):
    #     input_set = MemoryCacheDataset(FileSourceDataset(UASpeechDataSource(DATA_ROOT, cache_dir, fold, num_features=num_features, preprocess_function=preprocess)))
    #     # output_set = MemoryCacheDataset(FileSourceDataset(UASpeechDataSource(DATA_ROOT, cache_dir, fold, training=False, num_features=num_features, preprocess_function=preprocess)))

    #     meanstd(input_set, [len(y) for y in input_set])
    #     # meanstd(output_set, [len(y) for y in output_set])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Preprocess training data.")

    parser.add_argument('-c', '--cache-dir',
                        help='Cache name')


    parser.add_argument('-f', '--features', type=int,
                        help='Number of features', default=24)

    parser.add_argument('--speakers', nargs='+',
                        help='List of speakers', choices=available_speakers)

    args = parser.parse_args()

    print(args)

    preprocess(join(CACHE_ROOT, args.cache_dir), args.speakers, args.features)
