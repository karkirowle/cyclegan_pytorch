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
import numpy as np

DATA_ROOT = "/tudelft.net/staff-bulk/ewi/insy/SpeechLab/mlrprananta/clean_uaspeech"
CACHE_ROOT = './preprocessed/UASpeech'



def preprocess_test_data(cache_dir, speakers, num_features):

    def preprocess(wav):
        f0, _, sp, ap = world_decompose(wav, 16000)
        mcep = world_encode_spectral_envelop(sp, 16000, dim=num_features)
        features = np.hstack((f0[:, None], mcep, ap))
        return features

    speakers.sort()

    for fold in itertools.combinations(speakers, len(speakers) - 1):
        input_set = MemoryCacheDataset(FileSourceDataset(UASpeechDataSource(DATA_ROOT, cache_dir, fold, num_features=num_features, preprocess_function=preprocess)))
        output_set = MemoryCacheDataset(FileSourceDataset(UASpeechDataSource(DATA_ROOT, cache_dir, fold, training=False, num_features=num_features, preprocess_function=preprocess)))

        meanstd(input_set, [len(y) for y in input_set])
        meanstd(output_set, [len(y) for y in output_set])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Preprocess training data.")

    parser.add_argument('--cache-dir',
                        help="dir to store cached features", default=CACHE_ROOT)
    parser.add_argument('-f', '--features', type=int,
                        help='Number of features', default=24)

    parser.add_argument('--speakers', nargs='+',
                        help='List of speakers', choices=available_speakers)

    args = parser.parse_args()

    print(args)

    preprocess_test_data(join(CACHE_ROOT, args.cache_dir), args.speakers, args.features)
