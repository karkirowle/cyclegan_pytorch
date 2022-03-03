from nnmnkwii.datasets import FileSourceDataset, MemoryCacheDataset
from nnmnkwii.preprocessing import meanstd
from uaspeech import available_speakers, UASpeechDataSource
from mcep_wrapper import MCEPWrapper
from os.path import join, splitext, isdir, basename
from utils import world_decode_spectral_envelop, world_speech_synthesis, pitch_conversion_with_logf0, compute_log_f0_cwt_norm, denormalize, inverse_cwt

import argparse
import pickle
import itertools
import numpy as np
import librosa
import soundfile as sf

DATA_ROOT = "/tudelft.net/staff-bulk/ewi/insy/SpeechLab/corpora/UASpeech_2"
CACHE_ROOT = join("preprocessed", "UASpeech")
VALIDATION_ROOT = join("validation_output")
CHECKPOINT_ROOT = join("checkpoint")
SAMPLES_ROOT = join("samples")


def samples(cache_dir, speakers, num_features):

    speakers.sort()

    input_set = MemoryCacheDataset(FileSourceDataset(
        UASpeechDataSource(DATA_ROOT, cache_dir, speakers)))
    output_set = MemoryCacheDataset(FileSourceDataset(
        UASpeechDataSource(DATA_ROOT, cache_dir, speakers, training=False)))

    for i in range(len(output_set)):
        features = output_set[i]
        filename = basename(
            output_set.dataset.collected_files[i][0])
        print(filename)
        f0 = np.ascontiguousarray(features[:, 0])
        sp = world_decode_spectral_envelop(
            np.ascontiguousarray(features[:, 1:num_features+1]), 16000)
        ap = np.ascontiguousarray(features[:, (num_features+1):])
        wav = world_speech_synthesis(
            f0, sp, ap, 16000, frame_period=5)
        # librosa.output.write_wav(join(
        #     SAMPLES_ROOT, filename), wav, 16000)
        sf.write(join(
            SAMPLES_ROOT, filename), wav, 16000)
    for i in range(len(input_set)):
        features = input_set[i]
        filename = basename(
            input_set.dataset.collected_files[i][0])
        print(filename)
        f0 = np.ascontiguousarray(features[:, 0])
        sp = world_decode_spectral_envelop(
            np.ascontiguousarray(features[:, 1:num_features+1]), 16000)
        ap = np.ascontiguousarray(features[:, (num_features+1):])
        wav = world_speech_synthesis(
            f0, sp, ap, 16000, frame_period=5)
        # librosa.output.write_wav(join(
        #     SAMPLES_ROOT, filename), wav, 16000)
        sf.write(join(
            SAMPLES_ROOT, filename), wav, 16000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=" training data.")

    parser.add_argument('--cache-dir',
                        help="dir to store cached features", default=CACHE_ROOT)
    # parser.add_argument('-w', '--write', action='store_true',
    #                     help="write samples")
    parser.add_argument('-f', '--features', type=int,
                        help='Number of features', default=24)

    parser.add_argument('--speakers', nargs='+',
                        help='List of speakers', choices=available_speakers)
    # parser.add_argument('--train', nargs='+',
    #                     help='List of speakers for training', choices=available_speakers, default=['F02'])
    # parser.add_argument('--eval', nargs='+',
    #                     help='List of speakers for training', choices=available_speakers, default=['F02'])

    args = parser.parse_args()

    print(args)

    # if args.speakers is not None:
    samples(join(CACHE_ROOT, args.cache_dir), args.speakers, args.features)
    # else:
    #     preprocess(data_root, join(default_cache, args.cache_dir),
    #                args.denoise, args.cache, args.recalc or not args.cache, args.train, args.eval)
