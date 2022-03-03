from nnmnkwii.datasets import FileSourceDataset, MemoryCacheDataset
from nnmnkwii.preprocessing import meanstd
from uaspeech import available_speakers, UASpeechDataSource
from mcep_wrapper import MCEPWrapper
from os.path import join, splitext, isdir, basename
from utils import wav_padding

import argparse
import pickle
import itertools


def cache_features(data_source):
    print("Caching features")
    for _ in data_source:
        pass


def preprocess(data_root, cache_dir, denoise, cache, load, train_speakers, test_speakers):
    dysarthric_data_source = MemoryCacheDataset(FileSourceDataset(
        UASpeechDataSource(data_root, speakers=train_speakers, cache_dir=cache_dir, denoise=denoise, cache=cache, load=load, num_features=40)))
    control_data_source = MemoryCacheDataset(FileSourceDataset(
        UASpeechDataSource(data_root, speakers=train_speakers, training=False, cache_dir=cache_dir, denoise=denoise, cache=cache, load=load, num_features=40)))

    test_dysarthric_data_source = MemoryCacheDataset(FileSourceDataset(
        UASpeechDataSource(data_root, speakers=test_speakers, validating=True, cache_dir=cache_dir, denoise=denoise, cache=cache, load=load, num_features=40)))
    test_control_data_source = MemoryCacheDataset(FileSourceDataset(
        UASpeechDataSource(data_root, speakers=test_speakers, training=False, validating=True, cache_dir=cache_dir, denoise=denoise, cache=cache, load=load, num_features=40)))

    print("Compute norm statistics")
    norm_statistics = (meanstd(dysarthric_data_source, [
        len(y) for y in dysarthric_data_source]),
        meanstd(control_data_source, [
            len(y) for y in control_data_source]))

    pickle.dump(norm_statistics, open(
        join(cache_dir, "norm_statistics"), 'wb'))
    # cache_features(dysarthric_data_source)
    # cache_features(control_data_source)
    cache_features(test_dysarthric_data_source)
    cache_features(test_control_data_source)


def preprocess_group(data_root, cache_dir, denoise, cache, load, speakers, num_features, plot):

    speakers.sort()

    for fold in itertools.combinations(speakers, len(speakers) - 1):
        input_set = MemoryCacheDataset(FileSourceDataset(
            UASpeechDataSource(data_root, cache_dir, fold, denoise=denoise, cache=cache, load=load, num_features=num_features, plot=plot)))
        output_set = MemoryCacheDataset(FileSourceDataset(
            UASpeechDataSource(data_root, cache_dir, fold, training=False, denoise=denoise, cache=cache, load=load, num_features=num_features, plot=plot)))
        dump_name = '_'.join(fold)
        print("Compute norm statistics", dump_name)
        norm_statistics = (meanstd(input_set, [len(y) for y in input_set]),
                           meanstd(output_set, [len(y) for y in output_set]))

        pickle.dump(norm_statistics, open(
            join(cache_dir, "norm_statistics_" + dump_name), 'wb'))


if __name__ == '__main__':
    data_root = "/tudelft.net/staff-bulk/ewi/insy/SpeechLab/corpora/UASpeech_2"
    default_cache = './preprocessed/UASpeech'

    parser = argparse.ArgumentParser(
        description="Preprocess training data.")

    parser.add_argument('--cache-dir',
                        help="dir to store cached features", default=default_cache)
    parser.add_argument('-c', '--cache', action='store_true',
                        help="cache features")
    parser.add_argument('-r', '--recalc', action='store_false',
                        help="compute features without loading cache")
    parser.add_argument('-d', '--denoise', action='store_true',
                        help="denoise wav")
    parser.add_argument('-p', '--plot', action='store_true',
                        help="plot wavs and spectrograms")
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
    preprocess_group(data_root, join(default_cache, args.cache_dir),
                     args.denoise, args.cache, args.recalc, args.speakers, args.features, args.plot)
    # else:
    #     preprocess(data_root, join(default_cache, args.cache_dir),
    #                args.denoise, args.cache, args.recalc or not args.cache, args.train, args.eval)
