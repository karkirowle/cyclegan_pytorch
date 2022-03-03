from nnmnkwii.datasets import FileSourceDataset, MemoryCacheDataset
from nnmnkwii.preprocessing import meanstd
from uaspeech import available_speakers, UASpeechDataSource
from mcep_wrapper import MCEPWrapper
from os.path import join, splitext, isdir, basename
from os import listdir, mkdir
from utils import wav_padding

import argparse
import pickle
import itertools
import librosa
import numpy as np
import soundfile as sf
import re

DATA_ROOT = "/tudelft.net/staff-bulk/ewi/insy/SpeechLab/corpora/UASpeech_2"
DATA_NEW = "/tudelft.net/staff-bulk/ewi/insy/SpeechLab/mlrprananta/clean_uaspeech"
SAMPLING_RATE = 16000

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

def get_files(speaker_dir):
    files = [join(speaker_dir, f) for f in listdir(speaker_dir)]
    files = list(filter(lambda x: wav_filter(basename(x)), files))
    return sorted(files)

def create_wavs(speakers):

    for speaker in speakers:
        speaker_dir = join(DATA_ROOT, speaker)
        control_speaker_dir = join(DATA_ROOT, "control", "C" + speaker)

        new_speaker_dir = join(DATA_NEW, speaker)
        new_control_speaker_dir = join(DATA_NEW, "C" + speaker)

        if not isdir(new_speaker_dir):
            mkdir(new_speaker_dir)

        if not isdir(new_control_speaker_dir):
            mkdir(new_control_speaker_dir)

        files = get_files(speaker_dir)
        control_files = get_files(control_speaker_dir)

        for file in files:
            wav = clean_wav(file)
            sf.write(join(new_speaker_dir, basename(file)), wav, SAMPLING_RATE)     

        for file in control_files:
            wav = clean_wav(file)
            sf.write(join(new_control_speaker_dir, basename(file)), wav, SAMPLING_RATE)     


def clean_wav(file_path):
    wav, _ = librosa.load(file_path, sr=SAMPLING_RATE, mono=True)
    # Remove activation and deactivation clicks
    wav = wav[int(SAMPLING_RATE*0.2):(len(wav)-int(SAMPLING_RATE*0.2))]
    # Trim leading and trailing silences
    wav, _ = librosa.effects.trim(wav, top_db=30)
    # Double the length, invert second half
    # wav = np.append(wav, wav[::-1])
    # Pad if length is less than 1 second
    if len(wav) < SAMPLING_RATE:
        left = (SAMPLING_RATE - len(wav)) // 2
        right = (SAMPLING_RATE - len(wav)) - left
        wav = np.pad(wav, (left, right), "edge")
    return wav_padding(
        wav, sr=SAMPLING_RATE, frame_period=5, multiple=4)


if __name__ == '__main__':
    data_root = "/tudelft.net/staff-bulk/ewi/insy/SpeechLab/corpora/UASpeech_2"
    default_cache = './preprocessed/UASpeech'

    parser = argparse.ArgumentParser(
        description="Preprocess training data.")

    parser.add_argument('--speakers', nargs='+',
                        help='List of speakers', choices=available_speakers)
    # parser.add_argument('--train', nargs='+',
    #                     help='List of speakers for training', choices=available_speakers, default=['F02'])
    # parser.add_argument('--eval', nargs='+',
    #                     help='List of speakers for training', choices=available_speakers, default=['F02'])

    args = parser.parse_args()

    print(args)

    # if args.speakers is not None:
    create_wavs(speakers=args.speakers)
    # else:
    #     preprocess(data_root, join(default_cache, args.cache_dir),
    #                args.denoise, args.cache, args.recalc or not args.cache, args.train, args.eval)
