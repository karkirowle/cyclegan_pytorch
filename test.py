
import nnmnkwii
from datetime import datetime

from nnmnkwii.datasets import FileSourceDataset, MemoryCacheDataset
import matplotlib.pyplot as plt
import torch
from utils import world_decode_spectral_envelop, world_speech_synthesis, pitch_conversion_with_logf0, compute_log_f0_cwt_norm, denormalize, inverse_cwt

import librosa
import os
import argparse
import pickle

# from tf_adam import Adam
import numpy as np
import soundfile as sf
import re

from uaspeech import available_speakers, UASpeechDataSource
from mcep_wrapper import MCEPWrapper

from cyclegan import CycleGAN
from f0_wrapper import F0Wrapper

DATA_ROOT = "/tudelft.net/staff-bulk/ewi/insy/SpeechLab/corpora/UASpeech_2"
TEST_ROOT = "/tudelft.net/staff-bulk/ewi/insy/SpeechLab/mlrprananta/clean_uaspeech"
FAST_ROOT = "/tudelft.net/staff-bulk/ewi/insy/SpeechLab/mlrprananta/fast_uaspeech"
CACHE_ROOT = os.path.join("preprocessed", "UASpeech")
VALIDATION_ROOT = os.path.join("validation_output")
CHECKPOINT_ROOT = os.path.join("checkpoint")


class TestRoutine:

    def __init__(self, experiment, training_cache, eval_cache, num_features, parallel, dtw, twostep, timestretch=False, sr=16000):
        self.experiment = experiment
        self.data_root = DATA_ROOT
        self.cache_dir = os.path.join(CACHE_ROOT, training_cache) or CACHE_ROOT
        self.eval_cache = os.path.join(CACHE_ROOT, eval_cache)
        self.results_dirname = "results_new"

        self.experiment_dir = os.path.join(self.results_dirname, experiment)
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        self.validation_dir = os.path.join(self.experiment_dir, "SAMPLES" 
            + ('_P' if parallel or dtw else '_NP') 
            + ('_DTW' if dtw else '') 
            + ('_2STEP' if twostep else '')
            + ('_TS' if timestretch else ''))
        if not os.path.exists(self.validation_dir):
            os.makedirs(self.validation_dir)

        self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)


        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.num_features = num_features
        self.parallel = parallel
        self.dtw = dtw
        self.twostep = twostep


        self.model_f0 = CycleGAN(num_features=10)
        self.model_mcep = CycleGAN(num_features=num_features, twostep=twostep)
        self.epoch = 0

        self.model_filename = 'checkpoint_mcep' \
            + ('_parallel' if parallel or dtw else '') \
            + ('_dtw' if dtw else '') \
            + ('_twostep' if twostep else '') \
            + '.pt'

        model_f0_path = os.path.join(
            self.checkpoint_dir, 'checkpoint_f0.pt')
        self.model_f0.load(model_f0_path)

        model_path = os.path.join(self.checkpoint_dir, self.model_filename)
        self.epoch = self.model_mcep.load(model_path, True)
        self.sr = sr

    def generate_samples(self, eval_dataset, input_statistics, output_statistics, input_f0_statistics, output_f0_statistics):
        with torch.no_grad():
            for i in range(len(eval_dataset)):
                coeffs = self.num_features + 1

                feature_A = eval_dataset[i]
                filename_A = os.path.basename(
                    eval_dataset.dataset.collected_files[i][0])

                f0_A = feature_A[:, 0]
                ap_A = feature_A[:, coeffs:]

                mean_A, std_A = input_statistics
                mean_B, std_B = output_statistics

                f0_mean_A, f0_std_A = input_f0_statistics
                f0_mean_B, f0_std_B = output_f0_statistics

                mean_mcep_A = mean_A[1:coeffs]
                std_mcep_A = std_A[1:coeffs]

                mcep_A = (feature_A[None, :, 1:coeffs] -
                          mean_mcep_A)/std_mcep_A

                log_f0_cwt_norm, uv, scales, mean, std = compute_log_f0_cwt_norm(
                    f0_A, f0_mean_A, f0_std_A)

                log_f0_cwt_norm_tensor = torch.FloatTensor(
                    log_f0_cwt_norm[None, :, :]).permute(0, 2, 1).to(self.device)

                fake_log_f0_cwt_norm = self.model_f0.infer(
                    log_f0_cwt_norm_tensor)

                fake_log_f0_cwt_norm = fake_log_f0_cwt_norm.cpu().detach().numpy()[
                    0, :, :]
                fake_log_f0_cwt = denormalize(
                    fake_log_f0_cwt_norm.T, mean, std)              # [470,10]
                fake_log_f0 = inverse_cwt(fake_log_f0_cwt, scales)  # [470,1]
                fake_log_f0 = fake_log_f0 * f0_std_B + f0_mean_B
                fake_f0 = np.squeeze(uv) * np.exp(fake_log_f0)
                fake_f0 = np.ascontiguousarray(fake_f0)

                real_A = torch.FloatTensor(
                    mcep_A).permute(0, 2, 1).to(self.device)

                fake_B = self.model_mcep.infer(real_A)

                # Conversion of A -> B
                fake_B = fake_B.cpu().detach().numpy()[0, :, :]
                fake_B = fake_B.T*std_B[1:coeffs] + mean_B[1:coeffs]
                fake_B = np.ascontiguousarray(
                    fake_B).astype(np.float64)

                sp = world_decode_spectral_envelop(fake_B, self.sr)
                ap = np.ascontiguousarray(ap_A)

                speech_fake_B = world_speech_synthesis(
                    fake_f0, sp, ap, self.sr, frame_period=5)

                filepath = os.path.join(self.validation_dir, filename_A)
                # wav = np.nan_to_num(speech_fake_B) # self.speedup_wav(filepath, np.nan_to_num(speech_fake_B))
                # wav = librosa.util.normalize(speech_fake_B)
                wav = speech_fake_B
                sf.write(filepath, wav, self.sr)


    def speedup_wav(self, dysarthric_filepath, dysarthric_wav):

        match = re.match(r'(?P<speaker>[\w]+)\_(?P<rep>[\w]+)\_(?P<file_id>\w+)\_(?P<mic>\w+)\.wav', os.path.basename(dysarthric_filepath))
        name_dict = match.groupdict()

        control_filepath = os.path.join(TEST_ROOT, "C" + name_dict['speaker'], "C" + os.path.basename(dysarthric_filepath))
        control_wav, _ = librosa.load(control_filepath, sr=self.sr, mono=True)

        wav_trimmed, _ = librosa.effects.trim(dysarthric_wav, top_db=30) 
        control_wav, _ = librosa.effects.trim(control_wav, top_db=30) 

        ratio = len(wav_trimmed)/len(control_wav)

        return librosa.effects.time_stretch(dysarthric_wav, ratio)

    def load_statistics(self, filename):
        path = os.path.join(self.cache_dir, filename + '.pickle')
        if os.path.exists(path):
            return pickle.load(open(path, 'rb'))
        else: 
            return None

    def save_statistics(self, filename, statistics):
        path = os.path.join(self.cache_dir, filename + '.pickle')
        if not os.path.exists(path):
            pickle.dump(statistics,open(path, 'wb'))

    def test(self, train_speakers=[], test_speakers=[]):
        input_dataset = MemoryCacheDataset(FileSourceDataset(
            UASpeechDataSource(TEST_ROOT, self.cache_dir, speakers=train_speakers)))
        output_dataset = MemoryCacheDataset(FileSourceDataset(
            UASpeechDataSource(TEST_ROOT, self.cache_dir, speakers=train_speakers, training=False)))
        eval_dataset = MemoryCacheDataset(FileSourceDataset(
            UASpeechDataSource(TEST_ROOT, self.eval_cache, speakers=test_speakers)))   

        statistics_filename = 'statistics_mcep_' + '_'.join(train_speakers)
        statistics_f0_filename = 'statistics_f0_' + '_'.join(train_speakers)
        training_data = MCEPWrapper(input_dataset,
                                    output_dataset,
                                    num_features=self.num_features,
                                    dtw=self.dtw,
                                    parallel=self.parallel or self.dtw,
                                    norm_statistics=self.load_statistics(statistics_filename))

        self.save_statistics(statistics_filename, (training_data.input_meanstd, training_data.output_meanstd))

        f0_data = F0Wrapper(input_dataset, output_dataset, statistics=self.load_statistics(statistics_f0_filename))

        self.save_statistics(statistics_f0_filename, (f0_data.input_meanstd, f0_data.output_meanstd))

        self.generate_samples(
            eval_dataset, training_data.input_meanstd, training_data.output_meanstd, f0_data.input_meanstd, f0_data.output_meanstd)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train CycleGAN.")

    parser.add_argument('--cache', help="name of the cache dir")
    parser.add_argument('--eval-cache', help="name of eval data cache ")
    parser.add_argument('--parallel', action='store_true', help="Enable parallel data setup")
    parser.add_argument('--dtw', action='store_true', help="Enable DTW and parallel data setup")
    parser.add_argument('--twostep', action='store_true', help="Enable two-step adversarial loss")
    parser.add_argument('--timestretch', action='store_true', help='Enable when using timestretched speech')

    parser.add_argument('--features', type=int,
                        help="Number of features", default=24)

    parser.add_argument('--training-set', nargs='+',
                        help='List of speakers for training', choices=available_speakers)
    parser.add_argument('--eval-set', nargs='+',
                        help='List of speakers for eval', choices=available_speakers)    

    args = parser.parse_args()

    print(args)

    experiment_name = '_'.join(sorted(args.training_set))

    experiment = TestRoutine(
        experiment_name, args.cache, args.eval_cache, args.features, args.parallel or args.dtw, args.dtw, args.twostep, timestretch=args.timestretch)

    # if args.load_mcep:
    #     experiment.resume(epochs=args.epochs,
    #                       train_speakers=args.training_set, test_speakers=args.eval_set)
    # else:
    experiment.test(train_speakers=args.training_set, test_speakers=args.eval_set)
