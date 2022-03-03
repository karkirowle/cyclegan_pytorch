
from datetime import datetime

from nnmnkwii.datasets import FileSourceDataset, MemoryCacheDataset
import matplotlib.pyplot as plt
import torch
from utils import world_decode_spectral_envelop, world_speech_synthesis, compute_log_f0_cwt_norm, denormalize, inverse_cwt

import os
import argparse
import pickle

import numpy as np
import soundfile as sf

from uaspeech import available_speakers, UASpeechDataSource
from cyclegan import CycleGAN
from mcep_wrapper import MCEPWrapper
from f0_wrapper import F0Wrapper

DATA_ROOT = "/tudelft.net/staff-bulk/ewi/insy/SpeechLab/corpora/UASpeech_2"
UASPEECH_ROOT = "/tudelft.net/staff-bulk/ewi/insy/SpeechLab/mlrprananta/clean_uaspeech"
CACHE_ROOT = os.path.join("preprocessed", "UASpeech")


class UASpeechTraining:

    def __init__(self, experiment, cache_name, num_features, dtw, parallel, twostep, load_f0=False, load_mcep=False, sr=16000):
        self.experiment = experiment
        self.data_root = UASPEECH_ROOT
        self.cache_dir = os.path.join(CACHE_ROOT, cache_name) or CACHE_ROOT
        self.results_dirname = "results_new"
        
        self.experiment_dir = os.path.join(self.results_dirname, experiment)
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        self.validation_dir = os.path.join(self.experiment_dir, "SAMPLES" 
            + ('_P' if parallel or dtw else '_NP') 
            + ('_DTW' if dtw else '') 
            + ('_2STEP' if twostep else ''))
        if not os.path.exists(self.validation_dir):
            os.makedirs(self.validation_dir)

        self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.num_features = num_features
        self.dtw = dtw
        self.parallel = parallel
        self.twostep = twostep

        self.model_f0 = CycleGAN(num_features=10)
        self.model_mcep = CycleGAN(num_features=num_features, twostep=twostep)
        self.epoch = 0

        if load_f0:
            model_f0_path = os.path.join(
                self.checkpoint_dir, 'checkpoint_f0.pt')
            self.model_f0.load(model_f0_path)

        self.f0_loaded = load_f0

        self.model_filename = 'checkpoint_mcep' \
                + ('_parallel' if parallel or dtw else '') \
                + ('_dtw' if dtw else '') \
                + ('_twostep' if twostep else '') \
                + '.pt'

        if load_mcep:  
            model_path = os.path.join(self.checkpoint_dir, self.model_filename)
            self.epoch = self.model_mcep.load(model_path, True)

        self.mcep_loaded = load_mcep

        self.sr = sr

    def save_model_f0(self, epoch):
        self.model_f0.save(epoch, os.path.join(self.checkpoint_dir, 'checkpoint_f0.pt'))

    def save_model_mcep(self, epoch):
        self.model_mcep.save(epoch, os.path.join(self.checkpoint_dir, self.model_filename))

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

                sf.write(os.path.join(
                    self.validation_dir, filename_A), np.nan_to_num(speech_fake_B), self.sr)

    def train(self, save_interval=50, epochs=1000, train_speakers=[], test_speakers=[]):
        input_dataset = MemoryCacheDataset(FileSourceDataset(
            UASpeechDataSource(self.data_root, self.cache_dir, speakers=train_speakers)))  
        output_dataset = MemoryCacheDataset(FileSourceDataset(
            UASpeechDataSource(self.data_root, self.cache_dir, speakers=train_speakers, training=False)))  
        eval_dataset = MemoryCacheDataset(FileSourceDataset(
            UASpeechDataSource(self.data_root, self.cache_dir, speakers=test_speakers)))   

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

        if not self.f0_loaded:
            print("Train F0")

            for epoch in self.model_f0.train(f0_data, epochs//10):
                if epoch % save_interval == 0:
                    self.save_model_f0(epoch)
            self.save_model_f0(epoch)

        if self.epoch == 0:
            print("Train MCEP")
            print("Generate initial samples")
            self.generate_samples(
                eval_dataset, training_data.input_meanstd, training_data.output_meanstd, f0_data.input_meanstd, f0_data.output_meanstd)
        else:
            print("Resume training MCEP")

        for epoch in self.model_mcep.train(training_data, epochs, self.epoch):
            if epoch % save_interval == 0:
                self.save_model_mcep(epoch)
                self.generate_samples(
                    eval_dataset, training_data.input_meanstd, training_data.output_meanstd, f0_data.input_meanstd, f0_data.output_meanstd)

        self.save_model_mcep(epoch)
        self.generate_samples(
            eval_dataset, training_data.input_meanstd, training_data.output_meanstd, f0_data.input_meanstd, f0_data.output_meanstd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train CycleGAN.")

    parser.add_argument('--cache-name', help="name of the cache dir")
    parser.add_argument('--parallel', action='store_true', help="Enable parallel data setup")
    parser.add_argument('--dtw', action='store_true', help="Enable DTW and parallel data setup")
    parser.add_argument('--twostep', action='store_true', help="Enable two-step adversarial loss")

    parser.add_argument('--load-f0', action='store_true', help="Load f0 model")
    parser.add_argument('--load-mcep', action='store_true',
                        help="Load MCEP model")

    parser.add_argument('--epochs', type=int,
                        help="Number of epochs", default=1000)
    parser.add_argument('--features', type=int,
                        help="Number of features", default=24)

    parser.add_argument('--training-set', nargs='+',
                        help='List of speakers for training', choices=available_speakers)
    parser.add_argument('--eval-set', nargs='+',
                        help='List of speakers for eval', choices=available_speakers)

    args = parser.parse_args()

    print(args)

    experiment_name = '_'.join(sorted(args.training_set))

    experiment = UASpeechTraining(
        experiment_name, args.cache_name, args.features, args.dtw, args.parallel, args.twostep, args.load_f0, args.load_mcep)

    experiment.train(epochs=args.epochs,
                     train_speakers=args.training_set, test_speakers=args.eval_set)
