from nnmnkwii.datasets.vcc2016 import WavFileDataSource as VCC2016Super
from nnmnkwii.datasets import FileSourceDataset
from nnmnkwii.preprocessing import meanstd

import librosa
import numpy as np

from torch.utils.data import Dataset
import torch
import os
import matplotlib.pyplot as plt
from utils import get_log_f0_cwt_norm, transpose_in_list, compute_log_f0_cwt_norm, logf0_statistics as f0_statistics


class F0Wrapper(Dataset):
    """
    Wrapper around nnmnkwii datsets
    """

    def __init__(self, input_file_source, output_file_source, frames=128, normalize=True, statistics=None, load=False):
        self.input_file_source = input_file_source
        self.output_file_source = output_file_source
        self.input_meanstd = None
        self.output_meanstd = None
        self.frames = frames

        if statistics:
            self.input_meanstd = statistics[0]
            self.output_meanstd = statistics[1]
        elif normalize:
            self.normalize()

    def normalize(self):
        print("Computing (log) f0 statistics")
        f0_input = [features[:, 0] for features in self.input_file_source]
        f0_output = [features[:, 0] for features in self.output_file_source]

        self.input_meanstd = f0_statistics(f0_input)
        self.output_meanstd = f0_statistics(f0_output)

        print(self.input_meanstd)
        print(self.output_meanstd)

    def __len__(self):
        assert len(self.input_file_source) == len(self.output_file_source)
        return len(self.input_file_source)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # coeffs = self.num_features + 1
        # print(
        #     'before', self.input_file_source[idx].shape, self.output_file_source[idx].shape)

        input_temp = self.input_file_source[idx][:, 0]
        output_temp = self.output_file_source[idx][:, 0]

        filename_A = self.input_file_source.dataset.collected_files[idx]
        filename_B = self.output_file_source.dataset.collected_files[idx]

        # print(input_temp.shape)

        input_mean, input_std = self.input_meanstd
        output_mean, output_std = self.output_meanstd

        # if input_temp.shape[0] < self.frames:
        #     left = (self.frames - input_temp.shape[0]) // 2
        #     right = (self.frames - input_temp.shape[0]) - left
        #     input_temp = np.pad(
        #         input_temp, (left, right), 'edge')

        # if output_temp.shape[0] < self.frames:
        #     left = (self.frames - output_temp.shape[0]) // 2
        #     right = (self.frames - output_temp.shape[0]) - left
        #     output_temp = np.pad(
        #         output_temp, (left, right), 'edge')

        if (input_temp == 0).all():
            print(os.path.basename(filename_A[0]) + " has no f0")
            return self.__getitem__(idx + 1)

        if (output_temp == 0).all():
            print(os.path.basename(filename_B[0]) + " has no f0")
            return self.__getitem__(idx + 1)

        input = compute_log_f0_cwt_norm(input_temp, input_mean, input_std)[0]
        output = compute_log_f0_cwt_norm(
            output_temp, output_mean, output_std)[0]
        # print(input.shape)
        # input = input.T
        # output = output.T

        # print('shape2', self.input_file_source[idx][:, 1:25].shape)

        # frames_A = self.input_file_source[idx].shape[0]
        # frames_B = self.output_file_source[idx].shape[0]

        # print(input.shape)

        frames_A = input.shape[0]
        frames_B = output.shape[0]

        # if frames_A < self.frames:
        #     left = (self.frames - frames_A) // 2
        #     right = (self.frames - frames_A) - left
        #     input = np.pad(
        #         input, ((left, right), (0, 0)), 'edge')
        #     frames_A = input.shape[0]

        # if frames_B < self.frames:
        #     left = (self.frames - frames_B) // 2
        #     right = (self.frames - frames_B) - left
        #     output = np.pad(
        #         output, ((left, right), (0, 0)), 'edge')
        #     frames_B = output.shape[0]

        assert frames_A >= self.frames, os.path.basename(
            filename_A[0]) + " has " + str(frames_A) + " frames"
        start_A = np.random.randint(frames_A - self.frames + 1)
        end_A = start_A + self.frames

        assert frames_B >= self.frames, os.path.basename(
            filename_B[0]) + " has " + str(frames_B) + " frames"
        start_B = np.random.randint(frames_B - self.frames + 1)
        end_B = start_B + self.frames

        # print('frames A', start_A, end_A, frames_A)
        # print('frames B', start_B, end_B, frames_B)

        input_slice = input[start_A:end_A, :]
        output_slice = output[start_B:end_B, :]
        # input_mean, input_std = self.input_meanstd
        # output_mean, output_std = self.output_meanstd

        # Second index: selecting 24 MCEP features
        # Third index: randomly samping 128 frames
        # print(input_slice.shape)

        input_tensor = torch.FloatTensor(input_slice)
        output_tensor = torch.FloatTensor(output_slice)

        #other = OtherParameters(f0_A,f0_B,bap_A,bap_B)

        return (input_tensor, output_tensor, list(filename_A), list(filename_B))
