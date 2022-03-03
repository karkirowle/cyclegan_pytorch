from nnmnkwii.preprocessing import meanstd

import librosa
import numpy as np

from torch.utils.data import Dataset
import torch
import os
import matplotlib.pyplot as plt
from utils import wp_padding, world_decode_spectral_envelop


def plot_dtw(seq1, seq2, D, wp):
    fig, ax = plt.subplots(nrows=2, sharex=True)
    img = librosa.display.specshow(D, x_axis='frames', y_axis='frames',
                                   ax=ax[0])
    ax[0].set(title='DTW cost', xlabel='Noisy sequence', ylabel='Target')
    ax[0].plot(wp[:, 1], wp[:, 0], label='Optimal path', color='y')
    ax[0].legend()
    fig.colorbar(img, ax=ax[0])
    ax[1].plot(D[-1, :] / wp.shape[0])
    ax[1].set(xlim=[0, Y.shape[1]], ylim=[0, 2],
              title='Matching cost function')


def plot_mceps(mcep1, mcep2, filename, sr=16000):
    fig, axs = plt.subplots(2)
    sp = world_decode_spectral_envelop(mcep1, sr)
    axs[0].imshow(np.log10(sp))
    sp = world_decode_spectral_envelop(mcep2, sr)
    axs[1].imshow(np.log10(sp))
    plt.savefig(os.path.join("figures", filename +
                             "_spectrogram_" + ".png"))
    plt.close()


class MCEPWrapper(Dataset):
    """
    Wrapper around nnmnkwii datsets
    """

    def __init__(self, input_file_source, output_file_source, num_features=24, num_frames=128, norm_calc=True, parallel=False, dtw=False, norm_statistics=None):
        self.input_file_source = input_file_source
        self.output_file_source = output_file_source
        self.num_features = num_features
        self.num_frames = num_frames
        self.input_meanstd = None
        self.output_meanstd = None
        self.parallel = parallel or dtw
        self.dtw = dtw
        print('parallel={parallel}, dtw={dtw}'.format(parallel=parallel, dtw=dtw))
        if norm_statistics:
            self.input_meanstd = norm_statistics[0]
            self.output_meanstd = norm_statistics[1]
        elif norm_calc:
            print("Performing input normalization...")
            self.input_meanstd = meanstd(self.input_file_source, [
                                         len(y) for y in self.input_file_source])
            print("Performing output normalization...")
            self.output_meanstd = meanstd(self.output_file_source, [
                                          len(y) for y in self.output_file_source])

    def __len__(self):
        # assert len(self.input_file_source) == len(self.output_file_source)
        return min(len(self.input_file_source), len(self.output_file_source)) 

    def get_non_parallel(self, index):
        dataset_A = self.input_file_source
        dataset_B = self.output_file_source
        frames = self.num_frames
        features = self.num_features + 1

        index = self.__len__() - 1 if index > self.__len__() else index 

        num_samples = min(len(dataset_A), len(dataset_B))

        train_data_A_idx = np.arange(len(dataset_A))
        train_data_B_idx = np.arange(len(dataset_B))

        np.random.shuffle(train_data_A_idx)
        np.random.shuffle(train_data_B_idx)

        train_data_A_idx_subset = train_data_A_idx[:num_samples]
        train_data_B_idx_subset = train_data_B_idx[:num_samples]

        index_A = train_data_A_idx_subset[index]
        index_B = train_data_B_idx_subset[index]

        data_A = dataset_A[index_A][:, 1:features]
        frames_A = data_A.shape[0]
        assert frames_A >= frames
        start_A = np.random.randint(frames_A - frames + 1)
        end_A = start_A + frames

        data_B = dataset_B[index_B][:, 1:features]
        frames_B = data_B.shape[0]
        assert frames_B >= frames
        start_B = np.random.randint(frames_B - frames + 1)
        end_B = start_B + frames

        input_mean, input_std = self.input_meanstd
        output_mean, output_std = self.output_meanstd

        data_A_normalised = (
            data_A[start_A:end_A, :] - input_mean[1:features])/input_std[1:features]
        data_B_normalised = (
            data_B[start_B:end_B, :] - output_mean[1:features])/output_std[1:features]

        return (torch.FloatTensor(data_A_normalised), torch.FloatTensor(data_B_normalised))       

    def get_parallel(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        coeffs = self.num_features + 1
        # print(
        #     'before', self.input_file_source[idx].shape, self.output_file_source[idx].shape)

        input_temp = self.input_file_source[idx][:, 1:coeffs]
        output_temp = self.output_file_source[idx][:, 1:coeffs]

        # print('shape2', self.input_file_source[idx][:, 1:25].shape)
        if self.dtw:
            # Transposed because it accepts transposed form only
            D, wp = librosa.sequence.dtw(
                input_temp.T, output_temp.T, backtrack=True)
            # wp = wp_padding(wp, multiple=4)
            input_temp = input_temp[wp[::-1, 0], :]
            output_temp = output_temp[wp[::-1, 1], :]
            # print('after', input_temp.shape, output_temp.shape)
            # This snippet is responsible for sampling the frames
        frames_A = input_temp.shape[0]
        frames_B = output_temp.shape[0]

        # if frames_A < self.num_frames:
        #     left = (self.num_frames - frames_A) // 2
        #     right = (self.num_frames - frames_A) - left
        #     input_temp = np.pad(
        #         input_temp, ((left, right), (0, 0)), 'edge')
        #     frames_A = input_temp.shape[0]

        # if frames_B < self.num_frames:
        #     left = (self.num_frames - frames_B) // 2
        #     right = (self.num_frames - frames_B) - left
        #     output_temp = np.pad(
        #         output_temp, ((left, right), (0, 0)), 'edge')
        #     frames_B = output_temp.shape[0]

        assert frames_A >= self.num_frames
        start_A = np.random.randint(frames_A - self.num_frames + 1)
        end_A = start_A + self.num_frames

        assert frames_B >= self.num_frames
        start_B = np.random.randint(frames_B - self.num_frames + 1)
        end_B = start_B + self.num_frames

        # print('frames A', start_A, end_A, frames_A)
        # print('frames B', start_B, end_B, frames_B)


        if not self.dtw:
            input_slice = input_temp[start_A:end_A, :]
            output_slice = output_temp[start_B:end_B, :]
        else:
            input_slice = input_temp[start_A:end_A, :]
            output_slice = output_temp[start_A:end_A, :]
            # input_mean, input_std = self.input_meanstd
            # output_mean, output_std = self.output_meanstd


        input_mean, input_std = self.input_meanstd
        output_mean, output_std = self.output_meanstd

        mcep_A_normalised = (
            input_slice - input_mean[1:coeffs])/input_std[1:coeffs]
        mcep_B_normalised = (
            output_slice - output_mean[1:coeffs])/output_std[1:coeffs]

        # Second index: selecting 24 MCEP features
        # Third index: randomly samping 128 frames
        input_tensor = torch.FloatTensor(mcep_A_normalised)
        output_tensor = torch.FloatTensor(mcep_B_normalised)

        filename_A = list(self.input_file_source.dataset.collected_files[idx])
        filename_B = list(self.output_file_source.dataset.collected_files[idx])

        #other = OtherParameters(f0_A,f0_B,bap_A,bap_B)

        return (input_tensor, output_tensor, filename_A, filename_B)



    def __getitem__(self, idx):
        if self.parallel:
            return self.get_parallel(idx)
        else:
            return self.get_non_parallel(idx)