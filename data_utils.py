

from nnmnkwii.datasets.vcc2016 import WavFileDataSource as VCC2016Super
from nnmnkwii.datasets import FileSourceDataset
from nnmnkwii.preprocessing import meanstd

import librosa
import numpy as np

from utils import world_decompose, world_encode_spectral_envelop, wav_padding

from torch.utils.data import Dataset
import torch
import os
import matplotlib.pyplot as plt
import librosa.sequence


class VCC2016DataSource(VCC2016Super):

    def __init__(self,data_root,speakers,training):
        super().__init__(data_root,speakers,training=training)

        # We check for preprocessed dir and create a subfolder based on speaker name
        self.preprocess_dir = "./preprocessed"
        self.speaker = speakers[0]
        if not os.path.exists(os.path.join(self.preprocess_dir)):
            os.mkdir(self.preprocess_dir)
        if not os.path.exists(os.path.join(self.preprocess_dir,self.speaker)):
            os.mkdir(os.path.join(self.preprocess_dir,self.speaker))

    def collect_features(self,file_path):
        """PyWorld analysis"""
        sr = 16000

        save_path = os.path.join(self.preprocess_dir, self.speaker, os.path.basename(file_path))

        if os.path.exists(save_path):
            features = np.load(save_path, allow_pickle=True)
        else:

            wav, _ = librosa.load(file_path, sr=sr, mono=True)
            wav_padded = wav_padding(wav, sr=sr, frame_period=5, multiple=4)
            f0, _, sp, ap = world_decompose(wav_padded,sr)

            mcep = world_encode_spectral_envelop(sp, sr, dim=24)


            # Extending to 2D to stack and log zeroes 1e-16. TODO: Better solution for this
            f0 = np.ma.log(f0[:,None])
            #f0[f0 == -np.inf] = 1e-16

            features = np.hstack((f0, mcep, ap))
            features.dump(save_path)

        return features


class MCEPWrapper(Dataset):
    """
    Wrapper around nnmnkwii datsets
    """
    def __init__(self,input_file_source,output_file_source, mfcc_only, num_frames=128, norm_calc=True,dtw=False):
        self.input_file_source = input_file_source
        self.output_file_source = output_file_source
        self.num_frames = num_frames
        self.mfcc_only = mfcc_only
        self.input_meanstd = None
        self.output_meanstd = None
        self.dtw = dtw

        if norm_calc:
            print("Performing speaker 1 normalization...")
            SF1_lengths = [len(y) for y in self.input_file_source]
            self.input_meanstd = meanstd(self.input_file_source, SF1_lengths)
            print("Performing speaker 2 normalization...")
            TF2_lengths = [len(y) for y in self.output_file_source]
            self.output_meanstd = meanstd(self.output_file_source, TF2_lengths)



    def __len__(self):
        assert len(self.input_file_source) == len(self.output_file_source)
        return len(self.input_file_source)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # The main problem is the misalignment of the audios on this level
        input_temp = self.input_file_source[idx][:,1:25]
        output_temp = self.output_file_source[idx][:,1:25]


        if self.dtw:
            # Transposed because it accepts transposed form only
            D, wp = librosa.sequence.dtw(input_temp.T,output_temp.T,backtrack=True)
            input_temp = input_temp[wp[::-1,0],:]
        # This snippet is responsible for sampling the frames
        frames_a = input_temp.shape[0]
        assert frames_a >= self.num_frames

        start_a = np.random.randint(frames_a - self.num_frames + 1)
        end_a = start_a + self.num_frames

        frames_b = self.output_file_source[idx].shape[0]
        assert frames_b >= self.num_frames
        start_b = np.random.randint(frames_b - self.num_frames + 1)
        end_b = start_b + self.num_frames

        # This snippet is responsible for slicing and normalisation

        input_slice = input_temp[start_a:end_a,:]
        output_slice = output_temp[start_b:end_b,:]
        input_mean, input_std = self.input_meanstd
        output_mean, output_std = self.output_meanstd
        mcep_a_normalised = (input_slice - input_mean[1:25])/input_std[1:25]
        mcep_b_normalised = (output_slice - output_mean[1:25])/output_std[1:25]

        input_tensor = torch.FloatTensor(mcep_a_normalised)
        output_tensor = torch.FloatTensor(mcep_b_normalised)

        filename_a = list(self.input_file_source.dataset.collected_files[idx])
        filename_b = list(self.output_file_source.dataset.collected_files[idx])

        return input_tensor, output_tensor, filename_a, filename_b


if __name__ == '__main__':

    data_source = VCC2016DataSource("/home/boomkin/repos/Voice_Converter_CycleGAN/data", ["SF1"])
    something = FileSourceDataset(data_source)

    print(something.collected_files[15])
    print(something[0].shape)
# Doesn't provide acceleration
#class MyInt(int):

