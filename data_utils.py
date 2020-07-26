

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
            features = np.load(save_path)
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
    def __init__(self,input_file_source,output_file_source, mfcc_only, num_frames=128, norm_calc=True):
        self.input_file_source = input_file_source
        self.output_file_source = output_file_source
        self.num_frames = num_frames
        self.mfcc_only = mfcc_only
        self.input_meanstd = None
        self.output_meanstd = None

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


        # This snippet is responsible for sampling the frames
        frames_A = self.input_file_source[idx].shape[0]
        assert frames_A >= self.num_frames

        start_A = np.random.randint(frames_A - self.num_frames + 1)
        end_A = start_A + self.num_frames


        frames_B = self.output_file_source[idx].shape[0]
        assert frames_B >= self.num_frames
        start_B = np.random.randint(frames_B - self.num_frames + 1)
        end_B = start_B + self.num_frames

        # This snippet is responsible for slicing and normalisation

        #if self.mfcc_only:
        input_slice = self.input_file_source[idx][start_A:end_A,1:25]
        output_slice = self.output_file_source[idx][start_B:end_B, 1:25]
        input_mean, input_std = self.input_meanstd
        output_mean, output_std = self.output_meanstd
        mcep_A_normalised = (input_slice - input_mean[1:25])/input_std[1:25]

        mcep_B_normalised = (output_slice - output_mean[1:25])/output_std[1:25]



        #else:
            # We return everything, but we still have normalise
        #    input_slice = self.input_file_source[idx]
        #    output_slice = self.output_file_source[idx]
        #    input_mean, input_std = self.input_meanstd
        #    output_mean, output_std = self.output_meanstd
        #    input_slice[:,1:25] = (input_slice[:,1:25] - input_mean[1:25])/input_std[1:25]
        #    output_slice[:,1:25] = (output_slice[:,1:25] - output_mean[1:25])/output_std[1:25]
        #    input_slice_normalised = input_slice
        #    output_slice_normalised = output_slice

        # Second index: selecting 24 MCEP features
        # Third index: randomly samping 128 frames
        input_tensor = torch.FloatTensor(mcep_A_normalised)
        output_tensor = torch.FloatTensor(mcep_B_normalised)

        filename_A = list(self.input_file_source.dataset.collected_files[idx])
        filename_B = list(self.output_file_source.dataset.collected_files[idx])

        #other = OtherParameters(f0_A,f0_B,bap_A,bap_B)


        return (input_tensor, output_tensor, filename_A, filename_B)


if __name__ == '__main__':

    data_source = VCC2016DataSource("/home/boomkin/repos/Voice_Converter_CycleGAN/data", ["SF1"])
    something = FileSourceDataset(data_source)

    print(something.collected_files[15])
    print(something[0].shape)
# Doesn't provide acceleration
#class MyInt(int):

