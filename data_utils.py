

from nnmnkwii.datasets.vcc2016 import WavFileDataSource as VCC2016Super
from nnmnkwii.datasets import FileSourceDataset
import librosa
import numpy as np

from utils import world_decompose, world_encode_spectral_envelop, wav_padding

from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
# Basically, you have to be paranoid about the alignment of various datasources
# Or you hstacl

# MemoryCache


class VCC2016DataSource(VCC2016Super):

    def collect_features(self,file_path):
        """PyWorld analysis"""
        sr = 16000
        wav, _ = librosa.load(file_path, sr=sr, mono=True)
        wav_padded = wav_padding(wav, sr=sr, frame_period=5, multiple=4)
        f0, _, sp, ap = world_decompose(wav_padded,sr)

        mcep = world_encode_spectral_envelop(sp, sr, dim=24)

        # Extending to 2D to stack
        f0 = f0[:,None]

        features = np.hstack((f0, mcep, ap))

        return features, file_path


class MCEPWrapper(Dataset):
    """
    Wrapper around nnmnkwii datsets
    """
    def __init__(self,input_file_source,output_file_source, input_meanstd, output_meanstd, mfcc_only, num_frames=128):
        self.input_file_source = input_file_source
        self.output_file_source = output_file_source
        self.input_meanstd = input_meanstd
        self.output_meanstd = output_meanstd
        self.num_frames = num_frames
        self.mfcc_only = mfcc_only

    def __len__(self):
        assert len(self.input_file_source) == len(self.output_file_source)
        return len(self.input_file_source)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        # This snippet is responsible for sampling the frames

        frames_A = self.input_file_source[idx][0].shape[0]
        assert frames_A >= self.num_frames

        start_A = np.random.randint(frames_A - self.num_frames + 1)
        end_A = start_A + self.num_frames


        frames_B = self.output_file_source[idx][0].shape[0]
        assert frames_B >= self.num_frames
        start_B = np.random.randint(frames_B - self.num_frames + 1)
        end_B = start_B + self.num_frames

        # This snippet is responsible for slicing and normalisation

        if self.mfcc_only:
            input_slice = self.input_file_source[idx][0][start_A:end_A,1:25]
            output_slice = self.output_file_source[idx][0][start_B:end_B, 1:25]
            input_mean, input_std = self.input_meanstd
            output_mean, output_std = self.output_meanstd
            input_slice_normalised = (input_slice - input_mean)/input_std
            output_slice_normalised = (output_slice - output_mean)/output_std
        else:
            # We return everything, but we still have normalise
            input_slice = self.input_file_source[idx][0]
            output_slice = self.output_file_source[idx][0]
            input_mean, input_std = self.input_meanstd
            output_mean, output_std = self.output_meanstd
            input_slice[:,1:25] = (input_slice[:,1:25] - input_mean)/input_std
            output_slice[:,1:25] = (output_slice[:,1:25] - output_mean)/output_std
            input_slice_normalised = input_slice
            output_slice_normalised = output_slice

        # Second index: selecting 24 MCEP features
        # Third index: randomly samping 128 frames
        input_tensor = torch.FloatTensor(input_slice_normalised)
        output_tensor = torch.FloatTensor(output_slice_normalised)
        filename_A = self.input_file_source[idx][1]
        filename_B = self.output_file_source[idx][1]

        return (input_tensor, output_tensor, filename_A, filename_B)

if __name__ == '__main__':

    data_source = VCC2016DataSource("/home/boomkin/repos/Voice_Converter_CycleGAN/data", ["SF1"])
    something = FileSourceDataset(data_source)

    print(something[0].shape)
# Doesn't provide acceleration
#class MyInt(int):

