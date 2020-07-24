

from nnmnkwii.datasets.vcc2016 import WavFileDataSource as VCC2016Super
from nnmnkwii.datasets import FileSourceDataset
import librosa
import numpy as np

from utils import world_decompose

from torch.utils.data import Dataset
import torch

# Basically, you have to be paranoid about the alignment of various datasources
# Or you hstacl

# MemoryCache


class VCC2016DataSource(VCC2016Super):

    def collect_features(self,file_path):
        """PyWorld analysis"""
        sr = 16000
        wav, _ = librosa.load(file_path, sr=sr, mono=True)

        f0, _, sp, ap = world_decompose(wav,sr)

        # Extending to 2D to stack
        f0 = f0[:,None]
        features = np.hstack((f0, sp, ap))

        return features


class MCEPWrapper(Dataset):
    """
    Wrapper around nnmnkwii datsets
    """
    def __init__(self,input_file_source,output_file_source, num_frames=128):
        self.input_file_source = input_file_source
        self.output_file_source = output_file_source
        self.num_frames = num_frames

    def __len__(self):
        return len(self.speech)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # This snippet is responsible for sampling the frames
        frames_A = self.input_file_source[idx,:,:].shape[2]
        assert frames_A >= self.num_frames

        start_A = np.random.randint(frames_A - self.num_frames + 1)
        end_A = start_A + self.num_frames


        frames_B = self.output_file_source[idx,:,:].shape[2]
        assert frames_B >= self.num_frames

        start_B = np.random.randint(frames_B - self.num_frames + 1)
        end_B = start_B + self.num_frames

        # Second index: selecting 24 MCEP features
        # Third index: randomly samping 128 frames
        input_tensor = torch.FloatTensor(self.input_file_source[idx,1:25,start_A:end_A])
        output_tensor = torch.FloatTensor(self.output_file_source[idx,1:25,start_B:end_B])
        print("check", input_tensor.shape, output_tensor.shape)
        print("check", type(input_tensor), type(output_tensor))

        return (input_tensor,output_tensor)

if __name__ == '__main__':

    data_source = VCC2016DataSource("/home/boomkin/repos/Voice_Converter_CycleGAN/data", ["SF1"])
    something = FileSourceDataset(data_source)

    print(something[0].shape)
# Doesn't provide acceleration
#class MyInt(int):

