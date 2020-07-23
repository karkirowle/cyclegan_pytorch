

from nnmnkwii.datasets.vcc2016 import WavFileDataSource as VCC2016Super
from nnmnkwii.datasets import FileSourceDataset
import librosa
import numpy as np

from utils import world_decompose

# Basically, you have to be paranoid about the alignment of various datasources
# Or you hstacl


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


if __name__ == '__main__':

    data_source = VCC2016DataSource("/home/boomkin/repos/Voice_Converter_CycleGAN/data", ["SF1"])
    something = FileSourceDataset(data_source)

    print(something[0].shape)
# Doesn't provide acceleration
#class MyInt(int):

