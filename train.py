
import nnmnkwii

from data_utils import VCC2016DataSource, MCEPWrapper
from nnmnkwii.datasets import FileSourceDataset

SF1_data_source = FileSourceDataset(VCC2016DataSource("/home/boomkin/repos/Voice_Converter_CycleGAN/data", ["SF1"]))
TF2_data_source = FileSourceDataset(VCC2016DataSource("/home/boomkin/repos/Voice_Converter_CycleGAN/data", ["TF2"]))

dataset = MCEPWrapper(SF1_data_source, TF2_data_source)


