
from data_utils import VCC2016DataSource, MCEPWrapper
from nnmnkwii.datasets import FileSourceDataset, MemoryCacheDataset
import torch

from modules import Generator
from utils import *
import numpy as np
import argparse

def synth(file_path, domain_A,data_root,output_dir):
    sr = 16000

    model = Generator(24)

    if domain_A:
        generator_A2B = torch.load("checkpoint/generator_A2B.pt")
        model.load_state_dict(generator_A2B)
    else:
        generator_B2A = torch.load("checkpoint/generator_B2A.pt")
        model.load_state_dict(generator_B2A)

    filename_B = os.path.basename(file_path)
    SF1_train_data_source = MemoryCacheDataset(FileSourceDataset((VCC2016DataSource(data_root, ["SF1"],training=True))))
    TF2_train_data_source = MemoryCacheDataset(FileSourceDataset((VCC2016DataSource(data_root, ["TF2"],training=True))))
    SF1_test_data_source = MemoryCacheDataset(FileSourceDataset((VCC2016DataSource(data_root, ["SF1"],training=False))))
    TF2_test_data_source = MemoryCacheDataset(FileSourceDataset((VCC2016DataSource(data_root, ["TF2"],training=False))))

    train_dataset = MCEPWrapper(SF1_train_data_source, TF2_train_data_source, mfcc_only=True)
    test_dataset = MCEPWrapper(SF1_test_data_source, TF2_test_data_source, mfcc_only=False, norm_calc=False)
    test_dataset.input_meanstd = train_dataset.input_meanstd
    test_dataset.output_meanstd = train_dataset.output_meanstd

    wav, _ = librosa.load(file_path, sr=sr, mono=True)
    wav_padded = wav_padding(wav, sr=sr, frame_period=5, multiple=4)
    f0, _, sp, ap = world_decompose(wav_padded, sr)

    mcep = world_encode_spectral_envelop(sp,sr)

    # Normalising MCEPs
    mean_A, std_A = train_dataset.input_meanstd
    mean_B, std_B = train_dataset.output_meanstd

    mean_f0_A = mean_A[0]
    mean_f0_B = mean_B[0]
    std_f0_A = std_A[0]
    std_f0_B = std_B[0]
    mean_mcep_A = mean_A[1:25]
    mean_mcep_B = mean_B[1:25]
    std_mcep_A = std_A[1:25]
    std_mcep_B = std_B[1:25]

    if domain_A:
        normalised_mcep_source = torch.Tensor((mcep - mean_mcep_A)/std_mcep_A)
    else:
        normalised_mcep_source = torch.Tensor((mcep - mean_mcep_B)/std_mcep_B)

    normalised_mcep_source = normalised_mcep_source[None,:,:]
    normalised_mcep_source = normalised_mcep_source.permute(0,2,1)

    normalised_mcep_target = model(normalised_mcep_source)
    normalised_mcep_target = normalised_mcep_target.permute(0,2,1).cpu().detach().numpy()[0,:,:]

    if domain_A:
        mcep_target = normalised_mcep_target * std_mcep_B + mean_mcep_B
    else:
        mcep_target = normalised_mcep_target * std_mcep_B + mean_mcep_B

    mcep_target = np.ascontiguousarray(mcep_target)
    # Because here we directly decompose from signal, the nonmasked array implementation is used
    f0_target = pitch_conversion(f0,
                                     mean_f0_A,
                                     std_f0_A,
                                     mean_f0_B,
                                     std_f0_B)

    sp_target = world_decode_spectral_envelop(mcep_target, sr)
    ap_target = np.ascontiguousarray(ap)

    speech_fake_A = world_speech_synthesis(f0_target, sp_target, ap_target, sr, frame_period=5)

    librosa.output.write_wav(os.path.join(output_dir, filename_B), speech_fake_A, sr)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Convert CycleGAN utteran e')

    file_path = "/home/boomkin/repos/Voice_Converter_CycleGAN/data/evaluation_all/SF1/200001.wav"
    data_root="/home/boomkin/repos/Voice_Converter_CycleGAN/data"
    output_dir = "output_dir"

    parser.add_argument('--file_path', type = str, help = 'Path of speech file to convert.', default = file_path)
    parser.add_argument('--output_dir', type = str, help = 'Output directory for converted voice.', default = output_dir)
    parser.add_argument('--data_root', type = str, help = 'VCC 2016 dataroot', default = data_root)
    parser.add_argument('--domain_A', action='store_true' , help = 'Check if converting from domain A', default = True)

    argv = parser.parse_args()

    file_path = argv.file_path
    data_root = argv.data_root
    output_dir = argv.output_dir
    domain_A = argv.domain_A

    synth(file_path,domain_A,data_root,output_dir)