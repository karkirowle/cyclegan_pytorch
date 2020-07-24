
import nnmnkwii

from data_utils import VCC2016DataSource, MCEPWrapper
from nnmnkwii.datasets import FileSourceDataset, MemoryCacheDataset
from nnmnkwii.preprocessing import meanstd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from utils import world_decode_spectral_envelop, world_speech_synthesis, world_decode_data

from modules import Generator, Discriminator
import librosa
import os

from torch.nn.functional import l1_loss, mse_loss

import numpy as np
############## HYPERPARAMETER PART #######################################

batch_size = 1
num_epochs = 5000
num_features = 24
fs = 16000
data_root="/home/boomkin/repos/Voice_Converter_CycleGAN/data"
validation_A_dir="./validation_output/converted_A"
validation_B_dir="./validation_output/converted_B"

generator_lr = 0.0002
generator_lr_decay = generator_lr / 200000
discriminator_lr = 0.0001
discriminator_lr_decay = discriminator_lr / 200000
fake_label = 0
true_label = 1
lambda_cycle = 10
lambda_identity = 5
start_decay = 200000

############## HYPERPARAMETER PART #######################################

# We create nice dirs

if not os.path.exists(validation_A_dir):
    os.mkdir(validation_A_dir)
if not os.path.exists(validation_B_dir):
    os.mkdir(validation_B_dir)

SF1_train_data_source = MemoryCacheDataset(FileSourceDataset((VCC2016DataSource(data_root, ["SF1"],training=True))))
TF2_train_data_source = MemoryCacheDataset(FileSourceDataset((VCC2016DataSource(data_root, ["TF2"],training=True))))
SF1_test_data_source = MemoryCacheDataset(FileSourceDataset((VCC2016DataSource(data_root, ["SF1"],training=False))))
TF2_test_data_source = MemoryCacheDataset(FileSourceDataset((VCC2016DataSource(data_root, ["TF2"],training=False))))

# We need to fetch datasetwise normalisation parameters and this seems to be the best point for it
#SF1_lengths = [len(y) for y in SF1_train_data_source]
#SF1_data_meanstd = meanstd(SF1_train_data_source, SF1_lengths)
#TF2_lengths = [len(y) for y in TF2_train_data_source]
#TF2_data_meanstd = meanstd(TF2_train_data_source, TF2_lengths)
# TODO: solve the normalisation eventually
meanstd_1 = (0,1)
meanstd_2 = (0,1)

train_dataset = MCEPWrapper(SF1_train_data_source, TF2_train_data_source, SF1_data_meanstd, TF2_data_meanstd, mfcc_only=True)
test_dataset = MCEPWrapper(SF1_test_data_source, TF2_test_data_source, SF1_data_meanstd, TF2_data_meanstd, mfcc_only=False)

train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


generator_A2B = Generator(num_features).to("cuda")
generator_B2A = Generator(num_features).to("cuda")
discriminator_A = Discriminator(1).to("cuda")
discriminator_B = Discriminator(1).to("cuda")


generator_A_optimizer = torch.optim.Adam(generator_A2B.parameters(),
                                         lr=generator_lr)

discriminator_A_optimizer = torch.optim.Adam(discriminator_A.parameters(),
                                           lr=discriminator_lr)

generator_B_optimizer = torch.optim.Adam(generator_B2A.parameters(),
                                         lr=generator_lr)

discriminator_B_optimizer = torch.optim.Adam(discriminator_B.parameters(),
                                           lr=discriminator_lr)

for epoch in range(num_epochs):
    print("Epoch ", epoch)
    for i,sample in enumerate(train_dataset_loader):

        # Learning rate adjustment snippet
        num_iterations = (len(train_dataset_loader) // batch_size) * epoch + i
        if num_iterations > 10000:
            lambda_identity = 0
        if num_iterations > start_decay:
            generator_lr = max(0., generator_lr - generator_lr_decay)
            discriminator_lr = max(0., discriminator_lr - discriminator_lr_decay)
            for param_groups in generator_A_optimizer.param_groups:
                param_groups['lr'] = generator_lr
            for param_groups in generator_B_optimizer.param_groups:
                param_groups['lr'] = generator_lr
            for param_groups in discriminator_A_optimizer.param_groups:
                param_groups['lr'] = discriminator_lr
            for param_groups in discriminator_B_optimizer.param_groups:
                param_groups['lr'] = discriminator_lr

        # TODO: I wonder why nnmnkwii has this orientation for samples? Or did I just **** up somewhere?

        real_A = sample[0].permute(0, 2, 1).to("cuda")
        real_B = sample[1].permute(0, 2, 1).to("cuda")

        # Speech A -> Speech B -> Speech A
        fake_B = generator_A2B(real_A)
        cycle_A = generator_B2A(fake_B)

        # Speech B -> Speech A -> Speech B
        fake_A = generator_B2A(real_B)
        cycle_B = generator_A2B(fake_A)

        # Speech A -> Speech A (A2B), Speech B -> Speech B (B2A)
        identity_A = generator_B2A(real_A)
        identity_B = generator_A2B(real_B)

        # Spoofing - unsqueeze needed due to 2D conv
        antispoof_A = discriminator_A(fake_A.unsqueeze(1))
        antispoof_B = discriminator_B(fake_B.unsqueeze(1))

        # Loss functions
        cycle_loss = l1_loss(real_A, cycle_A) + l1_loss(real_B, cycle_B)
        identity_loss = l1_loss(identity_A, real_A) + l1_loss(identity_B, real_B)

        # When backpropagating for the generator, we want the discriminator to be cheated
        generation_loss = mse_loss(antispoof_A, torch.ones_like(antispoof_A)*true_label) + \
            mse_loss(antispoof_B, torch.ones_like(antispoof_B)*true_label)

        generator_loss = lambda_cycle * cycle_loss + lambda_identity * identity_loss + generation_loss

        generator_A_optimizer.zero_grad()
        generator_B_optimizer.zero_grad()
        generator_loss.backward()
        generator_A_optimizer.step()
        generator_B_optimizer.step()

        d_real_A = discriminator_A(real_A.unsqueeze(1))
        d_fake_A = discriminator_A(generator_B2A(real_B).unsqueeze(1))

        d_real_B = discriminator_B(real_B.unsqueeze(1))
        d_fake_B = discriminator_B(generator_A2B(real_A).unsqueeze(1))

        # When backpropagating for the discriminator, we want the discriminator to be powerful
        discriminator_loss = mse_loss(d_real_A, torch.ones_like(d_real_A)*true_label) + \
            mse_loss(d_fake_A, torch.ones_like(d_fake_A)*fake_label) + \
            mse_loss(d_real_B, torch.ones_like(d_real_B)*true_label) + \
            mse_loss(d_fake_B, torch.ones_like(d_fake_B)*fake_label)

        discriminator_A_optimizer.zero_grad()
        discriminator_B_optimizer.zero_grad()
        discriminator_loss.backward()
        discriminator_A_optimizer.step()
        discriminator_B_optimizer.step()

        if (i % 50) == 0:
            print("Iteration ", num_iterations,
                  " Generator loss: ", generator_loss.item(),
                  " Discriminator loss", discriminator_loss.item())

    if (epoch % 100) == 0:

        # Model save
        with torch.no_grad():
            for i,sample in enumerate(test_dataset_loader):

                real_A_full = sample[0].permute(0, 2, 1)

                real_B_full = sample[1].permute(0, 2, 1)
                real_A = real_A_full[:,1:25,:].to("cuda")
                real_B = real_B_full[:,1:25,:].to("cuda")

                fake_B = generator_A2B(real_A)
                fake_A = generator_B2A(real_B)

                # Conversion of A -> B

                fake_B = fake_B.cpu().detach().numpy()[0,:,:]
                fake_B = fake_B*meanstd_2[1] + meanstd_2[0]
                fake_B = np.float64(np.ascontiguousarray(fake_B.T))

                # Separation
                f0 = np.ascontiguousarray(real_A_full[0,0,:].T.cpu().detach().numpy()).astype(np.float64)

                sp = world_decode_spectral_envelop(fake_B, fs)
                ap = np.ascontiguousarray(real_A_full[0,25:,:].T.cpu().detach().numpy()).astype(np.float64)

                speech_fake_B = world_speech_synthesis(f0, sp, ap, fs, frame_period=5)

                filename_A = os.path.basename(sample[2][0])

                librosa.output.write_wav(os.path.join(validation_A_dir, filename_A), speech_fake_B, fs)

                # Conversion of B -> A

                fake_A = fake_A.cpu().detach().numpy()[0,:,:]
                fake_A = fake_A*meanstd_1[1] + meanstd_1[0]
                fake_A = np.float64(np.ascontiguousarray(fake_A.T))

                # Separation
                f0 = np.ascontiguousarray(real_B_full[0,0,:].T.cpu().detach().numpy()).astype(np.float64)

                sp = world_decode_spectral_envelop(fake_A, fs)
                ap = np.ascontiguousarray(real_B_full[0,25:,:].T.cpu().detach().numpy()).astype(np.float64)

                speech_fake_A = world_speech_synthesis(f0, sp, ap, fs, frame_period=5)

                filename_B = os.path.basename(sample[3][0])

                librosa.output.write_wav(os.path.join(validation_B_dir, filename_B), speech_fake_A, fs)