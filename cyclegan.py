
import nnmnkwii
from datetime import datetime
import time

from torch.utils.tensorboard import SummaryWriter
from nnmnkwii.datasets import FileSourceDataset, MemoryCacheDataset
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, ConcatDataset
from utils import world_decode_spectral_envelop, world_speech_synthesis, pitch_conversion_with_logf0, world_decompose

from modules import Generator, Discriminator
import librosa
import os
import itertools
import argparse
import pickle

from torch.nn.functional import l1_loss, mse_loss
# from tf_adam import Adam
import numpy as np
import noisereduce as nr

from uaspeech import available_speakers, UASpeechDataSource
from mcep_wrapper import MCEPWrapper
from utils import wp_padding


class CycleGAN:
    def __init__(self, batch_size=1, num_features=24, fs=16000, twostep=False):
        self.batch_size = batch_size
        self.num_features = num_features
        self.fs = fs
        self.twostep = twostep

        # Generator learning rate and decay
        self.generator_lr = 0.0002
        self.generator_lr_decay = self.generator_lr / 200000

        # Discriminator learning rate and decay
        self.discriminator_lr = 0.0001
        self.discriminator_lr_decay = self.discriminator_lr / 200000

        # Label designation
        self.fake_label = 0
        self.true_label = 1

        # Loss scalers
        self.lambda_cycle = 10
        self.lambda_identity = 5

        self.start_decay = 200000
        self.adam_betas = (0.5, 0.999)
        self.writer = None  # SummaryWriter()

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.generator_A2B = Generator(num_features).to(self.device)
        self.generator_B2A = Generator(num_features).to(self.device)

        self.discriminator_A = Discriminator(1).to(self.device)
        self.discriminator_B = Discriminator(1).to(self.device)

        # Optimizers
        generator_params = [
            self.generator_A2B.parameters(), self.generator_B2A.parameters()]
        self.generator_optimizer = torch.optim.Adam(itertools.chain(*generator_params),
                                                    lr=self.generator_lr)
        discriminator_params = [
            self.discriminator_A.parameters(), self.discriminator_B.parameters()]
        self.discriminator_optimizer = torch.optim.Adam(itertools.chain(*discriminator_params),
                                                        lr=self.discriminator_lr)

    def load(self, path, training=False):
        checkpoint = torch.load(path, map_location=self.device)
        epoch = checkpoint["epoch"]
        self.generator_A2B.load_state_dict(
            checkpoint['generator_A2B_state_dict'])
        if training:
            self.generator_B2A.load_state_dict(
                checkpoint['generator_B2A_state_dict'])
            self.discriminator_A.load_state_dict(
                checkpoint['discriminator_A_state_dict'])
            self.discriminator_B.load_state_dict(
                checkpoint['discriminator_B_state_dict'])
            self.generator_optimizer.load_state_dict(
                checkpoint['generator_optimizer_state_dict'])
            self.discriminator_optimizer.load_state_dict(
                checkpoint['discriminator_optimizer.state_dict'])

        return epoch

    def save(self, epoch: int, path: str):
        with torch.no_grad():
            torch.save({
                'epoch': epoch,
                'generator_A2B_state_dict': self.generator_A2B.state_dict(),
                'generator_B2A_state_dict': self.generator_B2A.state_dict(),
                'discriminator_A_state_dict': self.discriminator_A.state_dict(),
                'discriminator_B_state_dict': self.discriminator_B.state_dict(),
                'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
                'discriminator_optimizer.state_dict': self.discriminator_optimizer.state_dict()
            }, path)

    def infer(self, sample):
        return self.generator_A2B(sample)

    def train(self, training_data, epochs=500, start=0):

        # self.generator_A2B.to(self.device)
        # self.generator_B2A.to(self.device)

        # self.discriminator_A.to(self.device)
        # self.discriminator_B.to(self.device)

        assert training_data is not None, 'Training data has not been provided.'

        train_dataset_loader = DataLoader(
            training_data, batch_size=self.batch_size, shuffle=False)

        start_time = time.time()

        for epoch in range(start, epochs):
            time_elapsed = time.time() - start_time
            print("Epoch", epoch, '-', datetime.now().strftime("%H:%M:%S"), '-', '%02d:%02d:%02d Elapsed' % (
                time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))
            for i, sample in enumerate(train_dataset_loader):

                # print(sample[2], sample[3])

                # Learning rate adjustment snippet
                # TODO: len(train_dataset_loader) or len(train_dataset)
                num_iterations = (len(train_dataset_loader) //
                                  self.batch_size) * epoch + i
                if num_iterations > 10000:
                    self.lambda_identity = 0
                if num_iterations > self.start_decay:
                    self.generator_lr = max(
                        0., self.generator_lr - self.generator_lr_decay)
                    self.discriminator_lr = max(
                        0., self.discriminator_lr - self.discriminator_lr_decay)
                    for param_groups in self.generator_optimizer.param_groups:
                        param_groups['lr'] = self.generator_lr
                    for param_groups in self.discriminator_optimizer.param_groups:
                        param_groups['lr'] = self.discriminator_lr

                # print(sample[0].shape)

                real_A = sample[0].permute(0, 2, 1).to(self.device)
                real_B = sample[1].permute(0, 2, 1).to(self.device)

                # print(real_A.shape)

                # Speech A -> Speech B -> Speech A
                fake_B = self.generator_A2B(real_A)
                cycle_A = self.generator_B2A(fake_B)

                # Speech B -> Speech A -> Speech B
                fake_A = self.generator_B2A(real_B)
                cycle_B = self.generator_A2B(fake_A)

                # Speech A -> Speech A (A2B), Speech B -> Speech B (B2A)
                identity_A = self.generator_B2A(real_A)
                identity_B = self.generator_A2B(real_B)

                # Spoofing - unsqueeze needed due to 2D conv
                antispoof_A = self.discriminator_A(fake_A.unsqueeze(1))
                antispoof_B = self.discriminator_B(fake_B.unsqueeze(1))

                # Loss functions
                cycle_A_loss = l1_loss(cycle_A, real_A)
                cycle_B_loss = l1_loss(cycle_B, real_B)
                cycle_loss = cycle_A_loss + cycle_B_loss

                identity_A_loss = l1_loss(identity_A, real_A)
                identity_B_loss = l1_loss(identity_B, real_B)
                identity_loss = identity_A_loss + identity_B_loss

                # When backpropagating for the generator, we want the discriminator to be cheated
                generator_A2B_loss = mse_loss(
                    antispoof_A, torch.ones_like(antispoof_A))
                generator_B2A_loss = mse_loss(
                    antispoof_B, torch.ones_like(antispoof_B))

                generator_loss = generator_A2B_loss + generator_B2A_loss + \
                    self.lambda_cycle * cycle_loss + self.lambda_identity * identity_loss

                self.generator_optimizer.zero_grad()
                self.discriminator_optimizer.zero_grad()
                # Move discriminator zero grad here?

                generator_loss.backward()
                self.generator_optimizer.step()

                d_real_A = self.discriminator_A(real_A.unsqueeze(1))
                generated_A = self.generator_B2A(real_B)
                d_fake_A = self.discriminator_A(generated_A.unsqueeze(1))

                d_real_B = self.discriminator_B(real_B.unsqueeze(1))
                generated_B = self.generator_A2B(real_A)
                d_fake_B = self.discriminator_B(generated_B.unsqueeze(1))

                # 2nd step adversarial loss
                cycled_B = self.generator_A2B(generated_A)
                d_cycled_B = self.discriminator_B(cycled_B.unsqueeze(1))

                cycled_A = self.generator_B2A(generated_B)
                d_cycled_A = self.discriminator_A(cycled_A.unsqueeze(1))       

                # loss functions
                # When backpropagating for the discriminator, we want the discriminator to be powerful
                # 1-step advesarial loss

                discriminator_loss_A_real = mse_loss(d_real_A, torch.ones_like(d_real_A))
                discriminator_loss_A = (discriminator_loss_A_real + mse_loss(d_fake_A, torch.zeros_like(d_fake_A)))/2

                discriminator_loss_B_real = mse_loss(d_real_B, torch.ones_like(d_real_B))
                discriminator_loss_B = (discriminator_loss_B_real + mse_loss(d_fake_B, torch.zeros_like(d_fake_B)))/2

                # 2-step adverserial loss
  

                if not self.twostep:
                # 1-step final loss
                    discriminator_loss = (discriminator_loss_A + discriminator_loss_B)/2
                else:
                # 2-step final loss
                    discriminator_loss_A_cycled =  torch.mean((0 - d_cycled_A) ** 2)
                    discriminator_loss_B_cycled = torch.mean((0 - d_cycled_B) ** 2)
                    discriminator_loss_A_2nd = (discriminator_loss_A_real + discriminator_loss_A_cycled) / 2.0
                    discriminator_loss_B_2nd = (discriminator_loss_B_real + discriminator_loss_B_cycled) / 2.0  
                    discriminator_loss = (discriminator_loss_A + discriminator_loss_B) / 2.0 + (discriminator_loss_A_2nd + discriminator_loss_B_2nd) / 2.0

                self.generator_optimizer.zero_grad()
                self.discriminator_optimizer.zero_grad()

                discriminator_loss.backward()
                self.discriminator_optimizer.step()

                if (i % 100) == 0:
                    print("Iteration", num_iterations, "-",
                          "Generator loss:", generator_loss.item(), "-",
                          "Discriminator loss:", discriminator_loss.item(), "-",
                          "Cycle loss:", cycle_loss.item(), "-",
                          "Identity loss:", identity_loss.item())
                    #   "Discriminator A loss:", discriminator_A_loss.item(),
                    #   "Discriminator B loss:", discriminator_B_loss.item())

                    # if infering:
                    #     assert eval_set is not None, 'Training dataset has not been provided.'
                    #     self.infer(eval_set)

            yield epoch
