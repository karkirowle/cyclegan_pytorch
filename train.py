
import nnmnkwii

from data_utils import VCC2016DataSource, MCEPWrapper
from nnmnkwii.datasets import FileSourceDataset
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from modules import Generator, Discriminator

from torch.nn.functional import l1_loss, mse_loss
############## HYPERPARAMETER PART #######################################

batch_size = 1
num_epochs = 5000
num_features = 24
data_root="/home/boomkin/repos/Voice_Converter_CycleGAN/data"

generator_learning_rate = 0.0002
generator_learning_rate_decay = generator_learning_rate / 200000
discriminator_learning_rate = 0.0001
discriminator_learning_rate_decay = discriminator_learning_rate / 200000
fake_label = 0
true_label = 1

############## HYPERPARAMETER PART #######################################


SF1_train_data_source = FileSourceDataset(VCC2016DataSource(data_root, ["SF1"],training=True))
TF2_train_data_source = FileSourceDataset(VCC2016DataSource(data_root, ["TF2"],training=True))
SF1_test_data_source = FileSourceDataset(VCC2016DataSource(data_root, ["SF1"],training=False))
TF2_test_data_source = FileSourceDataset(VCC2016DataSource(data_root, ["TF2"],training=False))

train_dataset = MCEPWrapper(SF1_train_data_source, TF2_train_data_source)
test_dataset = MCEPWrapper(SF1_test_data_source, TF2_test_data_source)

train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


generator_A2B = Generator(num_features).to("cuda")
generator_B2A = Generator(num_features).to("cuda")
discriminator_A = Discriminator(1).to("cuda")
discriminator_B = Discriminator(1).to("cuda")



generator_A_optimizer = torch.optim.Adam(generator_A2B.parameters(),
                                         lr=generator_learning_rate,
                                         weight_decay=generator_learning_rate_decay)

discriminator_A_optimizer = torch.optim.Adam(discriminator_A.parameters(),
                                           lr=discriminator_learning_rate,
                                           weight_decay=discriminator_learning_rate_decay)

generator_B_optimizer = torch.optim.Adam(generator_B2A.parameters(),
                                         lr=generator_learning_rate,
                                         weight_decay=generator_learning_rate_decay)

discriminator_B_optimizer = torch.optim.Adam(discriminator_B.parameters(),
                                           lr=discriminator_learning_rate,
                                           weight_decay=discriminator_learning_rate_decay)



for epoch in range(num_epochs):

    for i,sample in enumerate(train_dataset_loader):

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
        generation_loss = mse_loss(antispoof_A, torch.ones_like(antispoof_A)*true_label) + mse_loss(antispoof_B, torch.ones_like(antispoof_B)*true_label)

        generator_loss = cycle_loss + identity_loss + generation_loss

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

        if (i % 500) == 0:
            print("Iteration ", i, " Generator loss: ", generator_loss.item(), " Discriminator loss", discriminator_loss.item())

    if (epoch % 100) == 0:

        # Model save
        for i,sample in enumerate(test_dataset_loader):
            real_A = sample[0].permute(0, 2, 1).to("cuda")
            real_B = sample[1].permute(0, 2, 1).to("cuda")

            fake_B = generator_A2B(real_A)
            fake_A = generator_B2A(real_B)

            # world synthesis