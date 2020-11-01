
import nnmnkwii

from torch.utils.tensorboard import SummaryWriter
from data_utils import VCC2016DataSource, MCEPWrapper
from nnmnkwii.datasets import FileSourceDataset, MemoryCacheDataset
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from utils import world_decode_spectral_envelop, world_speech_synthesis, pitch_conversion_with_logf0, wpindex
import fastdtw
from modules import Generator, Discriminator
import scipy
import librosa
import os
import itertools
from torch.nn.functional import l1_loss, mse_loss
#from tf_adam import Adam
import numpy as np
from nnmnkwii.autograd import modspec
import argparse

def train(dtw, modspec_loss):
    ############## HYPERPARAMETER PART #######################################

    batch_size = 1
    num_epochs = 1000
    num_features = 24
    fs = 16000
    data_root="/home/boomkin/repos/Voice_Converter_CycleGAN/data"
    #data_root="./data"
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
    adam_betas = (0.5, 0.999)
    writer = SummaryWriter()
    ############## HYPERPARAMETER PART #######################################

    # We create nice dirs

    if not os.path.exists(validation_A_dir):
        os.mkdir(validation_A_dir)
    if not os.path.exists(validation_B_dir):
        os.mkdir(validation_B_dir)
    if not os.path.exists("figures"):
        os.mkdir("figures")

    SF1_train_data_source = MemoryCacheDataset(FileSourceDataset((VCC2016DataSource(data_root, ["SF1"],training=True))))
    TF2_train_data_source = MemoryCacheDataset(FileSourceDataset((VCC2016DataSource(data_root, ["TF2"],training=True))))
    SF1_test_data_source = MemoryCacheDataset(FileSourceDataset((VCC2016DataSource(data_root, ["SF1"],training=False))))
    TF2_test_data_source = MemoryCacheDataset(FileSourceDataset((VCC2016DataSource(data_root, ["TF2"],training=False))))

    train_dataset = MCEPWrapper(SF1_train_data_source, TF2_train_data_source, mfcc_only=True, dtw=dtw)
    #test_dataset = MCEPWrapper(SF1_test_data_source, TF2_test_data_source, mfcc_only=False, norm_calc=False)
    #test_dataset.input_meanstd = train_dataset.input_meanstd
    #test_dataset.output_meanstd = train_dataset.output_meanstd

    train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #test_dataset_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    generator_A2B = Generator(24).to("cuda")
    generator_B2A = Generator(24).to("cuda")
    discriminator_A = Discriminator(1).to("cuda")
    discriminator_B = Discriminator(1).to("cuda")


    generator_params = [generator_A2B.parameters(), generator_B2A.parameters()]
    generator_optimizer = torch.optim.Adam(itertools.chain(*generator_params),
                                             lr=generator_lr)
    discriminator_params = [discriminator_A.parameters(), discriminator_B.parameters()]
    discriminator_optimizer = torch.optim.Adam(itertools.chain(*discriminator_params),
                                               lr=discriminator_lr)
    for epoch in range(num_epochs):
        print("Epoch ", epoch)
        for i,sample in enumerate(train_dataset_loader):

            # Learning rate adjustment snippet
            # TODO: len(train_dataset_loader) or len(train_dataset)
            num_iterations = (len(train_dataset_loader) // batch_size) * epoch + i
            if num_iterations > 10000:
                lambda_identity = 0
            if num_iterations > start_decay:
                generator_lr = max(0., generator_lr - generator_lr_decay)
                discriminator_lr = max(0., discriminator_lr - discriminator_lr_decay)
                for param_groups in generator_optimizer.param_groups:
                    param_groups['lr'] = generator_lr
                for param_groups in discriminator_optimizer.param_groups:
                    param_groups['lr'] = discriminator_lr

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
            cycle_A_loss = l1_loss(cycle_A, real_A)
            cycle_B_loss = l1_loss(cycle_B, real_B)
            cycle_loss = cycle_A_loss + cycle_B_loss

            identity_A_loss = l1_loss(identity_A, real_A)
            identity_B_loss = l1_loss(identity_B, real_B)
            identity_loss = identity_A_loss + identity_B_loss

            # When backpropagating for the generator, we want the discriminator to be cheated
            generation_loss = mse_loss(antispoof_A, torch.ones_like(antispoof_A)*true_label) + \
                mse_loss(antispoof_B, torch.ones_like(antispoof_B)*true_label)



            generator_loss = lambda_cycle * cycle_loss + lambda_identity * identity_loss + generation_loss

            if modspec_loss:
                modspec_b_loss = mse_loss(modspec(real_B.squeeze(0).T.cpu()), modspec(fake_B.squeeze(0).T.cpu()))
                modspec_a_loss = mse_loss(modspec(real_A.squeeze(0).T.cpu()), modspec(fake_A.squeeze(0).T.cpu()))
                modspec_loss = 0.0001 * (modspec_a_loss/2 + modspec_b_loss/2)


            generator_optimizer.zero_grad()
            if modspec_loss:
                # TODO: For some reason mod_spec has to be backpropped separately
                generator_loss.backward(retain_graph=True)
                modspec_loss.backward()
            else:
                generator_loss.backward()
            generator_optimizer.step()
            generator_optimizer.step()

            d_real_A = discriminator_A(real_A.unsqueeze(1))
            d_fake_A = discriminator_A(generator_B2A(real_B).unsqueeze(1))

            d_real_B = discriminator_B(real_B.unsqueeze(1))
            d_fake_B = discriminator_B(generator_A2B(real_A).unsqueeze(1))

            # When backpropagating for the discriminator, we want the discriminator to be powerful
            discriminator_A_loss = mse_loss(d_real_A, torch.ones_like(d_real_A)*true_label)/4 + \
                mse_loss(d_fake_A, torch.ones_like(d_fake_A)*fake_label)/4
            discriminator_B_loss = mse_loss(d_real_B, torch.ones_like(d_real_B)*true_label)/4 + \
                mse_loss(d_fake_B, torch.ones_like(d_fake_B)*fake_label)/4
            discriminator_loss = discriminator_A_loss + discriminator_B_loss

            discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            discriminator_optimizer.step()

            if (i % 50) == 0:
                print("Iteration ", num_iterations,
                      " Generator loss: ", generator_loss.item(),
                      " Discriminator loss:", discriminator_loss.item(),
                      " Cycle loss: ", cycle_loss.item(),
                      " Identity loss: ", identity_loss.item(),
                      " Discriminator A loss: ", discriminator_A_loss.item(),
                      " Discriminator B loss: ", discriminator_B_loss.item(),
                      " Modspec lkoss", modspec_loss.item())
                writer.add_scalar('Generator loss', generator_loss.item(), num_iterations)
                writer.add_scalar('Discriminator loss', discriminator_loss.item(), num_iterations)
                writer.add_scalar('Cycle loss', cycle_loss.item(), num_iterations)
                writer.add_scalar('Identity loss', identity_loss.item(), num_iterations)
                writer.add_scalar('Discriminator A loss', discriminator_B_loss.item(), num_iterations)
                writer.add_scalar('Discriminator B loss', discriminator_A_loss.item(), num_iterations)

        if (epoch % 50) == 0:

            # Model save
            with torch.no_grad():

                if not os.path.exists("checkpoint"):
                    os.mkdir("checkpoint")

                torch.save(generator_A2B.state_dict(), "checkpoint/generator_A2B.pt")
                torch.save(generator_B2A.state_dict(), "checkpoint/generator_B2A.pt")
                torch.save(discriminator_A.state_dict(), "checkpoint/discriminator_A.pt")
                torch.save(discriminator_B.state_dict(), "checkpoint/discriminator_B.pt")

                for i in range(len(SF1_test_data_source)):

                    feature_A = SF1_test_data_source[i]
                    feature_B = TF2_test_data_source[i]
                    filename_A = os.path.basename(SF1_test_data_source.dataset.collected_files[i][0])
                    filename_B = os.path.basename(TF2_test_data_source.dataset.collected_files[i][0])

                    f0_A = feature_A[:,0]
                    f0_B = feature_B[:,0]
                    ap_A = feature_A[:,25:]
                    ap_B = feature_B[:,25:]

                    mean_B, std_B = train_dataset.output_meanstd
                    mean_A, std_A = train_dataset.input_meanstd
                    mean_f0_A = mean_A[0]
                    mean_f0_B = mean_B[0]
                    std_f0_A = std_A[0]
                    std_f0_B = std_B[0]
                    mean_mcep_A = mean_A[1:25]
                    mean_mcep_B = mean_B[1:25]
                    std_mcep_A = std_A[1:25]
                    std_mcep_B = std_B[1:25]

                    mcep_A = (feature_A[None,:,1:25] - mean_mcep_A)/std_mcep_A        #other = OtherParameters(f0_A,f0_B,bap_A,bap_B)

                    mcep_B = (feature_B[None,:,1:25] - mean_mcep_B)/std_mcep_B


                    if dtw:
                        # DTW warping the validation
                        C, wp = librosa.sequence.dtw(mcep_A[0,:,:].T,mcep_B[0,:,:].T, backtrack=True)

                        # TODO: DTW path behaviour is not as I would expect, so I have to use a heuristic for now
                        # At conversion time, we pad because DTW doesn't guarantee the multiplicity needed
                        wp_pad = wpindex(wp[::-1,0], multiple=4)
                        mcep_A = mcep_A[:,wp_pad,:]
                        f0_A = f0_A[wp_pad]
                        ap_A = ap_A[wp_pad,:]

                    real_A = torch.FloatTensor(mcep_A).permute(0, 2, 1).to("cuda")
                    real_B = torch.FloatTensor(mcep_B).permute(0, 2, 1).to("cuda")

                    fake_B = generator_A2B(real_A)
                    fake_A = generator_B2A(real_B)


                    # Conversion of A -> B
                    fake_B = fake_B.cpu().detach().numpy()[0,:,:]
                    fake_B = fake_B.T*std_B[1:25] + mean_B[1:25]
                    fake_B = np.ascontiguousarray(fake_B).astype(np.float64)

                    # Separation
                    f0 = pitch_conversion_with_logf0(f0_A,
                                                     mean_f0_A,
                                                     std_f0_A,
                                                     mean_f0_B,
                                                     std_f0_B)

                    sp = world_decode_spectral_envelop(fake_B, fs)
                    ap = np.ascontiguousarray(ap_A)

                    # Save figure

                    decoded = world_decode_spectral_envelop(real_A.detach().cpu().numpy()[0,:,:].T*std_A[1:25] + mean_A[1:25],fs)

                    speech_fake_B = np.clip(world_speech_synthesis(f0, sp, ap, fs, frame_period=5),-1,1)

                    librosa.output.write_wav(os.path.join(validation_A_dir, filename_A), speech_fake_B, fs)

                    # Conversion of B -> A
                    fake_A = fake_A.cpu().detach().numpy()[0,:,:]
                    fake_A = fake_A.T*std_A[1:25] + mean_A[1:25]
                    fake_A = np.ascontiguousarray(fake_A).astype(np.float64)

                    debug_real_A = real_A.cpu().detach().numpy()[0,:,:]
                    debug_real_A = debug_real_A.T*std_A[1:25] + mean_A[1:25]
                    debug_real_A = np.ascontiguousarray(debug_real_A).astype(np.float64)
                    # Separation
                    f0 = pitch_conversion_with_logf0(f0_A,
                                                     mean_f0_A,
                                                     std_f0_A,
                                                     mean_f0_A,
                                                     std_f0_A)

                    sp = world_decode_spectral_envelop(debug_real_A, fs)
                    ap = np.ascontiguousarray(ap_A)

                    speech_fake_A = np.clip(world_speech_synthesis(f0, sp, ap, fs, frame_period=5),-1,1)

                    librosa.output.write_wav(os.path.join(validation_B_dir, filename_B), speech_fake_A, fs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Train CycleGAN model for datasets.')
    parser.add_argument('--dtw', action='store_true')
    parser.add_argument('--modspec', action='store_true')

    argv = parser.parse_args()

    print("dtw",argv.dtw)
    print("modspec",argv.modspec)
    train(argv.dtw, argv.modspec)
