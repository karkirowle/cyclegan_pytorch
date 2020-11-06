
import nnmnkwii

from torch.utils.tensorboard import SummaryWriter
from data_utils import VCC2016DataSource, MCEPWrapper
from nnmnkwii.datasets import FileSourceDataset, MemoryCacheDataset
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from utils import world_decode_spectral_envelop, world_speech_synthesis, pitch_conversion_with_logf0, wpindex
import fastdtw
from modules import Generator, Discriminator, Modern_DBLSTM_1, PatchSampleF, PatchNCELoss
import scipy
import librosa
import os
import itertools
from torch.nn.functional import l1_loss, mse_loss
#from tf_adam import Adam
import numpy as np
from nnmnkwii.autograd import modspec
import argparse

import sys

def train(dtw, modspec_loss,validation_A_dir, validation_B_dir, l2):
    ############## HYPERPARAMETER PART #######################################

    os.makedirs(validation_A_dir, exist_ok=True)
    os.makedirs(validation_B_dir, exist_ok=True)

    batch_size = 1
    num_epochs = 1000
    num_features = 24
    fs = 16000
    data_root="/home/boomkin/repos/Voice_Converter_CycleGAN/data"
    #data_root="./data"



    generator_lr = 0.0002
    generator_lr_decay = generator_lr / 200000
    discriminator_lr = 0.0001
    discriminator_lr_decay = discriminator_lr / 200000
    fake_label = 0
    true_label = 1
    #lambda_cycle = 10
    #lambda_identity = 5
    lambda_NCE = 1
    start_decay = 200000
    adam_betas = (0.5, 0.999)
    patch_number = 16
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


    generator = Generator(24).to("cuda")
    #generator_B2A = Generator(24).to("cuda")
    discriminator = Discriminator(1).to("cuda")
    #discriminator_B = Discriminator(1).to("cuda")
    patch_sampler_mlp = PatchSampleF()
    #nce_criterion = PatchNCELoss()
    #art_discriminator_A = Discriminator(1).to("cuda")
    #art_discriminator_B = Discriminator(1).to("cuda")

    #art_extractor = torch.load("articulatory_model/sarticulatory_model_pz.pt")

    generator_params = [generator.parameters()]
    generator_optimizer = torch.optim.Adam(itertools.chain(*generator_params),
                                             lr=generator_lr)
    discriminator_params = [discriminator.parameters()]

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

            # Fake and embeddings generations for NCE Loss
            feat_k, fake_B = generator(real_A)
            feat_q, fake_fake_B = generator(fake_B)


            # Sampling the patches


            feat_k_pool, patch_ids = patch_sampler_mlp(feat_k, patch_number, None)
            feat_q_pool, _ = patch_sampler_mlp(feat_q, patch_number, patch_ids)

            total_nce_loss_X = 0

            nce_criterion = []
            for nce_layer in range(len(feat_k_pool)):
                nce_criterion.append(PatchNCELoss().to("cuda"))

            for f_q, f_k, crit in zip(feat_q_pool, feat_k_pool, nce_criterion):
                loss = crit(f_q, f_k) * lambda_NCE
                total_nce_loss_X += loss.mean()

            feat_k_2, id_B = generator(fake_B)
            feat_q_2, id_id_B = generator(id_B)

            feat_k_pool_2, patch_ids = patch_sampler_mlp(feat_k_2, patch_number, None)
            feat_q_pool_2, _ = patch_sampler_mlp(feat_q_2, patch_number, patch_ids)

            total_nce_loss_Y = 0

            nce_criterion_2 = []
            for nce_layer in range(len(feat_k_pool)):
                nce_criterion_2.append(PatchNCELoss().to("cuda"))

            for f_q, f_k, crit in zip(feat_q_pool_2, feat_k_pool_2, nce_criterion_2):
                loss = crit(f_q, f_k) * lambda_NCE
                total_nce_loss_Y += loss.mean()



            # Spoofing - unsqueeze needed due to 2D conv
            antispoof_B = discriminator(fake_B.unsqueeze(1))

            # When backpropagating for the generator, we want the discriminator to be cheated
            generation_loss = mse_loss(antispoof_B, torch.ones_like(antispoof_B)*true_label)

            generator_loss = generation_loss + (total_nce_loss_X + total_nce_loss_Y)*0.5

            generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()

            d_real_A = discriminator(real_A.unsqueeze(1))
            d_fake_A = discriminator(generator(real_B)[1].unsqueeze(1))

            # When backpropagating for the discriminator, we want the discriminator to be powerful
            discriminator_loss = mse_loss(d_real_A, torch.ones_like(d_real_A)*true_label)/2 + \
                mse_loss(d_fake_A, torch.ones_like(d_fake_A)*fake_label)/2

            discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            discriminator_optimizer.step()

            if (i % 50) == 0:
                print("Iteration ", num_iterations,
                      " Generator loss: ", generator_loss.item(),
                      " Discriminator loss:", discriminator_loss.item(),
                      " NCE X loss: ", total_nce_loss_X.item(),
                      " NCE Y loss: ", total_nce_loss_Y.item())
                writer.add_scalar('Generator loss', generator_loss.item(), num_iterations)
                writer.add_scalar('Discriminator loss', discriminator_loss.item(), num_iterations)


        if (epoch % 50) == 0:

            # Model save
            with torch.no_grad():

                if not os.path.exists("checkpoint"):
                    os.mkdir("checkpoint")

                torch.save(generator.state_dict(), "checkpoint/generator_A2B.pt")
                #torch.save(generator_B2A.state_dict(), "checkpoint/generator_B2A.pt")
                torch.save(discriminator.state_dict(), "checkpoint/discriminator_A.pt")
                #torch.save(discriminator_B.state_dict(), "checkpoint/discriminator_B.pt")

                for i in range(len(SF1_test_data_source)):

                    feature_A = SF1_test_data_source[i]
                    feature_B = TF2_test_data_source[i]
                    filename_A = os.path.basename(SF1_test_data_source.dataset.collected_files[i][0])
                    filename_B = os.path.basename(TF2_test_data_source.dataset.collected_files[i][0])

                    # TODO: szerintem az F0-val van gond meg valahol
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

                    mcep_A = (feature_A[None,:,1:25] - mean_mcep_A)/std_mcep_A

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

                    _, fake_B = generator(real_A)
                    #fake_A = generator_B2A(real_B)


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



if __name__ == '__main__':

    validation_A_dir_default="./validation_output_debug_2/converted_A"
    validation_B_dir_default="./validation_output_debug_2/converted_B"

    parser = argparse.ArgumentParser(description = 'Train CycleGAN model for datasets.')
    parser.add_argument('--dtw', action='store_true')
    parser.add_argument('--modspec', action='store_true')
    parser.add_argument('--l2', action='store_true')


    parser.add_argument('--validation_A_dir',
                        type = str,
                        help = 'Convert validation A after each training epoch. ', default = validation_A_dir_default)
    parser.add_argument('--validation_B_dir',
                        type = str,
                        help = 'Convert validation B after each training epoch. ', default = validation_B_dir_default)

    argv = parser.parse_args()

    print("dtw",argv.dtw)
    print("modspec",argv.modspec)
    print("l2 cycle consistencyc",argv.modspec)

    print("validation A", argv.validation_A_dir)
    print("validation B", argv.validation_B_dir)

    train(argv.dtw, argv.modspec,argv.validation_A_dir,argv.validation_B_dir, argv.l2)
