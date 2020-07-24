# CycleGAN Pytorch

preprocessing: preprocessing data loader, normal data loader

neural network backends

metrics

hparam sensitivity?

Requirements:
- nnmnkwii
- PyTorch
- librosa
- pyworld

### Backend design points to check
Things to investigate:

- Doing GLU in one conv layer and splitting seems to be equivalent to doing with two CONV layers?

- Downsample blocks are stride 2 blocks Conv+Instance Norm blocks
- Due to the PixelShuffler dimensions divided by two twice, the main constraint seems to be that the input needs to be
multiples of 4
- Upsampling is basically at the expense of channel dimensions
- Residual block have an intersting implementational detail: after the GLU blocks, it projects down the results

- TODO: In preprocessing pad to multiples of four
- TODO: why kernel_size = 5 in conv? There are other intere
- Why discriminator is 2D convolution based? 
- The last few layers of the discriminator seem to be very random
  
 ### Problematic points that should be checked
 
 - TODO: DatasetLoader: is it called only once? (i.e is sampling in time happening every epoch?)
 - TODO: Pitch conversion (normalisation/denormalisation or CycleGAN-based) 