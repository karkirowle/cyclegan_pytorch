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

### Interesting points that I figured out while reimplementing
- Discriminator is 2D, because authors believe this is better for frequency info (agreed), but
1D generator to preserve temporal structure (not sure why 2D wouldn't work there?)
- InstanceNorm should be set to affine=True in PyTorch according to PyTorch reimplementation,
which is not the default option
- The discriminator is very odd in the sense that it is using (1 x 6 x ? x 1) as output,
check if that's the same in DCGAN? I can see half of the design motive here comes from the fact
that you might want to learn framewise information, but the six makes no sense to me.

### Backend design points to check
Things to investigate:

- Doing GLU in one conv layer and splitting seems to be equivalent to doing with two CONV layers?

- Downsample blocks are stride 2 blocks Conv+Instance Norm blocks
- Due to the PixelShuffler dimensions divided by two twice, the main constraint seems to be that the input needs to be
multiples of 4
- Upsampling is basically at the expense of channel dimensions
- Residual block have an intersting implementational detail: after the GLU blocks, it projects down the results

- Kernel sizes of 5 and 15 are interesting

- The last few layers of the discriminator seem to be very random
  
 ### Problematic points that should be checked
 
 - TODO: DatasetLoader: is it called only once? (i.e is sampling in time happening every epoch?)
 - TODO: Pitch conversion (normalisation/denormalisation or CycleGAN-based) 
 
 