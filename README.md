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

Things to investigate:
- How equivalent are the GLU implementations


Downsample blocks are stride 2 blocks Conv+Instance Norm blocks
Constraint seems to be: multiples of 4
Upsampling is basically at the expense of channels?

Residual block have an intersting implementational detail:
after the GLU blocks, it projects down the results

TODO: In preprocessing pad to multiples of four
TODO: why kernel_size = 5 in conv?

Why discriminator is 2D convolution based? 
The last few layers of the discriminator seem to be very random
    