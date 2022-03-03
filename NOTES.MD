### Interesting points that I figured out while reimplementing
- Discriminator is 2D, because authors believe this is better for frequency info (agreed), but
1D generator to preserve temporal structure (not sure why 2D wouldn't work there?)
- InstanceNorm should be set to affine=True to match Tensorflow default, but it doesn't seem to really matter
in practice.
- The discriminator is very odd in the sense that it is using (1 x 6 x ? x 1) as output, This is from the paper PatchGAN,
and they use the PatchGAN without sigmoid there. In the CycleGAN-VC paper they indicate that there is a sigmoid, but that's been
causing problems even in TF, and convergence seemed to be better also in PyTorch/
- The beta1 parameter of the Adam optimiser is non-default, but that's basically the same parameter since the DCGAN paper. Nobody
dares to touch it.
- DatasetLoader is applied lazily according to this post: https://discuss.pytorch.org/t/data-augmentation-folr-labels-and-images/38335
That means its safe to implement data augmentation there.
- Random sample frames are taken from the input and output slices too, that seems to be important, because  (1) there is no alignment enforced
(2) it could make conversion worse when the right segment is not there.

- How PyTorch handles sparsity causes problems: https://discuss.pytorch.org/t/poor-convergence-on-pytorch-compared-to-tensorflow-using-adam-optimizer/31425/6
https://discuss.pytorch.org/t/suboptimal-convergence-when-compared-with-tensorflow-model/5099/22
- The main difficulty was the PixleShuffler layer tf.reshape and pytorch reshpae seems to be not identical in reshaping, confirm it 
- Lot of people seem to actual two conv layers for GLU, but that is not neccessary
- It seems to be the main focus of tuning in CycleGAN-VC were the kernel sizes and making the generator 1D


### Other stuff
- Downsample blocks are stride 2 blocks Conv+Instance Norm blocks
- Due to the PixelShuffler dimensions divided by two twice, the main constraint seems to be that the input needs to be
multiples of 4
- Upsampling is basically at the expense of channel dimensions
- Kernel sizes of 5 and 15 are interesting
- [Blogpost about GLUs](https://leimao.github.io/blog/Gated-Linear-Units/)
