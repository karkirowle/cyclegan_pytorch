
import torch.nn as nn
import torch
import math

class _ConvLayer(nn.Module):

    #This acts as a base Conv Class and the forward func specifies whether 1D or 2D
    def __init__(self,in_filter,out_filter,kernel, stride):
        super(_ConvLayer, self).__init__()

        self.in_filter = in_filter
        self.out_filter = out_filter
        self.kernel = kernel
        self.stride = stride

        # Same padding with known stride and dilation of 1
        # i.e. L0/1 L0/2 L0/3 multiples
        assert (type(stride) == int) | (type(stride) == tuple)

        # TODO: I threw up a bit
        if type(stride) == tuple:

            self.cut_last_element_X = (self.kernel[0] % 2 == 0 and self.stride[0] == 1)
            self.cut_last_element_Y = (self.kernel[1] % 2 == 0 and self.stride[1] == 1)

            self.padding = (math.ceil((self.kernel[0] - self.stride[0]) / 2),
                            math.ceil((self.kernel[1] - self.stride[1]) / 2))

        else:
            self.cut_last_element_X = False
            self.cut_last_element_Y = False
            self.padding = math.ceil((self.kernel - self.stride) / 2)

        self.conv = None

    def forward(self, x):
        out = self.conv(x)

        if (self.cut_last_element_X) & (len(x.shape) == 4):
            out = out[:,:,:-1,:]
        if (self.cut_last_element_Y) & (len(x.shape) == 4):
            out = out[:,:,:,:-1]

        return out


class ConvLayer1D(_ConvLayer):

    def __init__(self, in_filter, out_filter, kernel, stride):
        super().__init__(in_filter,out_filter,kernel, stride)
        self.conv = nn.Conv1d(self.in_filter, self.out_filter, self.kernel, stride, padding=self.padding)


class ConvLayer2D(_ConvLayer):

    def __init__(self, in_filter, out_filter, kernel, stride):
        super().__init__(in_filter,out_filter,kernel, stride)
        self.conv = nn.Conv2d(self.in_filter, self.out_filter, self.kernel, stride, padding=self.padding)


class ResidualBlock1D(nn.Module):

    def __init__(self,in_filter,out_filter=1024,kernel=3,stride=1):
        # TODO: Assuming same length this might be equivalent to one in VC repo, but you should double check this
        # TODO: assuming GLU should act on GLU filter dim?
        #self.h1_glu = glu(torch.cat((self.h1_gates,self.h1_norm_gates))) # single input
        super().__init__()
        self.block = nn.Sequential(
            ConvLayer1D(in_filter, out_filter * 4, kernel,stride),
            nn.InstanceNorm1d(num_features=out_filter*4, eps=1e-6),
            nn.GLU(dim=1),
            ConvLayer1D(out_filter*2, out_filter, kernel,stride),
            nn.InstanceNorm1d(num_features=out_filter, eps=1e-6)
        )

    def forward(self, x):

        h3 = x + self.block(x)
        return h3


class Downsample(nn.Module):

    def __init__(self, out_filter):
        super(Downsample,self).__init__()
        self.conv = None
        self.instance_norm = None
        self.glu = nn.GLU(dim=1)

    def forward(self, x):
        return self.glu(self.instance_norm(self.conv(x)))


class Downsample1D(Downsample):

    def __init__(self,in_filter,out_filter,kernel,stride):
        super().__init__(out_filter)
        self.conv = ConvLayer1D(in_filter,out_filter*2,kernel,stride)
        self.instance_norm = nn.InstanceNorm1d(num_features=out_filter*2,eps=1e-6, affine=True)


class Downsample2D(Downsample):

    def __init__(self,in_filter,out_filter,kernel,stride,padding=None):
        """None lets the default padding go while padding=0 cancels it"""
        super().__init__(out_filter)
        self.conv = ConvLayer2D(in_filter,out_filter*2,kernel,stride)
        if padding is not None: self.conv.conv.padding = padding
        self.instance_norm = nn.InstanceNorm2d(num_features=out_filter*2,eps=1e-6, affine=True)

class PixelShuffle1D(torch.nn.Module):
    """
    1D pixel shuffler. https://arxiv.org/pdf/1609.05158.pdf
    Upscales sample length, downscales channel length
    "short" is input, "long" is output
    """
    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width

        x = x.contiguous().view([batch_size, self.upscale_factor, long_channel_len, short_width])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)

        return x

class Upsample1D(nn.Module):

    def __init__(self,in_filter,kernel,stride=1):
        super().__init__()

        # Outfilter is determined by shuffling factor
        shuffle_size = 2

        # Filter size decreses by the shuffle size
        out_filter = in_filter // shuffle_size
        self.block = nn.Sequential(ConvLayer1D(in_filter,out_filter * shuffle_size * 2 ,kernel,stride),
                                   PixelShuffle1D(2),
                                   nn.InstanceNorm1d(num_features=out_filter * 2), # GLU halves, so
                                   nn.GLU(dim=1))

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):

    # TODO: Maybe be a bit more explicit about shuffling size?
    def __init__(self, in_feature):
        super(Generator,self).__init__()
        self.block = nn.Sequential(ConvLayer1D(in_filter=in_feature, out_filter=128 * 2, kernel=15,stride=1),
                                   nn.GLU(dim=1),

                                   Downsample1D(in_filter=128, out_filter=256,kernel=5,stride=2),
                                   Downsample1D(in_filter=256, out_filter=512,kernel=5,stride=2),

                                   ResidualBlock1D(in_filter=512,out_filter=512,kernel=3),
                                   ResidualBlock1D(in_filter=512, out_filter=512, kernel=3),
                                   ResidualBlock1D(in_filter=512, out_filter=512, kernel=3),
                                   ResidualBlock1D(in_filter=512, out_filter=512, kernel=3),
                                   ResidualBlock1D(in_filter=512, out_filter=512, kernel=3),
                                   ResidualBlock1D(in_filter=512, out_filter=512, kernel=3),

                                   Upsample1D(in_filter=512,kernel=5),

                                   Upsample1D(in_filter=256,kernel=5),
                                   ConvLayer1D(in_filter=128, out_filter=in_feature, kernel=5,stride=1),
                                   )

    def forward(self, x):
        return self.block(x)

class PermuteBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x.permute(0,2,3,1).contiguous()


class Discriminator(nn.Module):

    def __init__(self, in_feature=1):
        super().__init__()

        self.block = nn.Sequential(
            # input: (1 x 1 x 24 x 128)
            ConvLayer2D(in_filter=in_feature,out_filter=256, kernel=(3, 3), stride=(1, 2)),
            nn.GLU(dim=1),
            # input: (1 x 128 x 24 x 64)
            Downsample2D(in_filter=128, out_filter=256, kernel=(3, 3), stride=(2, 2)),
            # input: (1 x 256 x 12 x 32)
            Downsample2D(in_filter=256, out_filter=512, kernel=(3, 3), stride=(2, 2)),
            # input: (1 x 512 x 6 x 16)
            #nn.ZeroPad2d((3, 2, 0, 1)),
            Downsample2D(in_filter=512, out_filter=1024, kernel=(6, 3), stride=(1, 2)),
            # input: (1 x 1024 x 1 x 1)
            PermuteBlock(),
            # input?: (1 x 1024)
            nn.Linear(1024, 1)
            #nn.Sigmoid()
        )

    def forward(self, x):
        return self.block(x)



if __name__ == '__main__':

    print("")