
import torch.nn as nn
import torch
import math
from torch.nn.functional import glu
# Some very interesting blogposts about GLUs https://leimao.github.io/blog/Gated-Linear-Units/
# TODO: Finding out in_filters

# nn.InstanceNorm2D eps=1e-6

class ConvLayer(nn.Module):

    #This acts as a base Conv Class and the forward func specifies whether 1D or 2D
    def __init__(self,in_filter,out_filter,kernel, stride):
        super(ConvLayer, self).__init__()

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

    def forward(self, input):

        out = self.conv(input)

        if (self.cut_last_element_X) & (len(input.shape) == 4):
            out = out[:,:,:-1,:]
        if (self.cut_last_element_Y) & (len(input.shape) == 4):
            out = out[:,:,:,:-1]

        return out


class ConvLayer1D(ConvLayer):

    def __init__(self, in_filter, out_filter, kernel, stride):
        super().__init__(in_filter,out_filter,kernel, stride)
        self.conv = nn.Conv1d(self.in_filter, self.out_filter, self.kernel, stride, padding=self.padding)


class ConvLayer2D(ConvLayer):

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

    def forward(self, inputs):

        h3 = inputs + self.block(inputs)
        return h3


class Downsample(nn.Module):

    def __init__(self, out_filter):
        super(Downsample,self).__init__()
        self.conv = None
        self.instance_norm = None
        self.glu = nn.GLU(dim=1)

    def forward(self,input):
        return self.glu(self.instance_norm(self.conv(input)))


class Downsample1D(Downsample):

    def __init__(self,in_filter,out_filter,kernel,stride):
        super().__init__(out_filter)
        self.conv = ConvLayer1D(in_filter,out_filter*2,kernel,stride)
        self.instance_norm = nn.InstanceNorm1d(num_features=out_filter*2,eps=1e-6)


class Downsample2D(Downsample):

    def __init__(self,in_filter,out_filter,kernel,stride):
        super().__init__(out_filter)
        self.conv = ConvLayer2D(in_filter,out_filter*2,kernel,stride)
        self.instance_norm = nn.InstanceNorm2d(num_features=out_filter*2,eps=1e-6)


class PixelShuffler(nn.Module):

    def __init__(self,shuffle_size=2):
        super().__init__()
        self.shuffle_size = shuffle_size

    def forward(self,input):
        # TODO: Check notation order. I think tensorflow has (n,w,c), we have (n,c,w)
        n,c,w = input.size()
        ow = c // self.shuffle_size
        oc = w * self.shuffle_size

        return torch.reshape(input, shape = [n,ow,oc])


class Upsample1D(nn.Module):

    def __init__(self,in_filter,kernel,stride=1):
        super().__init__()

        # Outfilter is determined by shuffling factor
        shuffle_size = 2
        out_filter = in_filter * shuffle_size
        self.block = nn.Sequential(ConvLayer1D(in_filter,in_filter * 2 ,kernel,stride),
                                   PixelShuffler(),
                                   # would be outfilter * 2 but shuffled by 2
                                   nn.InstanceNorm1d(num_features=out_filter * 2 ,eps=1e-6),
                                   nn.GLU(dim=1))

    def forward(self, input):
        return self.block(input)


class Generator(nn.Module):

    # TODO: Maybe be a bit more explicit about shuffling size?
    def __init__(self, in_feature):
        super(Generator,self).__init__()
        self.block = nn.Sequential(ConvLayer1D(in_filter=in_feature, out_filter=128 * 2, kernel=15,stride=1),
                                   nn.GLU(dim=1),

                                   Downsample1D(in_filter=128, out_filter=256,kernel=5,stride=2),
                                   Downsample1D(in_filter=256, out_filter=512,kernel=5,stride=2),

        # This is eqiuvalent to the 1024, but it's kind of difficult to explain
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

    def forward(self,input):
        return self.block(input)


class Debugger(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self,input):

        print("debugger",input.shape)
        return input

class Discriminator(nn.Module):

    def __init__(self, in_feature=1):
        super().__init__()

        self.block = nn.Sequential(
            ConvLayer2D(in_filter=in_feature,out_filter=256,kernel=(3,3), stride = (1,2)),
            nn.GLU(dim=1),

            # 128->256  24->11 128->63
            Downsample2D(in_filter=128, out_filter=256, kernel=(3,3), stride=(2,2)),
            Downsample2D(in_filter=256, out_filter=512, kernel=(3, 3), stride=(2, 2)),

            Downsample2D(in_filter=512, out_filter=1024, kernel=(6, 3), stride=(1, 2)),
            nn.Flatten(),
            nn.Linear(1024 * 6 * 8 ,1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.block(input)



if __name__ == '__main__':

    #print(ResidualBlock1D(1024)(torch.ones((10,1024,20))))

    #print(Downsample1D(10,10,3,1)(torch.ones((10,10,10))))

    # Length should be divisible by 4
    print(Generator(24)(torch.ones(10,24,400)))
    #print(padding_utility(6,3,1,6)) # 7,
    #print(padding_utility(16,1,2,3))
    print(Discriminator(1)(torch.ones(10,1,24,128)))

    # 10 x 512 x 6 x 16

    # gt: 10 x 6 x 8 x 1024
    #print(PixelShuffler()(PixelShuffler()(torch.ones(10,8,8))).shape)
    #print(Upsample1D(16,16,3)(torch.ones(10,16,4)).shape)
    #print(ConvLayer1D(10,40,3)(torch.ones(size=(10,10,10))))
