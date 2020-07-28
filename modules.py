
import torch.nn as nn
import torch
import math
from torch.nn.functional import glu
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

    def forward(self, x):
        out = self.conv(x)

        if (self.cut_last_element_X) & (len(x.shape) == 4):
            out = out[:,:,:-1,:]
        if (self.cut_last_element_Y) & (len(x.shape) == 4):
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


class PixelShuffler(nn.Module):

    def __init__(self,shuffle_size=2):
        super().__init__()
        self.shuffle_size = shuffle_size

    def forward(self, x):
        # TODO: Check notation order. I think tensorflow has (n,w,c), we have (n,c,w)
        n,c,w = x.size()
        ow = c // self.shuffle_size
        oc = w * self.shuffle_size

        return torch.reshape(x, shape = [n,ow,oc])


class Upsample1D(nn.Module):

    def __init__(self,in_filter,kernel,stride=1):
        super().__init__()

        # Outfilter is determined by shuffling factor
        shuffle_size = 2
        out_filter = in_filter * shuffle_size
        self.block = nn.Sequential(ConvLayer1D(in_filter,in_filter * 2 ,kernel,stride),
                                   PixelShuffler(),
                                   # would be outfilter * 2 but shuffled by 2
                                   nn.InstanceNorm1d(num_features=out_filter // 2 ,eps=1e-6, affine=True),
                                   nn.GLU(dim=1))

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad1d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad1d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv1d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad1d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad1d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv1d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm1d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        use_bias=True
        #if type(norm_layer) == functools.partial:
        #    use_bias = norm_layer.func == nn.InstanceNorm2d
        #else:
        #    use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad1d(3),
                 nn.Conv1d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv1d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose1d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad1d(3)]
        model += [nn.Conv1d(ngf, output_nc, kernel_size=7, padding=0)]
        #model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)






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
                                   #ResidualBlock1D(in_filter=512, out_filter=512, kernel=3),
                                   #ResidualBlock1D(in_filter=512, out_filter=512, kernel=3),
                                   #ResidualBlock1D(in_filter=512, out_filter=512, kernel=3),
                                   #ResidualBlock1D(in_filter=512, out_filter=512, kernel=3),
                                   #ResidualBlock1D(in_filter=512, out_filter=512, kernel=3),

                                   Upsample1D(in_filter=512,kernel=5),

                                   Upsample1D(in_filter=256,kernel=5),
                                   ConvLayer1D(in_filter=128, out_filter=in_feature, kernel=5,stride=1),
                                   )

    def forward(self, x):
        return self.block(x)


class Debugger(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):

        print("debugger",x.shape)
        return x


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
            #Debugger(),
            PermuteBlock(),
            # input?: (1 x 1024)
            nn.Linear(1024, 1)
            #nn.Sigmoid()
        )

    def forward(self, x):
        return self.block(x)



if __name__ == '__main__':

    #print(ResidualBlock1D(1024)(torch.ones((10,1024,20))))

    #print(Downsample1D(10,10,3,1)(torch.ones((10,10,10))))

    # Length should be divisible by 4
    #print(Generator(24)(torch.ones(1,24,128)).shape)

    gen = ResnetGenerator(24, 24, ngf = 64, norm_layer = nn.BatchNorm1d, use_dropout = False, n_blocks = 6, padding_type = 'reflect')
    print(gen(torch.ones(1, 24, 128)).shape)
    #print(padding_utility(6,3,1,6)) # 7,
    #print(padding_utility(16,1,2,3))
    #print(Discriminator(1)(torch.ones(1,1,24,128)).shape)

    # 10 x 512 x 6 x 16

    # gt: 10 x 6 x 8 x 1024
    #print(PixelShuffler()(PixelShuffler()(torch.ones(10,8,8))).shape)
    #print(Upsample1D(16,16,3)(torch.ones(10,16,4)).shape)
    #print(ConvLayer1D(10,40,3)(torch.ones(size=(10,10,10))))
