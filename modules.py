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

        feats = []
        feat = x
        feats.append(feat)

        for layer_id, layer in enumerate(self.block):
            feat = layer(feat)
            # Basically: Input (before), Conv layer, First ResNet, middle ResNet. Original paper also has second conv
            # at the beginning, for now, we don't have that
            if layer_id in [0,6,8]:
                feats.append(feat)

        out = feat

        return feats, out

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

import torch.nn as nn
from torch.functional import F


class Modern_DBLSTM_1(nn.Module):
    """
    DBLSTM implementation
    """

    def __init__(self, input_size, hidden_size, hidden_size_2, num_classes):
        super(Modern_DBLSTM_1, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # You could do more layers here, completely optional
        self.num_layers = 2
        self.hidden_dim = hidden_size
        self.lstm1 = nn.LSTM(hidden_size, hidden_size_2, bidirectional=True, num_layers=self.num_layers,
                             batch_first=True)
        # You had to pay attention to the directionality
        # and the last linear layer is important
        self.fc3 = nn.Linear(hidden_size_2 * 2, num_classes)

    def forward(self, x, mask=None):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))

        out, hidden = self.lstm1(out)

        out = self.fc3(out)
        return out


class PatchNCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        # bool because we have torch 1.5.0
        self.mask_dtype = torch.bool
        self.nce_t = 0.07 # TODO: HARDCODED HIDDEN TEMPERATURE

    def forward(self, feat_q, feat_k):
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        batch_dim_for_bmm = 1

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.nce_t

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


class PatchSampleF(nn.Module):
    """

    Sampling and projection to the embedding space
    How do sample patches
    """
    def __init__(self, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.nc = nc  # hard-coded

        self.gpu_ids = gpu_ids

        self.mlp = nn.Sequential(*[nn.Linear(1, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
        self.mlp.cuda()
        #self.mlp = None

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []

        for i, feat in enumerate(feats):
            # stepping through each feature-levels
            feat = feat.unsqueeze(0)
            # we deal with an  (N x C x H x W)
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)

            #print(feat_reshape.shape)
            # (N x C x H x W) -> (N x H x W x C) -> (N x HW x C)
            hw = feat_reshape.shape[1]
            # we then compress it to

            if patch_ids is None:
                patch_id = torch.randperm(hw)
                patch_id = patch_id[:num_patches]  # .to(patch_ids.device)
            else:
                patch_id = patch_ids[i]

            x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])

            x_sample = self.mlp(x_sample)

            x_sample = self.l2norm(x_sample)

            return_feats.append(x_sample)
            return_ids.append(patch_id)
        return return_feats, return_ids


    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)

        # feat_q generator features
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        # lets assume no equivariance and other fireworks

        feat_k = self.netG(src, self.nce_layers, encode_only=True)



        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers



if __name__ == '__main__':
    from PIL import Image
    from torchvision import transforms
    import matplotlib.pyplot as plt
    image = Image.open("example.jpg")

    #print(image)
    pil_to_tensor = transforms.ToTensor()(image)
    #image = torch.tensor(image)

    #print(pil_to_tensor.shape)
    #plt.imshow(pil_to_tensor[0,:,:].squeeze(0))
    #plt.show()

    print(pil_to_tensor.shape)

    #( I x N x C x H x W)
    feats = pil_to_tensor.unsqueeze(0).unsqueeze(0)
    print(feats.shape)
    #feats = torch.rand((10,10,10,10,10))
    out = PatchSampleF()(feats)
    #print(out)
