import torch
import torch.nn as nn
from torch.autograd import Variable
from pdb import set_trace as st

###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1 or  classname.find('InstanceNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def define_G(input_nc, output_nc, ngf, which_model_netG, norm, gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    if norm == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm == 'instance':
        norm_layer = InstanceNormalization
    else:
        print('normalization layer [%s] is not found' % norm)
    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer, 6, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer, gpu_ids=gpu_ids)
    else:
        print('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda()
    netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, use_sigmoid=False, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = define_D(input_nc, ndf, 'n_layers', use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, use_sigmoid, gpu_ids=gpu_ids)
    else:
        print('Discriminator model name [%s] is not recognized' %
              which_model_netD)
    if use_gpu:
        netD.cuda()
    netD.apply(weights_init)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss used in LSGAN.
# It is basically same as MSELoss, but it abstracts away the need to create
# the target label tensor that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, n_blocks=6, gpu_ids=[]):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        model = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, 'zero', norm_layer=norm_layer)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.gpu_ids:
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer)

    def build_conv_block(self, dim, padding_type, norm_layer):
        conv_block = []
        p = 0
        # TODO: support padding types
        assert(padding_type == 'zero')
        p = 1

        # TODO: InstanceNorm
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       nn.ReLU(True)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block)
        unet_block = UnetSkipConnectionBlock(input_nc, ngf, unet_block,
                                             outermost=True)

        self.model = unet_block

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.gpu_ids:
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        kw = 4
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=2),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                kernel_size=kw, stride=2, padding=2),
                # TODO: use InstanceNorm
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                            kernel_size=1, stride=2, padding=2),
            # TODO: useInstanceNorm
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=1)]
        sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.gpu_ids:
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

# Instance Normalization layer from
# https://github.com/darkstar112358/fast-neural-style

class InstanceNormalization(torch.nn.Module):
    """InstanceNormalization
    Improves convergence of neural-style.
    ref: https://arxiv.org/pdf/1607.08022.pdf
    """

    def __init__(self, dim, eps=1e-5):
        super(InstanceNormalization, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(dim))
        self.bias = nn.Parameter(torch.FloatTensor(dim))
        self.eps = eps
        self._reset_parameters()

    def _reset_parameters(self):
        self.weight.data.uniform_()
        self.bias.data.zero_()

    def forward(self, x):
        n = x.size(2) * x.size(3)
        t = x.view(x.size(0), x.size(1), n)
        mean = torch.mean(t, 2).unsqueeze(2).expand_as(x)
        # Calculate the biased var. torch.var returns unbiased var
        var = torch.var(t, 2).unsqueeze(2).expand_as(x) * ((n - 1) / float(n))
        scale_broadcast = self.weight.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        scale_broadcast = scale_broadcast.expand_as(x)
        shift_broadcast = self.bias.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        shift_broadcast = shift_broadcast.expand_as(x)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = out * scale_broadcast + shift_broadcast
        return out




import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.utils import model_zoo
from torchvision import models

class FCN8(nn.Module):

    def __init__(self, num_classes,gpu_ids):
        super(FCN8, self).__init__()
        self.gpu_ids=gpu_ids
        feats = list(models.vgg16(pretrained=True).features.children())

        self.feats = nn.Sequential(*feats[0:9])
        self.feat3 = nn.Sequential(*feats[10:16])
        self.feat4 = nn.Sequential(*feats[17:23])
        self.feat5 = nn.Sequential(*feats[24:30])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.requires_grad = False

        self.fconn = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.score_feat3 = nn.Conv2d(256, num_classes, 1)
        self.score_feat4 = nn.Conv2d(512, num_classes, 1)
        self.score_fconn = nn.Conv2d(4096, num_classes, 1)

    def forward(self, x):
        feats = self.feats(x)
        feat3 = self.feat3(feats)
        feat4 = self.feat4(feat3)
        feat5 = self.feat5(feat4)
        fconn = self.fconn(feat5)

        score_feat3 = self.score_feat3(feat3)
        score_feat4 = self.score_feat4(feat4)
        score_fconn = self.score_fconn(fconn)

        score = F.upsample_bilinear(score_fconn, score_feat4.size()[2:])
        score += score_feat4
        score = F.upsample_bilinear(score, score_feat3.size()[2:])
        score += score_feat3

        return F.upsample_bilinear(score, x.size()[2:])


class FCN32(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.feats = models.vgg16(pretrained=True).features
        self.fconn = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.score = nn.Conv2d(4096, num_classes, 1)

    def forward(self, x):
        feats = self.feats(x)
        fconn = self.fconn(feats)
        score = self.score(fconn)

        return F.upsample_bilinear(score, x.size()[2:])


class UNetEnc(nn.Module):

    def __init__(self, in_channels, features, out_channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.Conv2d(in_channels, features, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(features, out_channels, 2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class UNetDec(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers += [nn.Dropout(.5)]
        layers += [nn.MaxPool2d(2, stride=2, ceil_mode=True)]

        self.down = nn.Sequential(*layers)

    def forward(self, x):
        return self.down(x)


class UNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.dec1 = UNetDec(3, 64)
        self.dec2 = UNetDec(64, 128)
        self.dec3 = UNetDec(128, 256)
        self.dec4 = UNetDec(256, 512, dropout=True)
        self.center = nn.Sequential(
            nn.Conv2d(512, 1024, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.ConvTranspose2d(1024, 512, 2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.enc4 = UNetEnc(1024, 512, 256)
        self.enc3 = UNetEnc(512, 256, 128)
        self.enc2 = UNetEnc(256, 128, 64)
        self.enc1 = nn.Sequential(
            nn.Conv2d(128, 64, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        dec1 = self.dec1(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)
        center = self.center(dec4)
        enc4 = self.enc4(torch.cat([
            center, F.upsample_bilinear(dec4, center.size()[2:])], 1))
        enc3 = self.up3(torch.cat([
            enc4, F.upsample_bilinear(dec3, enc4.size()[2:])], 1))
        enc2 = self.up2(torch.cat([
            enc3, F.upsample_bilinear(dec2, enc3.size()[2:])], 1))
        enc1 = self.up1(torch.cat([
            enc2, F.upsample_bilinear(dec1, enc2.size()[2:])], 1))

        return F.upsample_bilinear(self.final(enc1), x.size()[2:])


class SegNetEnc(nn.Module):

    def __init__(self, in_channels, out_channels, num_layers):
        super().__init__()

        layers = [
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        ]
        layers += [
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        ] * num_layers
        layers += [
            nn.Conv2d(in_channels // 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class SegNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        # should be vgg16bn but at the moment we have no pretrained bn models
        decoders = list(models.vgg16(pretrained=True).features.children())

        self.dec1 = nn.Sequential(*decoders[:5])
        self.dec2 = nn.Sequential(*decoders[5:10])
        self.dec3 = nn.Sequential(*decoders[10:17])
        self.dec4 = nn.Sequential(*decoders[17:24])
        self.dec5 = nn.Sequential(*decoders[24:])

        # gives better results
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.requires_grad = False

        self.enc5 = SegNetEnc(512, 512, 1)
        self.enc4 = SegNetEnc(1024, 256, 1)
        self.enc3 = SegNetEnc(512, 128, 1)
        self.enc2 = SegNetEnc(256, 64, 0)
        self.enc1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, 3, padding=1)

    def forward(self, x):
        dec1 = self.dec1(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)
        dec5 = self.dec5(dec4)
        enc5 = self.enc5(dec5)
        enc4 = self.enc4(torch.cat([dec4, enc5], 1))
        enc3 = self.enc3(torch.cat([dec3, enc4], 1))
        enc2 = self.enc2(torch.cat([dec2, enc3], 1))
        enc1 = self.enc1(torch.cat([dec1, enc2], 1))

        return F.upsample_bilinear(self.final(enc1), x.size()[2:])


class PSPDec(nn.Module):

    def __init__(self, in_features, out_features, downsize, upsize=60):
        super().__init__()

        self.features = nn.Sequential(
            nn.AvgPool2d(downsize, stride=downsize),
            nn.Conv2d(in_features, out_features, 1, bias=False),
            nn.BatchNorm2d(out_features, momentum=.95),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(upsize)
        )

    def forward(self, x):
        return self.features(x)


class PSPNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        '''
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=.95),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        '''

        resnet = models.resnet101(pretrained=True)

        self.conv1 = resnet.conv1
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.stride = 1
                m.requires_grad = False
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad = False

        self.layer5a = PSPDec(2048, 512, 60)
        self.layer5b = PSPDec(2048, 512, 30)
        self.layer5c = PSPDec(2048, 512, 20)
        self.layer5d = PSPDec(2048, 512, 10)

        self.final = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(.1),
            nn.Conv2d(512, num_classes, 1),
        )

    def forward(self, x):
        print('x', x.size())
        x = self.conv1(x)
        print('conv1', x.size())
        x = self.layer1(x)
        print('layer1', x.size())
        x = self.layer2(x)
        print('layer2', x.size())
        x = self.layer3(x)
        print('layer3', x.size())
        x = self.layer4(x)
        print('layer4', x.size())
        x = self.final(torch.cat([
            x,
            self.layer5a(x),
            self.layer5b(x),
            self.layer5c(x),
            self.layer5d(x),
        ], 1))
        print('final', x.size())

        return F.upsample_bilinear(final, x.size()[2:])


def network_classification_model(type,gpu_ids,feats=None,feat4=None, feat5=None,fconn=None, score_fconn=None,score_feat4=None,num_classes=22):
    if feats== None:
        return FCN16(num_classes,gpu_ids).cuda()
    elif feats is not None and fconn is None:
        return FCN16Modified(num_classes,gpu_ids,feats,feat4,feat5).cuda()
    else:
        return FCN16ModifiedS(num_classes,gpu_ids,feats,feat4,feat5,fconn,score_fconn,score_feat4)

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.utils import model_zoo
from torchvision import models

class FCN16(nn.Module):

    def __init__(self, num_classes,gpu_ids):
        super(FCN16,self).__init__()
        self.gpu_ids=gpu_ids
        feats = list(models.vgg16(pretrained=True).features.children())
        self.feats = nn.Sequential(*feats[0:16])
        self.feat4 = nn.Sequential(*feats[17:23])
        st
        self.feat5 = nn.Sequential(*feats[24:30])
        self.fconn = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.score_fconn = nn.Conv2d(4096, num_classes, 1)
        self.score_feat4 = nn.Conv2d(512, num_classes, 1)

    def forward(self, x):
        feats = self.feats(x)
        feat4 = self.feat4(feats)
        feat5 = self.feat5(feat4)
        fconn = self.fconn(feat5)

        score_feat4 = self.score_feat4(feat4)
        score_fconn = self.score_fconn(fconn)

        score = F.upsample_bilinear(score_fconn, score_feat4.size()[2:])
        score += score_feat4

        return F.upsample_bilinear(score, x.size()[2:])


class FCN16Modified(nn.Module):

    def __init__(self, num_classes,gpu_ids,feats,feat4,feat5):
        super(FCN16Modified,self).__init__()
        self.gpu_ids=gpu_ids
        #feats = list(models.vgg16(pretrained=True).features.children())
        self.feats = nn.Sequential(*feats)
        self.feat4 = nn.Sequential(*feat4)
        self.feat5 = nn.Sequential(*feat5)
        self.fconn = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.score_fconn = nn.Conv2d(4096, num_classes, 1)
        self.score_feat4 = nn.Conv2d(512, num_classes, 1)

    def forward(self, x):
        feats = self.feats(x)
        feat4 = self.feat4(feats)
        feat5 = self.feat5(feat4)
        fconn = self.fconn(feat5)

        score_feat4 = self.score_feat4(feat4)
        score_fconn = self.score_fconn(fconn)

        score = F.upsample_bilinear(score_fconn, score_feat4.size()[2:])
        score += score_feat4

        return F.upsample_bilinear(score, x.size()[2:])

class FCN16ModifiedS(nn.Module):

    def __init__(self, num_classes,gpu_ids,feats,feat4,feat5,fconn,score_fconn,score_feat4):
        super(FCN16ModifiedS,self).__init__()
        self.gpu_ids=gpu_ids
        #feats = list(models.vgg16(pretrained=True).features.children())
        self.feats = nn.Sequential(*feats)
        self.feat4 = nn.Sequential(*feat4)
        self.feat5 = nn.Sequential(*feat5)
        self.fconn = nn.Sequential(*fconn)
        self.score_fconn = score_fconn
        self.score_feat4 = score_feat4

    def forward(self, x):
        feats = self.feats(x)
        feat4 = self.feat4(feats)
        feat5 = self.feat5(feat4)
        fconn = self.fconn(feat5)

        score_feat4 = self.score_feat4(feat4)
        score_fconn = self.score_fconn(fconn)

        score = F.upsample_bilinear(score_fconn, score_feat4.size()[2:])
        score += score_feat4

        return F.upsample_bilinear(score, x.size()[2:])