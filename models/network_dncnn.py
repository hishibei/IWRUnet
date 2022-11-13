import torch
import torch.nn as nn
import models.basicblock as B
import numpy as np
from torch.autograd import Variable

import torch.nn.functional as F

"""
# --------------------------------------------
# DnCNN (20 conv layers)
# FDnCNN (20 conv layers)
# IRCNN (7 conv layers)
# --------------------------------------------
# References:
@article{zhang2017beyond,
  title={Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={26},
  number={7},
  pages={3142--3155},
  year={2017},
  publisher={IEEE}
}
@article{zhang2018ffdnet,
  title={FFDNet: Toward a fast and flexible solution for CNN-based image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={27},
  number={9},
  pages={4608--4622},
  year={2018},
  publisher={IEEE}
}
# --------------------------------------------
"""







# # 4层的网络
# class Double(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""

#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)


# class Down(nn.Module):
#     """Downscaling with maxpool then double conv"""

#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             Double(in_channels, out_channels)
#         )

#     def forward(self, x):
#         return self.maxpool_conv(x)


# class Up(nn.Module):
#     """Upscaling then double conv"""

#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super().__init__()

#         # if bilinear, use the normal convolutions to reduce the number of channels
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.conv = Double(in_channels, out_channels, in_channels // 2)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv = Double(in_channels, out_channels)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         # input is CHW
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]

#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
#         # if you have padding issues, see
#         # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
#         # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)


# class Out(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Out, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         return self.conv(x)

# class UNet(nn.Module):
#     def __init__(self, n_channels = 3 , n_classes = 3 , bilinear=False):
#         super(UNet, self).__init__()
#         self.n_channels = 3
#         self.n_classes = 3
#         self.bilinear = bilinear

#         self.inc = Double(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         factor = 2 if bilinear else 1
#         self.down4 = Down(512, 1024 // factor)
#         self.up1 = Up(1024, 512 // factor, bilinear)
#         self.up2 = Up(512, 256 // factor, bilinear)
#         self.up3 = Up(256, 128 // factor, bilinear)
#         self.up4 = Up(128, 64, bilinear)
#         self.outc = Out(64, n_classes)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits

# # --------------------------------------------
# # Unet2
# # --------------------------------------------
# class UNet2(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3):
#         """Initializes U-Net."""
#         super(UNet2, self).__init__()
#         self.unet = UNet()
#         self.unet1 = UNet()

#     def forward(self, x):
#         unet = self.unet(x)
#         unet2 = self.unet1(unet)
#         return unet2








# # 无上下采样的网络
# # --------------------------------------------
# # Unet
# # --------------------------------------------
# class UNet(nn.Module):
#     """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

#     def __init__(self, in_channels=3, out_channels=3):
#         """Initializes U-Net."""
#         super(UNet, self).__init__()

#         # Layers: enc_conv0, enc_conv1, pool1
#         self._block1 = nn.Sequential(
#             nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(48, 48, 3, padding=1),
#             nn.ReLU(inplace=True),
#             # nn.MaxPool2d(2))
#         )

#         # Layers: enc_conv(i), pool(i); i=2..5
#         self._block2 = nn.Sequential(
#             nn.Conv2d(48, 48, 3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             # nn.MaxPool2d(2))
#         )

#         # 增大一层感受野
#         self._block2_2 = nn.Sequential(
#             nn.Conv2d(48, 48, 3, stride=1, padding=6, dilation=6),
#             nn.ReLU(inplace=True),
#             # nn.MaxPool2d(2))
#         )

#         # Layers: enc_conv6, upsample5
#         self._block3 = nn.Sequential(
#             nn.Conv2d(48, 48, 3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             # nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))
#             # nn.Upsample(scale_factor=2, mode='nearest'))
#         )

#         # Layers: dec_conv5a, dec_conv5b, upsample4
#         self._block4_1 = nn.Sequential(
#             nn.Conv2d(96, 96, 3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(96, 96, 3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             # nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
#         )

#         # Layers: dec_conv5a, dec_conv5b, upsample4
#         self._block4 = nn.Sequential(
#             nn.Conv2d(144, 96, 3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(96, 96, 3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             # nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
#             # nn.Upsample(scale_factor=2, mode='nearest'))
#         )

#         # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
#         self._block5 = nn.Sequential(
#             nn.Conv2d(144, 96, 3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(96, 96, 3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             # nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
#             # nn.Upsample(scale_factor=2, mode='nearest'))
#         )

#         # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
#         self._block6 = nn.Sequential(
#             nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 32, 3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
#             nn.LeakyReLU(0.1))

#         # Initialize weights
#         self._init_weights()

#     def _init_weights(self):
#         """Initializes weights using He et al. (2015)."""
#         for m in self.modules():
#             if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight.data)
#                 m.bias.data.zero_()

#     def forward(self, x):
#         """Through encoder, then decoder by adding U-skip connections. """

#         # Encoder
#         pool1 = self._block1(x)
#         # print('==pool1.shape:', pool1.shape)
#         pool2 = self._block2(pool1)
#         # print('==pool2.shape:', pool2.shape)
#         pool3 = self._block2(pool2)
#         # print('==pool3.shape:', pool3.shape)
#         pool4 = self._block2(pool3)
#         # print('==pool4.shape:', pool4.shape)
#         pool5 = self._block2_2(pool4)
#         # print('==pool5.shape:', pool5.shape)
#         pool6 = self._block2_2(pool5)
#         # print('==pool6.shape:', pool6.shape)

#         # Decoder
#         upsample6 = self._block3(pool6)
#         # print('==upsample6.shape:', upsample6.shape)
#         concat6 = torch.cat((upsample6, pool5), dim=1)
#         # print('==concat6.shape', concat6.shape)
#         upsample5 = self._block4_1(concat6)
#         # print('==upsample5.shape:', upsample5.shape)
#         concat5 = torch.cat((upsample5, pool4), dim=1)
#         upsample4 = self._block4(concat5)
#         # print('==upsample4.shape:', upsample4.shape)
#         concat4 = torch.cat((upsample4, pool3), dim=1)
#         upsample3 = self._block5(concat4)
#         # print('==upsample3.shape:', upsample3.shape)
#         concat3 = torch.cat((upsample3, pool2), dim=1)
#         upsample2 = self._block5(concat3)
#         # print('==upsample2.shape:', upsample2.shape)
#         concat2 = torch.cat((upsample2, pool1), dim=1)
#         upsample1 = self._block5(concat2)
#         # print('==upsample1.shape:', upsample1.shape)
#         concat1 = torch.cat((upsample1, x), dim=1)

#         # Final activation
#         return self._block6(concat1)


# # --------------------------------------------
# # Unet2
# # --------------------------------------------
# class UNet2(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3):
#         """Initializes U-Net."""
#         super(UNet2, self).__init__()
#         self.unet = UNet()
#         self.unet1 = UNet()

#     def forward(self, x):
#         unet = self.unet(x)

#         unet2 = self.unet1(unet)

#         return unet2


















# 原版网络结构
# --------------------------------------------
# Unet
# --------------------------------------------
class UNet(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

    def __init__(self, in_channels=3, out_channels=3):
        """Initializes U-Net."""
        super(UNet, self).__init__()

        # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        #增大一层感受野
        self._block2_2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv6, upsample5
        self._block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block4_1 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block4 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block5 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block6 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1))

        # Initialize weights
        self._init_weights()


    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """

        # Encoder
        pool1 = self._block1(x)
        # print('==pool1.shape:', pool1.shape)
        pool2 = self._block2(pool1)
        # print('==pool2.shape:', pool2.shape)
        pool3 = self._block2(pool2)
        # print('==pool3.shape:', pool3.shape)
        pool4 = self._block2(pool3)
        # print('==pool4.shape:', pool4.shape)
        pool5 = self._block2_2(pool4)
        # print('==pool5.shape:', pool5.shape)
        pool6 = self._block2_2(pool5)
        # print('==pool6.shape:', pool6.shape)

        # Decoder
        upsample6 = self._block3(pool6)
        # print('==upsample6.shape:', upsample6.shape)
        concat6 = torch.cat((upsample6, pool5), dim=1)
        # print('==concat6.shape', concat6.shape)
        upsample5 = self._block4_1(concat6)
        # print('==upsample5.shape:', upsample5.shape)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self._block4(concat5)
        # print('==upsample4.shape:', upsample4.shape)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self._block5(concat4)
        # print('==upsample3.shape:', upsample3.shape)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block5(concat3)
        # print('==upsample2.shape:', upsample2.shape)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block5(concat2)
        # print('==upsample1.shape:', upsample1.shape)
        concat1 = torch.cat((upsample1, x), dim=1)

        # Final activation
        return self._block6(concat1)

# --------------------------------------------
# Unet2
# --------------------------------------------
class UNet2(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        """Initializes U-Net."""
        super(UNet2, self).__init__()
        self.unet = UNet()
        self.unet1 = UNet()

    def forward(self, x):
        # unet = self.unet(x)
        # node = unet + x
        # unet2 = self.unet1(node)
        # result = unet2 + node
        # return result

        unet = self.unet(x)
        unet2 = self.unet1(unet)
        return unet2

















# --------------------------------------------
# DnCNN
# --------------------------------------------
class DnCNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=17, act_mode='BR'):
        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super(DnCNN, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True

        m_head = B.conv(in_nc, nc, mode='C' + act_mode[-1], bias=bias)
        m_body = [B.conv(nc, nc, mode='C' + act_mode, bias=bias) for _ in range(nb - 2)]
        m_tail = B.conv(nc, out_nc, mode='C', bias=bias)

        self.model = B.sequential(m_head, *m_body, m_tail)

    def forward(self, x):
        n = self.model(x)
        return x - n


# --------------------------------------------
# FFDNet
# --------------------------------------------
class FFDNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=15, act_mode='R'):
        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        # ------------------------------------
        """
        super(FFDNet, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True
        sf = 2

        self.m_down = B.PixelUnShuffle(upscale_factor=sf)

        m_head = B.conv(in_nc * sf * sf + 1, nc, mode='C' + act_mode[-1], bias=bias)
        m_body = [B.conv(nc, nc, mode='C' + act_mode, bias=bias) for _ in range(nb - 2)]
        m_tail = B.conv(nc, out_nc * sf * sf, mode='C', bias=bias)

        self.model = B.sequential(m_head, *m_body, m_tail)

        self.m_up = nn.PixelShuffle(upscale_factor=sf)

    def forward(self, x, sigma=0):
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h / 2) * 2 - h)
        paddingRight = int(np.ceil(w / 2) * 2 - w)
        x = torch.nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        sigma = torch.FloatTensor(np.array([sigma for idx in range(x.shape[0])]))
        sigma = Variable(sigma).cuda()

        x = self.m_down(x)
        # m = torch.ones(sigma.size()[0], sigma.size()[1], x.size()[-2], x.size()[-1]).type_as(x).mul(sigma)
        m = sigma.repeat(1, 1, x.size()[-2], x.size()[-1])
        x = torch.cat((x, m), 1)
        x = self.model(x)
        x = self.m_up(x)

        x = x[..., :h, :w]
        return x


# --------------------------------------------
# IRCNN denoiser
# --------------------------------------------
class IRCNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64):
        """
        # ------------------------------------
        denoiser of IRCNN
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super(IRCNN, self).__init__()
        L = []
        L.append(
            nn.Conv2d(in_channels=in_nc, out_channels=nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=2, dilation=2, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=3, dilation=3, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=4, dilation=4, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=3, dilation=3, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=2, dilation=2, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(
            nn.Conv2d(in_channels=nc, out_channels=out_nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        self.model = B.sequential(*L)

    def forward(self, x):
        n = self.model(x)
        return x - n


# --------------------------------------------
# FDnCNN
# --------------------------------------------
# Compared with DnCNN, FDnCNN has three modifications:
# 1) add noise level map as input
# 2) remove residual learning and BN
# 3) train with L1 loss
# may need more training time, but will not reduce the final PSNR too much.
# --------------------------------------------
class FDnCNN(nn.Module):
    def __init__(self, in_nc=2, out_nc=1, nc=64, nb=20, act_mode='R'):
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        """
        super(FDnCNN, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True

        m_head = B.conv(in_nc, nc, mode='C' + act_mode[-1], bias=bias)
        m_body = [B.conv(nc, nc, mode='C' + act_mode, bias=bias) for _ in range(nb - 2)]
        m_tail = B.conv(nc, out_nc, mode='C', bias=bias)

        self.model = B.sequential(m_head, *m_body, m_tail)

    def forward(self, x):
        x = self.model(x)
        return x

# --------------------------------------------
# RDDCNN
# --------------------------------------------

class Deform2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(Deform2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset

class RDDCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=3, use_bnorm=True, kernel_size=3):
        super(RDDCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(Deform2d(inc=image_channels, outc=n_channels, kernel_size=kernel_size, padding=padding, bias=False, modulation=True))
        # layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            if _ == 11:
                layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=2, bias=False, dilation=2))
            else:
                layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            #if _ >= 11:
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        #layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
        #layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                # print('init weight')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        print('init weights finished')



# --------------------------------------------
# ADNet
# --------------------------------------------


class Conv_BN_Relu_first(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding,groups,bias):
        super(Conv_BN_Relu_first,self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        groups =1 
        self.conv = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(features)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))

class Conv_BN_Relu_other(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding,groups,bias):
        super(Conv_BN_Relu_other,self).__init__()
        kernel_size = 3
        padding = 1
        features = out_channels
        groups =1 
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=kernel_size, padding=padding,groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(features)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))


class Conv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding,groups,bais):
        super(Conv,self).__init__()
        kernel_size = 3
        padding = 1
        features = 1
        groups =1 
        self.conv = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,groups=groups, bias=False)
    def forward(self,x):
        return self.conv(x)

class Self_Attn(nn.Module):
    def __init__(self,in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim//8,kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim//8,kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim,kernel_size=1)
        self.gamma=nn.Parameter(torch.zeros(1))
        self.softmax=nn.Softmax(dim=-1)
    def forward(self,x):
        m_batchsize, C, width,height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1)
        proj_key = self.key_conv(x).view(m_batchsize,-1,width*height)
        #print proj_query.size()
        #print proj_key.size()
        energy = torch.bmm(proj_query,proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) 
        out = torch.bmm(proj_value,attention.permute(0,2,1))
        out = out.view(m_batchsize,C,width,height)
        out = self.gamma*out + x
        return out, attention

class ADNet(nn.Module):
    def __init__(self, channels, num_of_layers=15):
        super(ADNet, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        groups =1 
        layers = []
        kernel_size1 = 1
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels=channels,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias=False,dilation=2),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_3 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_4 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_5 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias=False,dilation=2),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_6 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_7 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_8 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_9 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias=False,dilation=2),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_10 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_11 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_12 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias=False,dilation=2),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_13 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_14 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_15 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_16 = nn.Conv2d(in_channels=features,out_channels=3,kernel_size=kernel_size,padding=1,groups=groups,bias=False)
        self.conv3 = nn.Conv2d(in_channels=6,out_channels=3,kernel_size=1,stride=1,padding=0,groups=1,bias=True)
        self.ReLU = nn.ReLU(inplace=True)
        self.Tanh= nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
                clip_b = 0.025
                w = m.weight.data.shape[0]
                for j in range(w):
                    if m.weight.data[j] >= 0 and m.weight.data[j] < clip_b:
                        m.weight.data[j] = clip_b
                    elif m.weight.data[j] > -clip_b and m.weight.data[j] < 0:
                        m.weight.data[j] = -clip_b
                m.running_var.fill_(0.01)
    def _make_layers(self, block,features, kernel_size, num_of_layers, padding=1, groups=1, bias=False):
        layers = []
        for _ in range(num_of_layers):
            layers.append(block(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, groups=groups, bias=bias))
        return nn.Sequential(*layers)
    def forward(self, x):
        input = x 
        x1 = self.conv1_1(x)
        x1 = self.conv1_2(x1)
        x1 = self.conv1_3(x1)
        x1 = self.conv1_4(x1)
        x1 = self.conv1_5(x1)
        x1 = self.conv1_6(x1)
        x1 = self.conv1_7(x1)   
        x1t = self.conv1_8(x1)
        x1 = self.conv1_9(x1t)
        x1 = self.conv1_10(x1)
        x1 = self.conv1_11(x1)
        x1 = self.conv1_12(x1)
        x1 = self.conv1_13(x1)
        x1 = self.conv1_14(x1)
        x1 = self.conv1_15(x1)
        x1 = self.conv1_16(x1)
        out = torch.cat([x,x1],1)
        out= self.Tanh(out)
        out = self.conv3(out)
        out = out*x1
        out2 = x - out
        return out2










# --------------------------------------------
# HDCNN
# --------------------------------------------
def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(
                num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=padding_11, groups=groups)
            print('RepVGG Block, identity = ', self.rbr_identity)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

        #   Optional. This improves the accuracy and facilitates quantization.
        #   1.  Cancel the original weight decay on rbr_dense.conv.weight and rbr_1x1.conv.weight.
        #   2.  Use like this.
        #       loss = criterion(....)
        #       for every RepVGGBlock blk:
        #           loss += weight_decay_coefficient * 0.5 * blk.get_cust_L2()
        #       optimizer.zero_grad()
        #       loss.backward()
        def get_custom_L2(self):
            K3 = self.rbr_dense.conv.weight
            K1 = self.rbr_1x1.conv.weight
            t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(
                -1, 1, 1, 1).detach()
            t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1,
                                                                                                                 1,
                                                                                                                 1).detach()

            l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2,
                                                1:2] ** 2).sum()  # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
            eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1  # The equivalent resultant central point of 3x3 kernel.
            l2_loss_eq_kernel = (eq_kernel ** 2 / (
                    t3 ** 2 + t1 ** 2)).sum()  # Normalize for an L2 coefficient comparable to regular L2.
            return l2_loss_eq_kernel + l2_loss_circle

        #   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
        #   You can get the equivalent kernel and bias at any time and do whatever you want,
        #   for example, apply some penalties or constraints during training, just like you do to the other models.
        #   May be useful for quantization or pruning.
        def get_equivalent_kernel_bias(self):
            kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
            kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
            kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
            return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

        def _pad_1x1_to_3x3_tensor(self, kernel1x1):
            if kernel1x1 is None:
                return 0
            else:
                return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

        def _fuse_bn_tensor(self, branch):
            if branch is None:
                return 0, 0
            if isinstance(branch, nn.Sequential):
                kernel = branch.conv.weight
                running_mean = branch.bn.running_mean
                running_var = branch.bn.running_var
                gamma = branch.bn.weight
                beta = branch.bn.bias
                eps = branch.bn.eps
            else:
                assert isinstance(branch, nn.BatchNorm2d)
                if not hasattr(self, 'id_tensor'):
                    input_dim = self.in_channels // self.groups
                    kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                    for i in range(self.in_channels):
                        kernel_value[i, i % input_dim, 1, 1] = 1
                    self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
                kernel = self.id_tensor
                running_mean = branch.running_mean
                running_var = branch.running_var
                gamma = branch.weight
                beta = branch.bias
                eps = branch.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std

        def switch_to_deploy(self):
            if hasattr(self, 'rbr_reparam'):
                return
            kernel, bias = self.get_equivalent_kernel_bias()
            self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                         out_channels=self.rbr_dense.conv.out_channels,
                                         kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                         padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                         groups=self.rbr_dense.conv.groups, bias=True)
            self.rbr_reparam.weight.data = kernel
            self.rbr_reparam.bias.data = bias
            for para in self.parameters():
                para.detach_()
            self.__delattr__('rbr_dense')
            self.__delattr__('rbr_1x1')
            if hasattr(self, 'rbr_identity'):
                self.__delattr__('rbr_identity')
            if hasattr(self, 'id_tensor'):
                self.__delattr__('id_tensor')
            self.deploy = True


class RepVGG(nn.Module):

    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, override_groups_map=None, deploy=False,
                 use_se=False):
        super(RepVGG, self).__init__()

        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        self.use_se = use_se

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepVGGBlock(in_channels=64, out_channels=self.in_planes, kernel_size=3, stride=1, padding=1,
                                  deploy=self.deploy, use_se=self.use_se)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=1)
        self.stage2 = self._make_stage(int(64 * width_multiplier[1]), num_blocks[1], stride=1)
        self.stage3 = self._make_stage(int(64 * width_multiplier[2]), num_blocks[2], stride=1)
        self.stage4 = self._make_stage(int(64 * width_multiplier[3]), num_blocks[3], stride=1)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)

    def _make_stage(self, planes, num_blocks, stride):
        # strides = [stride] + [1]*(num_blocks-1)
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy,
                                      use_se=self.use_se))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        # print('stage 0: ', out.shape)
        out = self.stage1(out)
        # print('stage 1: ', out.shape)
        out = self.stage2(out)
        # print('stage 2: ', out.shape)
        out = self.stage3(out)
        # print('stage 3: ', out.shape)
        out = self.stage4(out)
        # print('stage 4: ', out.shape)
        return out


class RepVGGCNN(nn.Module):
    def __init__(self, deploy=False):
        super(RepVGGCNN, self).__init__()
        kernel_size3x3 = 3
        kernel_size1x1 = 1
        features = 64
        inchannels = 3
        groups = 1
        padding = 1
        dilation_padding = 2
        dilation = 2
        # SB
        self.convSB1 = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=features, kernel_size=kernel_size3x3, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
        self.convSB2_dilation = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size3x3, padding=dilation_padding,
                      groups=groups,
                      bias=False, dilation=dilation),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
        self.convSB2 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size3x3, padding=padding,
                      groups=groups,
                      bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
        self.convSB3 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size3x3, padding=padding,
                      groups=groups,
                      bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
        self.convSB3_dilation = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size3x3, padding=dilation_padding,
                      groups=groups,
                      bias=False, dilation=dilation),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
        self.convSB4 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size3x3, padding=padding,
                      groups=groups,
                      bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
        self.convSB5_dilation = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size3x3, padding=dilation_padding,
                      groups=groups,
                      bias=False, dilation=dilation),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
        self.convSB5 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size3x3, padding=padding,
                      groups=groups,
                      bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
        self.convSB6 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size3x3, padding=padding,
                      groups=groups,
                      bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
        self.convSB7_dilation = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size3x3, padding=dilation_padding,
                      groups=groups,
                      bias=False, dilation=dilation),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
        self.convSB7 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size3x3, padding=padding,
                      groups=groups,
                      bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
        # ReqVGG
        self.reqVGG = RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                             width_multiplier=[1, 1, 1, 1], override_groups_map=None, deploy=deploy)
        # FEB
        self.convFEB1 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size3x3, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
        self.convFEB2 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size3x3, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
        self.convReLU = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size3x3, padding=padding,
                      groups=groups, bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv_re = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size3x3, padding=padding,
                      groups=groups, bias=False),
            nn.ReLU(inplace=True)
        )
        self.convlast = nn.Conv2d(in_channels=features, out_channels=inchannels, kernel_size=kernel_size3x3,
                                  padding=padding)

    def forward(self, input):
        x = self.convSB1(input)
        x = self.convSB2(x)
        x = self.convSB3_dilation(x)
        x = self.convSB4(x)
        x = self.convSB5(x)
        x = self.convSB6(x)
        x = self.convSB7(x)
        x = self.reqVGG(x)
        x = self.convFEB1(x)
        x = self.convFEB2(x)
        x = self.convReLU(x)
        x = self.conv_re(x)
        x = self.convlast(x)
        return x + input


def repvgg_model_convert(model: torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model





if __name__ == '__main__':
    from utils import utils_model
    import torch

    model1 = DnCNN(in_nc=1, out_nc=1, nc=64, nb=20, act_mode='BR')
    print(utils_model.describe_model(model1))

    model2 = FDnCNN(in_nc=2, out_nc=1, nc=64, nb=20, act_mode='R')
    print(utils_model.describe_model(model2))

    x = torch.randn((1, 1, 240, 240))
    x1 = model1(x)
    print(x1.shape)

    x = torch.randn((1, 2, 240, 240))
    x2 = model2(x)
    print(x2.shape)

    #  run models/network_dncnn.py
