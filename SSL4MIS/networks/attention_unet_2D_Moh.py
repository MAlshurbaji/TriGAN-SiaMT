import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# This one as ChatGPT designed
class AttentionUNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(AttentionUNet2D, self).__init__()

        # Encoder
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(), nn.Conv2d(512, 512, 3, padding=1), nn.ReLU())
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.conv5 = nn.Sequential(nn.Conv2d(512, 1024, 3, padding=1), nn.ReLU(), nn.Conv2d(1024, 1024, 3, padding=1), nn.ReLU())

        # Decoder with Attention
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.att6 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.conv6 = nn.Sequential(nn.Conv2d(1024, 512, 3, padding=1), nn.ReLU(), nn.Conv2d(512, 512, 3, padding=1), nn.ReLU())

        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att7 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.conv7 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())

        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att8 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.conv8 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())

        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att9 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.conv9 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        c5 = self.conv5(p4)

        u6 = self.up6(c5)
        a6 = self.att6(g=u6, x=c4)
        c6 = self.conv6(torch.cat([u6, a6], dim=1))

        u7 = self.up7(c6)
        a7 = self.att7(g=u7, x=c3)
        c7 = self.conv7(torch.cat([u7, a7], dim=1))

        u8 = self.up8(c7)
        a8 = self.att8(g=u8, x=c2)
        c8 = self.conv8(torch.cat([u8, a8], dim=1))

        u9 = self.up9(c8)
        a9 = self.att9(g=u9, x=c1)
        c9 = self.conv9(torch.cat([u9, a9], dim=1))

        output = self.final(c9)
        return output



import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch.nn.functional as F
from networks.utils import unetConv2, unetUp, UnetGatingSignal3, UnetDsv3
from networks.networks_other import init_weights
from networks.grid_attention_layer import GridAttentionBlock2D

class AttentionUNet2DOrig(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3,
                 nonlocal_mode='concatenation', attention_dsample=(2,2), is_batchnorm=True):
        super(AttentionUNet2DOrig, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # Downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        # Bottleneck
        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.gating = UnetGatingSignal2D(filters[4], filters[4], is_batchnorm=self.is_batchnorm)

        # Attention Mechanism
        self.attentionblock2 = MultiAttentionBlock_2D(in_size=filters[1], gate_size=filters[2], inter_size=filters[1],
                                                      nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        self.attentionblock3 = MultiAttentionBlock_2D(in_size=filters[2], gate_size=filters[3], inter_size=filters[2],
                                                      nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        self.attentionblock4 = MultiAttentionBlock_2D(in_size=filters[3], gate_size=filters[4], inter_size=filters[3],
                                                      nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)

        # Upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], is_deconv)

        # Deep Supervision
        self.dsv4 = UnetDsv2D(in_size=filters[3], out_size=n_classes, scale_factor=8)
        self.dsv3 = UnetDsv2D(in_size=filters[2], out_size=n_classes, scale_factor=4)
        self.dsv2 = UnetDsv2D(in_size=filters[1], out_size=n_classes, scale_factor=2)
        self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=n_classes, kernel_size=1)

        # Final Convolution (without concat)
        self.final = nn.Conv2d(n_classes * 4, n_classes, 1)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # Feature Extraction
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)
        gating = self.gating(center)

        # Attention Mechanism
        g_conv4, att4 = self.attentionblock4(conv4, gating)
        up4 = self.up_concat4(g_conv4, center)
        g_conv3, att3 = self.attentionblock3(conv3, up4)
        up3 = self.up_concat3(g_conv3, up4)
        g_conv2, att2 = self.attentionblock2(conv2, up3)
        up2 = self.up_concat2(g_conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        final = self.final(torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1))

        return final


class MultiAttentionBlock_2D(nn.Module):
    def __init__(self, in_size, gate_size, inter_size, nonlocal_mode, sub_sample_factor):
        super(MultiAttentionBlock_2D, self).__init__()
        self.gate_block_1 = GridAttentionBlock2D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor=sub_sample_factor)
        self.gate_block_2 = GridAttentionBlock2D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor=sub_sample_factor)
        self.combine_gates = nn.Sequential(nn.Conv2d(in_size * 2, in_size, kernel_size=1, stride=1, padding=0),
                                           nn.BatchNorm2d(in_size),
                                           nn.ReLU(inplace=True))

        # Initialize the blocks
        for m in self.children():
            if m.__class__.__name__.find('GridAttentionBlock2D') != -1:
                continue
            init_weights(m, init_type='kaiming')

    def forward(self, input, gating_signal):
        gate_1, attention_1 = self.gate_block_1(input, gating_signal)
        gate_2, attention_2 = self.gate_block_2(input, gating_signal)
        return self.combine_gates(torch.cat([gate_1, gate_2], 1)), torch.cat([attention_1, attention_2], 1)



class UnetGatingSignal2D(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True):
        super(UnetGatingSignal2D, self).__init__()
        
        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_size),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(output_size=(4, 4))  # 2D pooling
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(output_size=(4, 4))
            )

        # Initialize weights
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        return self.conv1(inputs)

class UnetDsv2D(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv2D, self).__init__()
        self.dsv = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)  # Use 2D upsampling
        )

    def forward(self, input):
        return self.dsv(input)
