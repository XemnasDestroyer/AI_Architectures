import torch
import torch.nn as nn

from unet_parts import DoubleConv2D, DoubleConv3D, DownSample2D, DownSample3D, UpSample2D, UpSample3D


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_convolution_1 = DownSample2D(in_channels, 64)
        self.down_convolution_2 = DownSample2D(64, 128)
        self.down_convolution_3 = DownSample2D(128, 256)
        self.down_convolution_4 = DownSample2D(256, 512)

        self.bottle_neck = DoubleConv2D(512, 1024)

        self.up_convolution_1 = UpSample2D(1024, 512)
        self.up_convolution_2 = UpSample2D(512, 256)
        self.up_convolution_3 = UpSample2D(256, 128)
        self.up_convolution_4 = UpSample2D(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
       down_1, p1 = self.down_convolution_1(x)
       down_2, p2 = self.down_convolution_2(p1)
       down_3, p3 = self.down_convolution_3(p2)
       down_4, p4 = self.down_convolution_4(p3)

       b = self.bottle_neck(p4)

       up_1 = self.up_convolution_1(b, down_4)
       up_2 = self.up_convolution_2(up_1, down_3)
       up_3 = self.up_convolution_3(up_2, down_2)
       up_4 = self.up_convolution_4(up_3, down_1)

       out = self.out(up_4)
       return out
    
class UNet3D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # Bajada (Encoder)
        self.down_1 = DownSample3D(in_channels, 64)
        self.down_2 = DownSample3D(64, 128)
        self.down_3 = DownSample3D(128, 256)
        self.down_4 = DownSample3D(256, 512)

        # Base (Bottleneck)
        self.bottleneck = DoubleConv3D(512, 1024)

        # Subida (Decoder)
        self.up_1 = UpSample3D(1024, 512)
        self.up_2 = UpSample3D(512, 256)
        self.up_3 = UpSample3D(256, 128)
        self.up_4 = UpSample3D(128, 64)

        # Salida final
        self.out = nn.Conv3d(64, num_classes, kernel_size=1)

    def forward(self, x):
        d1, p1 = self.down_1(x)
        d2, p2 = self.down_2(p1)
        d3, p3 = self.down_3(p2)
        d4, p4 = self.down_4(p3)

        bn = self.bottleneck(p4)

        u1 = self.up_1(bn, d4)
        u2 = self.up_2(u1, d3)
        u3 = self.up_3(u2, d2)
        u4 = self.up_4(u3, d1)

        return self.out(u4)
