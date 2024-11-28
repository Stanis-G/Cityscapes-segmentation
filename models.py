import torch
from torch import nn
from torch.nn.functional import leaky_relu


class ConvCell(nn.Module):
    
    def __init__(self, in_channels, out_channels, conv_kernel_size, padding, pool_kernel_size, pool_stride, negative_slope):
        super().__init__()
        self.negative_slope = negative_slope
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel_size, padding=padding)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='leaky_relu', a=self.negative_slope)
        self.maxpool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = leaky_relu(x, negative_slope=self.negative_slope)
        x = self.maxpool(x)
        x = self.batchnorm(x)
        return x
    

class TransposeConvCell(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, negative_slope, is_linear=False):
        super().__init__()
        self.is_linear = is_linear
        self.negative_slope = negative_slope if not self.is_linear else 0
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        nn.init.kaiming_normal_(self.deconv.weight, nonlinearity='leaky_relu', a=self.negative_slope)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.deconv(x)
        if self.is_linear:
            x = leaky_relu(x, negative_slope=self.negative_slope)
        x = self.batchnorm(x)
        return x
        

class Encoder(nn.Module):
    
    def __init__(self, negative_slope):
        super().__init__()
        self.convcell1 = ConvCell(3, 10, (3,3), 'same', (2,2), 2, negative_slope)
        self.convcell2 = ConvCell(10, 20, (3,3), 'same', (2,2), 2, negative_slope)
        self.convcell3 = ConvCell(20, 40, (3,3), 'same', (2,2), 2, negative_slope)
        self.convcell4 = ConvCell(40, 60, (3,3), 'same', (2,2), 2, negative_slope)
        self.convcell5 = ConvCell(60, 80, (3,3), 'same', (2,2), 2, negative_slope)

    def forward(self, x):
        x = self.convcell1(x)
        x = self.convcell2(x)
        x = self.convcell3(x)
        x = self.convcell4(x)
        x = self.convcell5(x)
        return x


class FCN(nn.Module):

    def __init__(self, negative_slope):
        super().__init__()
        self.negative_slope = negative_slope
        self.encoder = Encoder(self.negative_slope)
        self.deconvcell1 = TransposeConvCell(80, 100, kernel_size=(3, 3), dilation=4, negative_slope=0.1)
        self.deconvcell2 = TransposeConvCell(100, 120, kernel_size=(3, 3), dilation=8, negative_slope=0.1)
        self.deconvcell3 = TransposeConvCell(120, 140, kernel_size=(3, 3), dilation=16, negative_slope=0.1)
        self.deconvcell4 = TransposeConvCell(140, 160, kernel_size=(3, 3), dilation=32, negative_slope=0.1)
        self.deconvcell5 = TransposeConvCell(160, 19, kernel_size=(3, 3), dilation=64, negative_slope=0, is_linear=True)

    def forward(self, x):
        x = self.encoder(x)
        x = self.deconvcell1(x)
        x = self.deconvcell2(x)
        x = self.deconvcell3(x)
        x = self.deconvcell4(x)
        x = self.deconvcell5(x)
        return x


class Unet(nn.Module):

    def __init__(self, negative_slope):
        super().__init__()
        self.convcell1 = ConvCell(3, 10, (3,3), 'same', (2,2), 2, negative_slope)
        self.convcell2 = ConvCell(10, 20, (3,3), 'same', (2,2), 2, negative_slope)
        self.convcell3 = ConvCell(20, 40, (3,3), 'same', (2,2), 2, negative_slope)
        self.convcell4 = ConvCell(40, 60, (3,3), 'same', (2,2), 2, negative_slope)
        self.convcell5 = ConvCell(60, 80, (3,3), 'same', (2,2), 2, negative_slope)
        self.deconvcell1 = TransposeConvCell(80, 100, kernel_size=(3, 3), dilation=4, negative_slope=negative_slope)
        self.deconvcell2 = TransposeConvCell(100 + 60, 120, kernel_size=(3, 3), dilation=8, negative_slope=negative_slope)
        self.deconvcell3 = TransposeConvCell(120 + 40, 140, kernel_size=(3, 3), dilation=16, negative_slope=negative_slope)
        self.deconvcell4 = TransposeConvCell(140 + 20, 160, kernel_size=(3, 3), dilation=32, negative_slope=negative_slope)
        self.deconvcell5 = TransposeConvCell(160 + 10, 19, kernel_size=(3, 3), dilation=64, negative_slope=0, is_linear=True)

    def forward(self, x):
        x1 = self.convcell1(x)
        x2 = self.convcell2(x1)
        x3 = self.convcell3(x2)
        x4 = self.convcell4(x3)
        x5 = self.convcell5(x4)

        y4 = self.deconvcell1(x5)
        out4 = torch.concat((y4, x4), dim=1) # Skip connection
        y3 = self.deconvcell2(out4)
        out3 = torch.concat((y3, x3), dim=1)
        y2 = self.deconvcell3(out3)
        out2 = torch.concat((y2, x2), dim=1)
        y1 = self.deconvcell4(out2)
        out1 = torch.concat((y1, x1), dim=1)
        y0 = self.deconvcell5(out1)
        return y0
