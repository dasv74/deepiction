import torch
from torch import nn 
import torch.nn.functional as F


class Unet_N2N(nn.Module):
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
    def __init__(self) -> None:
        super().__init__() 

        self.conv_pad = 'same'
        self.conv_stride = 1
        self.alpha = 0.1
        self.n = 1                           # number of input channels
        self.m = 1                           # number of output channels
        
        # encoder convolutional layer 
        self.enc_conv0 = nn.Conv2d(self.n, 48, kernel_size=3, padding=self.conv_pad, stride=self.conv_stride)
        self.enc_conv1 = nn.Conv2d(48, 48, kernel_size=3, padding=self.conv_pad, stride=self.conv_stride)
        self.enc_conv2 = nn.Conv2d(48, 48, kernel_size=3, padding=self.conv_pad, stride=self.conv_stride)
        self.enc_conv3 = nn.Conv2d(48, 48, kernel_size=3, padding=self.conv_pad, stride=self.conv_stride)
        self.enc_conv4 = nn.Conv2d(48, 48, kernel_size=3, padding=self.conv_pad, stride=self.conv_stride)
        self.enc_conv5 = nn.Conv2d(48, 48, kernel_size=3, padding=self.conv_pad, stride=self.conv_stride)
        self.enc_conv6 = nn.Conv2d(48, 48, kernel_size=3, padding=self.conv_pad, stride=self.conv_stride)
        # decoder convolutional layer 
        self.dec_conv5A = nn.Conv2d(96, 96, kernel_size=3, padding=self.conv_pad, stride=self.conv_stride)
        self.dec_conv5B = nn.Conv2d(96, 96, kernel_size=3, padding=self.conv_pad, stride=self.conv_stride)
        self.dec_conv4A = nn.Conv2d(144, 96, kernel_size=3, padding=self.conv_pad, stride=self.conv_stride)
        self.dec_conv4B = nn.Conv2d(96, 96, kernel_size=3, padding=self.conv_pad, stride=self.conv_stride)
        self.dec_conv3A = nn.Conv2d(144, 96, kernel_size=3, padding=self.conv_pad, stride=self.conv_stride)
        self.dec_conv3B = nn.Conv2d(96, 96, kernel_size=3, padding=self.conv_pad, stride=self.conv_stride)
        self.dec_conv2A = nn.Conv2d(144, 96, kernel_size=3, padding=self.conv_pad, stride=self.conv_stride)
        self.dec_conv2B = nn.Conv2d(96, 96, kernel_size=3, padding=self.conv_pad, stride=self.conv_stride)
        self.dec_conv1A = nn.Conv2d(96+self.n, 64, kernel_size=3, padding=self.conv_pad, stride=self.conv_stride)
        self.dec_conv1B = nn.Conv2d(64, 32, kernel_size=3, padding=self.conv_pad, stride=self.conv_stride)
        self.dec_conv1C = nn.Conv2d(32, self.m, kernel_size=3, padding=self.conv_pad, stride=self.conv_stride)
        # downsampling layers (Max pool)
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)
        self.pool5 = nn.MaxPool2d(2)
        # upsampling layers (Nearest neighbor)
        self.upsample5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        
        # x = F.leaky_relu(self.enc_conv0(x), self.alpha)
        # encoder
        skips = [x]                 # skip connection container
        
        x = F.leaky_relu(self.enc_conv0(x), self.alpha)
        x = F.leaky_relu(self.enc_conv1(x), self.alpha)
        x = self.pool1(x)
        skips.append(x)
        x = F.leaky_relu(self.enc_conv2(x), self.alpha)
        x = self.pool2(x)
        skips.append(x)
        x = F.leaky_relu(self.enc_conv3(x), self.alpha)
        x = self.pool3(x)
        skips.append(x)
        x = F.leaky_relu(self.enc_conv4(x), self.alpha)
        x = self.pool4(x)
        skips.append(x)
        x = F.leaky_relu(self.enc_conv5(x), self.alpha)
        x = self.pool5(x)
        x = F.leaky_relu(self.enc_conv6(x), self.alpha)

        # decoder
        x = self.upsample5(x)
        x = torch.cat((x, skips.pop()), axis=1)
        x = F.leaky_relu(self.dec_conv5A(x), self.alpha)
        x = F.leaky_relu(self.dec_conv5B(x), self.alpha)
        x = self.upsample4(x)
        x = torch.cat((x, skips.pop()), axis=1)
        x = F.leaky_relu(self.dec_conv4A(x), self.alpha)
        x = F.leaky_relu(self.dec_conv4B(x), self.alpha)
        x = self.upsample3(x)
        x = torch.cat((x, skips.pop()), axis=1)
        x = F.leaky_relu(self.dec_conv3A(x), self.alpha)
        x = F.leaky_relu(self.dec_conv3B(x), self.alpha)
        x = self.upsample2(x)
        x = torch.cat((x, skips.pop()), axis=1)
        x = F.leaky_relu(self.dec_conv2A(x), self.alpha)
        x = F.leaky_relu(self.dec_conv2B(x), self.alpha)
        x = self.upsample1(x)
        x = torch.cat((x, skips.pop()), axis=1)
        x = F.leaky_relu(self.dec_conv1A(x), self.alpha)
        x = F.leaky_relu(self.dec_conv1B(x), self.alpha)
        x = torch.sigmoid(self.dec_conv1C(x))

        return x
    

