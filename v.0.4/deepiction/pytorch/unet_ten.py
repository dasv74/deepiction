import torch
from torch import nn 
import torch.nn.functional as F
from torch import cat 
import torchvision 
import numpy as np
class Unet_Ten(nn.Module):

    def __init__(self, input_shape, noutputs, nchannels, npools, batchnorm, dropout, activation) -> None:
        super().__init__() 
        self.n = input_shape[2]                           # number of input channels
        self.m = noutputs 
        self.npools = npools
        self.batchnorm = batchnorm
        self.dropout = dropout        
        self.nchannels = nchannels
        
        # convolutional layers for encoder
        self.enc_conv0 = nn.Conv2d(self.n, nchannels, padding="same", kernel_size=3, stride=1)
        self.enc_conv1 = nn.Conv2d(nchannels, nchannels*2, padding="same", kernel_size=3, stride=1)
        self.enc_conv2 = nn.Conv2d(nchannels*2, nchannels*4, padding="same", kernel_size=3, stride=1)
        self.enc_conv3 = nn.Conv2d(nchannels*4, nchannels*8, padding="same", kernel_size=3, stride=1)  
        self.enc_conv4 = nn.Conv2d(nchannels*8, nchannels*16, padding="same", kernel_size=3, stride=1)  

        # transpose convolutional layers for encoder
        self.dec_conv0 = nn.ConvTranspose2d(nchannels*16, nchannels*8, kernel_size=3, stride=1)
        self.dec_conv1 = nn.ConvTranspose2d(nchannels*16, nchannels*4, kernel_size=3, stride=1)
        self.dec_conv2 = nn.ConvTranspose2d(nchannels*8, nchannels*2, kernel_size=3, stride=1)
        self.dec_conv3 = nn.ConvTranspose2d(nchannels*4, nchannels, kernel_size=3, stride=1)
        self.dec_conv4 = nn.ConvTranspose2d(nchannels*2, self.m, kernel_size=3, stride=1)
    
        self.pool0 = nn.MaxPool2d(2)
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)
        
        self.upsample0 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample4 = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Batch normalization blocks 
        self.batch_norm0 = nn.BatchNorm2d(nchannels)
        self.batch_norm1 = nn.BatchNorm2d(nchannels*2)
        self.batch_norm2 = nn.BatchNorm2d(nchannels*4)
        self.batch_norm3 = nn.BatchNorm2d(nchannels*8)
        self.batch_norm4 = nn.BatchNorm2d(nchannels*16)
        self.batch_norm5 = nn.BatchNorm2d(nchannels*8)
        self.batch_norm6 = nn.BatchNorm2d(nchannels*4)
        self.batch_norm7 = nn.BatchNorm2d(nchannels*2)
        self.batch_norm8 = nn.BatchNorm2d(nchannels)

        self.drop_out0 = nn.Dropout2d(p=self.dropout)
        self.drop_out1 = nn.Dropout2d(p=self.dropout)
        self.drop_out2 = nn.Dropout2d(p=self.dropout)
        self.drop_out3 = nn.Dropout2d(p=self.dropout)
        self.drop_out4 = nn.Dropout2d(p=self.dropout)
        self.drop_out5 = nn.Dropout2d(p=self.dropout)
        self.drop_out6 = nn.Dropout2d(p=self.dropout)
        self.drop_out7 = nn.Dropout2d(p=self.dropout)
        self.drop_out8 = nn.Dropout2d(p=self.dropout)
    
    def forward(self, x):
        skips = [x]   # skip connection container
        self.npools = 3
        # encoder
        x = self.enc_conv0(x)
        x = F.relu(x)
        if self.batchnorm == True: x = self.batch_norm0(x)
        if self.dropout > 0: x = self.drop_out0(x)
        x = self.pool0(x)
        skips.append(x)
        
        x = self.enc_conv1(x)
        x = F.relu(x)
        if self.batchnorm == True: x = self.batch_norm1(x)
        if self.dropout > 0: x = self.drop_out1(x)
        x = self.pool1(x)
        skips.append(x)
        
        x = self.enc_conv2(x)
        x = F.relu(x)
        if self.batchnorm == True: x = self.batch_norm2(x)
        if self.dropout > 0: x = self.drop_out2(x)
        x = self.pool2(x)
        skips.append(x)
        
        x = self.enc_conv3(x)
        x = F.relu(x)
        if self.batchnorm == True: x = self.batch_norm3(x)
        if self.dropout > 0: x = self.drop_out3(x)
        x = self.pool3(x)
        
        skips.append(x) 
        x = self.enc_conv4(x)
        x = F.relu(x)
        if self.batchnorm == True: x = self.batch_norm4(x)
        if self.dropout > 0: x = self.drop_out4(x)

        b, c, w, h = x.shape

        # decoder

        x = self.dec_conv0(x)
        x = F.relu(x)
        if self.batchnorm == True: x = self.batch_norm5(x)
        if self.dropout > 0: x = self.drop_out5(x)
        y = skips.pop()
        x = self.crop(x, (x.shape[0],x.shape[1],w,h))
        y = self.crop(y, (y.shape[0],y.shape[1],w,h))
        x = cat((x, y), axis=1)
        x = self.upsample0(x) 
                 
        x = self.dec_conv1(x)
        x = F.relu(x)
        if self.batchnorm == True: x = self.batch_norm6(x)
        if self.dropout > 0: x = self.drop_out6(x)
        y = skips.pop()
        x = self.crop(x, (x.shape[0],x.shape[1],w*2,h*2))
        y = self.crop(y, (y.shape[0],y.shape[1],w*2,h*2))
        x = cat((x, y), axis=1)
        x = self.upsample1(x)   
         
        x = self.dec_conv2(x)
        x = F.relu(x)
        if self.batchnorm == True: x = self.batch_norm7(x)
        if self.dropout > 0: x = self.drop_out7(x)
        y = skips.pop()
        x = self.crop(x, (x.shape[0],x.shape[1],w*4,h*4))
        y = self.crop(y, (y.shape[0],y.shape[1],w*4,h*4))    
        x = cat((x, y), axis=1)
        x = self.upsample2(x)
        
        x = self.dec_conv3(x)
        x = F.relu(x)
        if self.batchnorm == True: x = self.batch_norm8(x)
        if self.dropout > 0: x = self.drop_out8(x)
        y = skips.pop()
        x = self.crop(x, (x.shape[0],x.shape[1],w*8,h*8))
        y = self.crop(y, (y.shape[0],y.shape[1],w*8,h*8))
        x = cat((x, y), axis=1)
        x = self.upsample3(x)
        x = self.dec_conv4(x)
        x = self.crop(x, (x.shape[0],x.shape[1],w*16,h*16))   
        x = torch.sigmoid(x)

        return x
    
    def crop(self, y, shape):
        y   = torchvision.transforms.CenterCrop([shape[2], shape[3]])(y)
        return y