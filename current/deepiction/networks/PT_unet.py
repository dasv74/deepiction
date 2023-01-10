import torch
from torch import nn 
import torch.nn.functional as F
from  torchvision.transforms.functional import resize
from torch import cat 
from deepiction.imagedataset import TorchDataset1
from deepiction.tools import Tools
from  torch.utils.data import DataLoader
import numpy as np

class PT_Unet(nn.Module):

    def double_conv(self, nin, nout):
        if self.batchnorm == False:
            if self.dropout <= 0:
                block = nn.Sequential(
                    nn.Conv2d(nin, nout, kernel_size=3, stride=1, padding='same'),
                    nn.ReLU(),
                    nn.Conv2d(nout, nout, kernel_size=3, stride=1, padding='same'),
                    nn.ReLU(),
                )
            else:
                block = nn.Sequential(
                    nn.Conv2d(nin, nout, kernel_size=3, stride=1, padding='same'),
                    nn.Dropout2d(p=self.dropout),
                    nn.ReLU(),
                    nn.Conv2d(nout, nout, kernel_size=3, stride=1, padding='same'),
                    nn.Dropout2d(p=self.dropout),
                    nn.ReLU(),
                )
        else:
            if self.dropout <= 0:
                block = nn.Sequential(
                    nn.Conv2d(nin, nout, kernel_size=3, stride=1, padding='same'),
                    nn.BatchNorm2d(nout),
                    nn.ReLU(),
                    nn.Conv2d(nout, nout, kernel_size=3, stride=1, padding='same'),
                    nn.BatchNorm2d(nout),
                    nn.ReLU(),
                )
            else:
                block = nn.Sequential(
                    nn.Conv2d(nin, nout, kernel_size=3, stride=1, padding='same'),
                    nn.BatchNorm2d(nout),
                    nn.Dropout2d(p=self.dropout),
                    nn.ReLU(),
                    nn.Conv2d(nout, nout, kernel_size=3, stride=1, padding='same'),
                    nn.BatchNorm2d(nout),
                    nn.Dropout2d(p=self.dropout),
                    nn.ReLU(),
                )
        return block
    
    def __init__(self, ninputs, noutputs, nchannels, npools, batchnorm, dropout, activation) -> None:
        super().__init__() 
        self.n = ninputs # number of input channels
        self.m = noutputs 
        self.npools = npools
        self.nchannels = nchannels
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.activation = activation
        nc = nchannels
        mode = 'nearest'
        self.enc_initial = self.double_conv(self.n, nc)
        
        if npools >= 1: self.pool1 = nn.MaxPool2d(2); self.contract1 = self.double_conv(nc, nc*2); nc = nc*2
        if npools >= 2: self.pool2 = nn.MaxPool2d(2); self.contract2 = self.double_conv(nc, nc*2); nc = nc*2
        if npools >= 3: self.pool3 = nn.MaxPool2d(2); self.contract3 = self.double_conv(nc, nc*2); nc = nc*2
        if npools >= 4: self.pool4 = nn.MaxPool2d(2); self.contract4 = self.double_conv(nc, nc*2); nc = nc*2
        if npools >= 5: self.pool5 = nn.MaxPool2d(2); self.contract5 = self.double_conv(nc, nc*2); nc = nc*2            
        self.bottleneck = self.double_conv(nc, nc//2) 
        nc = nchannels
        self.final_layer = self.double_conv(nc//2, self.m) 
        if npools >= 1: self.exp1 = self.double_conv(nc*2, nc//2); self.up1 = nn.Upsample(scale_factor=2, mode=mode); nc = nc * 2  
        if npools >= 2: self.exp2 = self.double_conv(nc*2, nc//2); self.up2 = nn.Upsample(scale_factor=2, mode=mode); nc = nc * 2
        if npools >= 3: self.exp3 = self.double_conv(nc*2, nc//2); self.up3 = nn.Upsample(scale_factor=2, mode=mode); nc = nc * 2
        if npools >= 4: self.exp4 = self.double_conv(nc*2, nc//2); self.up4 = nn.Upsample(scale_factor=2, mode=mode); nc = nc * 2
        if npools >= 5: self.exp5 = self.double_conv(nc*2, nc//2); self.up5 = nn.Upsample(scale_factor=2, mode=mode); nc = nc * 2
        self.sigmoid = nn.Sigmoid()
        
    def concat(self, x, y):
        x = resize(x, y.shape[2:])
        x = cat((x, y), dim=1)
        return x
    
    def forward(self, x):
        skips = []   # skip connection container
        input = x
        
        x = self.enc_initial(x); x = F.relu(x); skips.append(x)
                
        if self.npools >= 1: x = self.pool1(x); x = self.contract1(x); skips.append(x)
        if self.npools >= 2: x = self.pool2(x); x = self.contract2(x); skips.append(x)
        if self.npools >= 3: x = self.pool3(x); x = self.contract3(x); skips.append(x)
        if self.npools >= 4: x = self.pool4(x); x = self.contract4(x); skips.append(x)
        if self.npools >= 5: x = self.pool5(x); x = self.contract5(x); skips.append(x)  
                  
        x = self.bottleneck(x); y = skips.pop()

        if self.npools >= 5: x = self.up5(x); x = self.concat(x, skips.pop()); x = self.exp5(x) 
        if self.npools >= 4: x = self.up4(x); x = self.concat(x, skips.pop()); x = self.exp4(x) 
        if self.npools >= 3: x = self.up3(x); x = self.concat(x, skips.pop()); x = self.exp3(x)
        if self.npools >= 2: x = self.up2(x); x = self.concat(x, skips.pop()); x = self.exp2(x)
        if self.npools >= 1: x = self.up1(x); x = self.concat(x, skips.pop()); x = self.exp1(x)

        x = resize(x, input.shape[2:]); x = self.final_layer(x)
        '''
        if self.activation == 'relu': x = F.relu(x)
        elif self.activation == 'softmax': x = F.softmax(x)
        elif self.activation == 'sigmoid': x = self.sigmoid(x)
        else: print('\nError in the final activation\n')
        '''
        return x

    def predict(self, images):
        device = Tools.device_torch()
        sources = np.transpose(images, (0, 3, 1, 2))
        test_dataset = TorchDataset1(sources, device, transform=None)
        testDataLoader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
        preds = np.zeros(sources.shape)
        with torch.no_grad():
            for i, images in enumerate(testDataLoader, 0):
                outs = self(images) #bcwh  
                preds[i,:,:,:] = outs.cpu().numpy()
        predictions = np.transpose(preds, (0, 2, 3, 1)) 
        return predictions 
