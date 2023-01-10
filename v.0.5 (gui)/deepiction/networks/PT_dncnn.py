from torch import nn 
import torch
from deepiction.imagedataset import TorchDataset1
from deepiction.tools import Tools
from  torch.utils.data import DataLoader
import numpy as np

class PT_DnCNN(nn.Module):

    def __init__(self, ninputs, noutputs, nchannels, npools, batchnorm, dropout, activation) -> None:
        super(PT_DnCNN, self).__init__()
        layers = [nn.Sequential(
            nn.Conv2d(ninputs, nchannels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True))]
        for i in range(npools - 2):
            layers.append(nn.Sequential(
                nn.Conv2d(nchannels, nchannels, kernel_size=3, padding=1),
                nn.BatchNorm2d(nchannels),
                nn.ReLU(inplace=True)))
        layers.append(nn.Conv2d(nchannels, noutputs, kernel_size=3, padding=1))
        self.layers = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, input):
        y = input
        residual = self.layers(y)
        return y - residual
    
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

