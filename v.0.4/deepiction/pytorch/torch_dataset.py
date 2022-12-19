import torch

class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, device, transform=None):
        self.data = torch.from_numpy(data).float()
        self.targets = torch.from_numpy(targets).float()
        self.device = device
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        x = x.to(self.device)
        y = y.to(self.device)
        return x, y
    
    def __len__(self):
        return len(self.data)
