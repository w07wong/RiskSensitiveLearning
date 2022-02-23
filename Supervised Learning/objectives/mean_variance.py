import torch
import torch.nn as nn

class MeanVariance(nn.Module):
    def __init__(self, c=0.1, criterion=nn.CrossEntropyLoss(reduction='none')):
        super().__init__()
        self.c = c
        self.criterion = criterion
    
    def forward(self, output, labels):
        loss = self.criterion(output, labels)
        var, mean = torch.var_mean(loss, unbiased=False)
        
        return mean + torch.mul(var, self.c)