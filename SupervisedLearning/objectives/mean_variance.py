import torch
import torch.nn as nn

class MeanVariance(nn.Module):
    def __init__(self, c=0.1, criterion=nn.CrossEntropyLoss(reduction='none'), reduction='mean'):
        super().__init__()
        self.c = c
        self.criterion = criterion
        self.reduction = reduction
    
    def forward(self, output, labels):
        loss = self.criterion(output, labels)
        var, mean = torch.var_mean(loss, unbiased=False)
        
        if self.reduction == 'mean':
            return mean + torch.mul(var, self.c)
        elif self.reduction == 'sum':
            return torch.sum(loss + torch.mul(var, self.c))
        elif self.reduction == 'none':
            return loss + torch.mul(var, self.c)
        else:
            raise Exception('Only mean, sum, none reduction types supported.')