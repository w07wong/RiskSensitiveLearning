import torch
import torch.nn as nn

class EntropicRisk(nn.Module):
    def __init__(self, t=10, criterion=nn.CrossEntropyLoss(reduction='none'), reduction='mean'):
        super().__init__()
        self.t = t
        self.criterion = criterion
        self.reduction = reduction
    
    def forward(self, output, labels):
        loss = self.criterion(output, labels)
        # t = 0 should return ERM
        if self.t == 0:
            return torch.mean(loss)
        
        if self.reduction == 'mean':
            return (1 / self.t) * torch.log(torch.mean(torch.exp(self.t * loss)))
        else:
            raise Exception('Only mean reduction type supported.')