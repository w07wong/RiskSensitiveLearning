import torch
import torch.nn as nn

class EntropicRisk(nn.Module):
    def __init__(self, t=10, criterion=nn.CrossEntropyLoss(reduction='none')):
        super().__init__()
        self.t = t
        self.criterion = criterion
    
    def forward(self, output, labels):
        loss = self.criterion(output, labels)
        # t = 0 should return ERM
        if self.t == 0:
            return torch.mean(loss)
        
        return (1 / self.t) * torch.log(torch.mean(torch.exp(self.t * loss)))