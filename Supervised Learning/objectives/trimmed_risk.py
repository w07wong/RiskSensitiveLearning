import torch
import torch.nn as nn

class TrimmedRisk(nn.Module):
    def __init__(self, a=0.05, criterion=nn.CrossEntropyLoss(reduction='none'), reduction='mean'):
        super().__init__()
        assert a >= 0 and a <= 0.5 'a must be in [0, 0.5]'
        self.a = a
        self.criterion = criterion
        self.reduction = reduction
    
    def _get_untrimmed_losses(self, loss):
        sorted_indices = torch.argsort(loss, dim=0, descending=False)
        empirical_cdf = torch.argsort(sorted_indices) / len(loss)
        return ((empirical_cdf >= self.a) & (empirical_cdf <= 1 - self.a)).nonzero().squeeze()
    
    def forward(self, output, labels):
        loss = self.criterion(output, labels)
        untrimmed_losses = self._get_untrimmed_losses(loss)
        
        if self.reduction == 'mean':
            return torch.mean(torch.index_select(loss, 0, untrimmed_losses))
        elif self.reduction == 'sum':
            return torch.sum(torch.index_select(loss, 0, untrimmed_losses))
        return torch.index_select(loss, 0, untrimmed_losses)