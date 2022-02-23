import torch
import torch.nn as nn

class CVaRLoss(nn.Module):
    def __init__(self, a=0.05, criterion=nn.CrossEntropyLoss(reduction='none'), reduction='mean'):
        super().__init__()
        self.a = a
        self.criterion = criterion
        self.reduction = reduction
    
    def _value_at_risk(self, loss):
        sorted_loss, sorted_indices = torch.sort(loss, dim=0, descending=False, stable=True)
        empirical_cdf = torch.argsort(sorted_indices) / len(loss)
        sorted_cdf, _ = torch.sort(empirical_cdf, dim=0, descending=False, stable=True)
        value_at_risk_idx = np.searchsorted(sorted_cdf, 1 - self.a, side='left')
        return sorted_loss[value_at_risk_idx]
    
    def forward(self, output, labels):
        loss = self.criterion(output, labels)
        values_at_risk = (loss >= self._value_at_risk(loss)).nonzero().squeeze()
        
        if self.reduction == 'mean':
            return torch.mean(torch.index_select(loss, 0, values_at_risk))
        elif self.reduction == 'sum':
            return torch.sum(torch.index_select(loss, 0, values_at_risk))
        return torch.index_select(loss, 0, values_at_risk)