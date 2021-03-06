import torch
import torch.nn as nn
import numpy as np

class CVaR(nn.Module):
    def __init__(self, a=0.05, inverted=False, reduction='mean'):
        super().__init__()
        self.a = a
        self.inverted = inverted
        self.reduction = reduction
    
    def _value_at_risk(self, loss):
        sorted_loss, sorted_indices = torch.sort(loss, dim=0, descending=False, stable=True)
        empirical_cdf = torch.argsort(sorted_indices) / len(loss)
        sorted_cdf, _ = torch.sort(empirical_cdf, dim=0, descending=False, stable=True)
        value_at_risk_idx = np.searchsorted(sorted_cdf, 1 - self.a, side='left')
        return sorted_loss[value_at_risk_idx]
    
    def forward(self, loss):
        multiplier = 1
        if self.inverted:
            loss *= -1
            multiplier = -1
            
        values_at_risk = (loss >= self._value_at_risk(loss)).nonzero().squeeze()
        
        if self.reduction == 'mean':
            risk = multiplier * torch.mean(torch.index_select(loss, 0, values_at_risk))
            loss *= multiplier # Undo modifier to loss passed in.
            return risk
        elif self.reduction == 'sum':
            risk = multiplier * torch.sum(torch.index_select(loss, 0, values_at_risk))
            loss *= multiplier
            return risk
        elif self.reduction == 'none':
            risk = values_at_risk
            loss *= multiplier
            return risk
        else:
            raise Exception('Only mean, sum, none reduction types supported.')