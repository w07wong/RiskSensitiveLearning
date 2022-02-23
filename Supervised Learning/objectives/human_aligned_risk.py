import torch
import torch.nn as nn

class HumanAlignedRisk(nn.Module):
    def __init__(self, a=0.4, b=0.3, criterion=nn.CrossEntropyLoss(reduction='none'), reduction='mean'):
        assert(reduction == 'none' or reduction == 'mean' or reduction == 'sum')
        super().__init__()
        self.a = a
        self.b = b
        self.reduction = reduction
        self.criterion = criterion
      
    def _cpt_poly_derivative(self, a, b, Fx):
        return (3 - 3 * b) / (a**2 - a + 1) * (3 * Fx**2 - 2 * (a + 1) * Fx + a) + 1

    def forward(self, output, labels):
        loss = self.criterion(output, labels)

        empirical_cdf = torch.argsort(torch.argsort(loss, dim=0, descending=False)) / len(loss)
        weighted_cdf = torch.Tensor(self._cpt_poly_derivative(self.a, self.b, empirical_cdf))
        loss *= weighted_cdf

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        return loss