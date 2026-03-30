import torch
from torch import nn, Tensor
import torch.nn.functional as F

class Drifting(nn.Module):
    def __init__(self, temp: float =0.05):
        super().__init__()
        self.temp = temp

    def sample_noise_like(self, pos: Tensor) -> Tensor:
        return torch.randn_like(pos)

    def compute_drift(self, gen: Tensor, pos: Tensor) -> Tensor:
        """
        Compute drift field V with attention-based kernel,
        """
        targets = torch.cat([gen, pos], dim=0)
        G = gen.shape[0]

        dist = torch.cdist(gen, targets)
        dist[:, :G].fill_diagonal_(1e6) # mask itself
        kernel = (-dist / self.temp).exp() # unnormal kernel

        normalizer = kernel.sum(dim=-1, keepdim=True) * kernel.sum(dim=-2, keepdim=True)
        normalizer = normalizer.clamp(1e-12).sqrt()
        normalized_kernel = kernel / normalizer

        pos_coeff = normalized_kernel[:, G:] * normalized_kernel[:, :G].sum(dim=-1, keepdim=True)
        pos_V = pos_coeff @ targets[G:]
        neg_coeff = normalized_kernel[:, :G] * normalized_kernel[:, G:].sum(dim=-1, keepdim=True)
        neg_V = neg_coeff @ targets[:G]

        return pos_V - neg_V
    
    def forward(self, model:nn.Module, pos: Tensor, **model_extras):

        noise = self.sample_noise_like(pos)
        gen = model(noise, **model_extras)

        with torch.no_grad():
            V = self.compute_drift(gen, pos)
            target = (gen + V).detach()

        loss = F.mse_loss(gen, target)

        return {
            "loss": loss,
            "V": V,
            "target": target,
        }