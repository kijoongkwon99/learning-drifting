import torch
from torch import nn, Tensor
import torch.nn.functional as F

class Drifting(nn.Module):
    def __init__(self, temp: float =0.05, mask_self: bool = True, normalize: str = "xy", eps: float =1e-12):
        super().__init__()
        self.temp = temp
        self.mask_self = mask_self
        self.normalize = normalize
        self.eps = eps

    def sample_noise_like(self, pos: Tensor) -> Tensor:
        return torch.randn_like(pos)

    def compute_V(
        self,
        x: torch.Tensor,
        y_pos: torch.Tensor,
        y_neg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the drifting field V (Algorithm 2 from paper, Page 12).

        This is the EXACT implementation from the paper's pseudocode.

        Args:
            x: Generated samples in feature space, shape (N, D)
            y_pos: Positive (real data) samples, shape (N_pos, D)
            y_neg: Negative (generated) samples, shape (N_neg, D)
            temperature: Temperature for softmax (smaller = sharper)
            mask_self: Whether to mask self-distances (when y_neg == x)

        Returns:
            V: Drifting field, shape (N, D)
        """
        N = x.shape[0]          # B
        N_pos = y_pos.shape[0]  # B
        N_neg = y_neg.shape[0]  # B

        device = x.device

        # 1. Compute pairwise L2 distances
        dist_pos = torch.cdist(x, y_pos, p=2)  # (N, N_pos)
        dist_neg = torch.cdist(x, y_neg, p=2)  # (N, N_neg)

        # 2. Mask self-distances (when y_neg contains x)
        if self.mask_self and N == N_neg:
            mask = torch.eye(N, device=device) * 1e6
            dist_neg = dist_neg + mask

        # 3. Compute logits
        logit_pos = -dist_pos / self.temp  # (N, N_pos)
        logit_neg = -dist_neg / self.temp  # (N, N_neg)

        logit = torch.cat([logit_pos, logit_neg], dim=1)
        kernel = (logit).exp()

        # 5. Normalization
        if self.normalize == "none":
            A = kernel

        elif self.normalize == "y":
            # softmax over y only
            A = torch.softmax(logit, dim=1)

        elif self.normalize == "xy":
            # symmetric normalization over x and y
            # row_sum = kernel.sum(dim=1, keepdim=True)  # (N, 1)
            # col_sum = kernel.sum(dim=0, keepdim=True)  # (1, N_pos + N_neg)
            # A = kernel / (row_sum * col_sum).clamp(1e-12).sqrt()

            A_row = torch.softmax(logit, dim=1)
            A_col = torch.softmax(logit, dim=0)
            A = torch.sqrt(A_row * A_col)

        else:
            raise ValueError(f"Unknown normalize mode: {self.normalize}")

        # 6. Split back
        A_pos = A[:, :N_pos]   # (N, N_pos)
        A_neg = A[:, N_pos:]   # (N, N_neg)

        # 7. Cross-weighting
        W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)  # (N, N_pos)
        W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)  # (N, N_neg)

        # 8. Compute drift: i guess this is figuring out the CENTROID of each distribution
        drift_pos = torch.mm(W_pos, y_pos)  # (N, D)
        drift_neg = torch.mm(W_neg, y_neg)  # (N, D)
        V = drift_pos - drift_neg

        return V
        
    
    def forward(self, model:nn.Module, pos: Tensor, **model_extras):

        noise = self.sample_noise_like(pos)
        gen = model(noise, **model_extras)
        neg = gen # reuse gen as negative

        # with torch.no_grad():
        V = self.compute_V(gen, pos, neg)
        gen_drifted = (gen + V).detach()

        loss = F.mse_loss(gen, gen_drifted)

        return {
            "loss": loss,
            "V": V,
            "target": gen_drifted,
        }