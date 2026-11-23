import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftClusterAssignment(nn.Module):
    def __init__(
            self,
            num_cluster: int,
            hidden_dim: int,
            alpha: float = 1.0,
            temperature: float = 0.5,
    ):
        super(SoftClusterAssignment, self).__init__()
        self.num_cluster = num_cluster
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.temperature = temperature

        self.centroid = nn.Parameter(torch.randn(num_cluster, hidden_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [N, hidden_dim]
        diff = torch.sum((z.unsqueeze(1) - self.centroid) ** 2, dim=2)  # [N, num_cluster]
        numerator = (1.0 + (diff / self.alpha)) ** (-(self.alpha + 1.0) / 2)
        denominator = numerator.sum(dim=1, keepdim=True) + 1e-8
        q = numerator / denominator
        return q

    def _get_target_distribution(self, q: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q = torch.clamp(q, min=1e-8)
            weight = q ** (1.0 / self.temperature)
            freq = weight.sum(dim=0, keepdim=True)
            p = weight / (freq + 1e-8)
            p = p / (p.sum(dim=1, keepdim=True) + 1e-8)
        return p.detach()


    def kl_divergence_loss(self, q: torch.Tensor) -> torch.Tensor:
        p = self._get_target_distribution(q)
        kl = torch.sum(p * torch.log((p + 1e-8) / (q + 1e-8)), dim=1).mean()
        return kl


