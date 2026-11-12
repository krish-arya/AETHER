"""Conditional TimeGAN (Yoon, Jarrett & van der Schaar, NeurIPS 2019) in PyTorch.

Five recurrent networks:

    embedder       X -> H        real sequence to latent sequence
    recovery       H -> X~       latent sequence back to data space
    generator      Z, c -> E^    noise (and condition) to a latent sequence
    supervisor     H, c -> H'    predicts the next latent step (teaches temporal dynamics)
    discriminator  H, c -> y     real vs generated latent sequences

The condition ``c`` (one-hot market regime) is appended to every time step of the
generator, supervisor and discriminator inputs, which makes this a conditional GAN:
at generation time the regime is chosen, not inferred.

Differences from the reference implementation: inputs are z-scored rather than
min-max scaled (returns are heavy-tailed), so the recovery network has a linear
output; the latent space keeps the sigmoid of the original.
"""

import torch
from torch import nn


class _Sequence(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, layers: int, sigmoid: bool) -> None:
        super().__init__()
        self.rnn = nn.GRU(in_dim, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, out_dim)
        self.sigmoid = sigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        out = self.head(out)
        return torch.sigmoid(out) if self.sigmoid else out


def _with_condition(x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
    """Append the per-sequence condition vector to every time step."""
    return torch.cat([x, condition.unsqueeze(1).expand(-1, x.shape[1], -1)], dim=-1)


class TimeGAN(nn.Module):
    def __init__(
        self,
        data_dim: int,
        condition_dim: int,
        hidden_dim: int = 48,
        num_layers: int = 2,
        noise_dim: int = 32,
    ) -> None:
        super().__init__()
        self.data_dim, self.condition_dim = data_dim, condition_dim
        self.hidden_dim, self.noise_dim = hidden_dim, noise_dim
        self.embedder = _Sequence(data_dim, hidden_dim, hidden_dim, num_layers, sigmoid=True)
        self.recovery = _Sequence(hidden_dim, hidden_dim, data_dim, num_layers, sigmoid=False)
        self.generator = _Sequence(
            noise_dim + condition_dim, hidden_dim, hidden_dim, num_layers, sigmoid=True
        )
        self.supervisor = _Sequence(
            hidden_dim + condition_dim, hidden_dim, hidden_dim, max(num_layers - 1, 1), sigmoid=True
        )
        self.discriminator = _Sequence(
            hidden_dim + condition_dim, hidden_dim, 1, num_layers, sigmoid=False
        )

    # --------------------------------------------------------------- components

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedder(x)

    def recover(self, h: torch.Tensor) -> torch.Tensor:
        return self.recovery(h)

    def supervise(self, h: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.supervisor(_with_condition(h, condition))

    def generate_latent(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.generator(_with_condition(z, condition))

    def discriminate(self, h: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.discriminator(_with_condition(h, condition)).squeeze(-1)

    # --------------------------------------------------------------- generation

    @torch.no_grad()
    def sample(
        self, condition: torch.Tensor, length: int, generator: torch.Generator
    ) -> torch.Tensor:
        """Synthetic sequences [batch, length, data_dim] in standardised units."""
        self.eval()
        z = torch.rand(len(condition), length, self.noise_dim, generator=generator)
        latent = self.supervise(self.generate_latent(z, condition), condition)
        return self.recover(latent)
