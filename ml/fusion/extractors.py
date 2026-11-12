"""Feature-fusion architectures for the RL policy (PRD 10.2).

The unified observation is a flat vector made of named groups (``market``, ``regime``,
``news``, ``scenario``, ``portfolio`` -- see ``ml.rl.observation``). Three ways of
combining them:

    early       the policy network reads the flat vector directly (no extractor)
    late        each group has its own encoder; the embeddings are concatenated
    attention   each group has its own encoder; a learned, input-dependent softmax
                weights the embeddings before they are summed

``late`` and ``attention`` are Stable-Baselines3 feature extractors. The attention
weights of the last forward pass are kept in ``last_attention`` so the explainability
engine can report how much each information source contributed to a decision. Those
weights describe what the network attended to; they are not a causal explanation.

Ablation-based fusion (PRD 10.2 item 4) is not an architecture: it is done by leaving a
group out of ``RLConfig.groups`` and comparing results (``ml.fusion.experiments``).
"""

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class GroupFusionExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        spec: dict[str, tuple[int, int]],
        mode: str = "late",
        embed_dim: int = 64,
    ) -> None:
        if mode not in ("late", "attention"):
            raise ValueError("mode must be 'late' or 'attention'")
        groups = [g for g, (lo, hi) in spec.items() if hi > lo]
        features_dim = embed_dim * len(groups) if mode == "late" else embed_dim
        super().__init__(observation_space, features_dim)
        self.mode = mode
        self.groups = groups
        self.slices = {g: (int(spec[g][0]), int(spec[g][1])) for g in groups}
        self.encoders = nn.ModuleDict(
            {
                g: nn.Sequential(
                    nn.Linear(hi - lo, embed_dim),
                    nn.ReLU(),
                    nn.Linear(embed_dim, embed_dim),
                    nn.ReLU(),
                )
                for g, (lo, hi) in self.slices.items()
            }
        )
        self.scorer = (
            nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Tanh(), nn.Linear(embed_dim, 1))
            if mode == "attention"
            else None
        )
        self.last_attention: torch.Tensor | None = None

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        embeddings = [
            self.encoders[g](observations[:, lo:hi]) for g, (lo, hi) in self.slices.items()
        ]
        if self.mode == "late":
            return torch.cat(embeddings, dim=1)
        stacked = torch.stack(embeddings, dim=1)  # [batch, groups, embed]
        weights = torch.softmax(self.scorer(stacked).squeeze(-1), dim=1)
        self.last_attention = weights.detach()
        return (stacked * weights.unsqueeze(-1)).sum(dim=1)

    def attention_by_group(self) -> dict[str, float] | None:
        """Attention weights of the most recent single-observation forward pass."""
        if self.last_attention is None:
            return None
        return dict(zip(self.groups, self.last_attention[0].tolist(), strict=True))


def fusion_policy_kwargs(mode: str, spec: dict, n_assets: int, embed_dim: int = 64) -> dict:
    """``policy_kwargs`` that plug the fusion extractor into an SB3 policy."""
    return {
        "features_extractor_class": GroupFusionExtractor,
        "features_extractor_kwargs": {
            "spec": {g: tuple(bounds) for g, bounds in spec.items()},
            "mode": mode,
            "embed_dim": embed_dim,
        },
    }
