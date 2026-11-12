"""Per-regime moment calibration of generated scenarios.

A TimeGAN trained on a few thousand windows tends to under-disperse: its samples are
smoother than the data. This step corrects the first two moments only. For each regime
``c`` and each variable ``j`` (in standardised units):

    x'  =  (x - m_gen[c, j]) / s_gen[c, j] * s_train[c, j] + m_train[c, j]

where ``m_gen, s_gen`` are measured on a large sample of generated windows for that
regime and ``m_train, s_train`` on the *training* windows of that regime. Regimes with
fewer than ``min_regime_windows`` training windows use the all-regime training moments.

What it does and does not do
    * It is a fixed affine map fitted on training data only -- no held-out data, no
      look-ahead.
    * It restores the level and scale of each variable per regime (e.g. stress windows
      are as volatile as stress windows were).
    * It does **not** add fat tails, volatility clustering or cross-asset dependence
      the generator failed to learn. Those are still measured -- and reported -- by
      ``ml.gan.evaluation``, on the calibrated output that is actually used.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from ml.gan.config import GANConfig
from ml.gan.data import CONDITIONS, WindowSet


@dataclass
class Calibration:
    shift: np.ndarray  # [conditions, dim]
    scale: np.ndarray  # [conditions, dim]

    def apply(self, standardised: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """Calibrate windows [n, L, dim] given each window's condition code."""
        return standardised * self.scale[codes][:, None, :] + self.shift[codes][:, None, :]


def fit_calibration(model, windows: WindowSet, cfg: GANConfig, seed: int) -> Calibration:
    k, dim = len(CONDITIONS), windows.dim
    overall = windows.windows.reshape(-1, dim)
    overall_mean, overall_std = overall.mean(axis=0), overall.std(axis=0)
    shift, scale = np.zeros((k, dim)), np.ones((k, dim))
    generator = torch.Generator().manual_seed(seed + 10_000)
    for code in range(k):
        members = windows.windows[windows.conditions == code]
        if len(members) >= cfg.min_regime_windows:
            flat = members.reshape(-1, dim)
            target_mean, target_std = flat.mean(axis=0), flat.std(axis=0)
        else:
            target_mean, target_std = overall_mean, overall_std
        condition = F.one_hot(torch.full((cfg.calibration_windows,), code), num_classes=k).float()
        sample = model.sample(condition, cfg.sequence_length, generator).numpy().reshape(-1, dim)
        generated_std = np.where(sample.std(axis=0) > 1e-6, sample.std(axis=0), 1.0)
        scale[code] = target_std / generated_std
        shift[code] = target_mean - sample.mean(axis=0) * scale[code]
    return Calibration(shift=shift, scale=scale)
