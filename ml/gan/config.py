"""Typed configuration for the GAN scenario engine (loaded from configs/gan.yaml)."""

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QualityGates(_Strict):
    """Thresholds a trained generator must meet to be marked usable (PRD phase 4 exit
    criterion). They are judgement calls: tighten them as the generator improves."""

    max_return_ks: float = Field(0.10, gt=0, le=1)
    max_correlation_error: float = Field(0.15, gt=0)
    max_discriminative_score: float = Field(0.25, gt=0, le=0.5)
    # Synthetic windows must not sit much closer to the training data than unseen real
    # windows do (ratio of median nearest-neighbour distances).
    min_memorization_ratio: float = Field(0.6, gt=0)
    max_copied_fraction: float = Field(0.0, ge=0, le=1)


class GANConfig(_Strict):
    sequence_length: int = Field(24, ge=4)
    volume_window: int = Field(60, ge=5)  # sessions in the rolling volume baseline
    hidden_dim: int = Field(48, ge=4)
    num_layers: int = Field(2, ge=1)
    noise_dim: int = Field(32, ge=1)
    batch_size: int = Field(128, ge=8)
    learning_rate: float = Field(1e-3, gt=0)
    embedding_iterations: int = Field(600, ge=1)
    supervised_iterations: int = Field(600, ge=1)
    joint_iterations: int = Field(1200, ge=1)
    gamma: float = Field(1.0, ge=0)  # weight of the embedding-space adversarial loss
    clip_sigma: float = Field(6.0, gt=0)  # standardised inputs are clipped to +/- this
    # After training, rescale generated values per regime so their mean and standard
    # deviation match the training windows of that regime (see ml/gan/calibration.py).
    calibrate_moments: bool = True
    calibration_windows: int = Field(500, ge=50)
    min_regime_windows: int = Field(30, ge=2)  # fewer -> use all-regime moments
    evaluation_windows: int = Field(1000, ge=50)  # synthetic windows used for evaluation
    gates: QualityGates = QualityGates()
    torch_threads: int = Field(4, ge=1)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GANConfig":
        with open(path, encoding="utf-8") as handle:
            return cls.model_validate(yaml.safe_load(handle))

    def config_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"torch_threads", "gates"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
