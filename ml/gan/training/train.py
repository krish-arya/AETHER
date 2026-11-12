"""TimeGAN training pipeline (independent of every other model, PRD 3.2).

Three phases, as in the paper:

    1. embedding    autoencoder: reconstruct real windows through the latent space
    2. supervised   the supervisor learns one-step-ahead latent dynamics on real data
    3. joint        generator + supervisor vs discriminator, while the embedder keeps
                    the latent space useful; the generator also matches the first two
                    moments of the data

Deterministic for a given seed and config (CPU). The loss history is returned and
saved so a run can be inspected; nothing about training is reported as "good" -- that
is the job of ``ml.gan.evaluation``.

Artifacts (``save_gan``): ``model.pt`` (weights), ``scaler.npz`` (training mean/std and
asset anchors), ``meta.json`` (symbols, config, seed, training period, loss history).
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from ml.backtesting.data import MarketData
from ml.gan.calibration import Calibration, fit_calibration
from ml.gan.config import GANConfig
from ml.gan.data import CONDITIONS, WindowSet, build_windows
from ml.gan.models.timegan import TimeGAN


@dataclass
class TrainedGAN:
    model: TimeGAN
    windows: WindowSet
    cfg: GANConfig
    seed: int
    train_start: str
    train_end: str
    losses: dict[str, list[float]] = field(default_factory=dict)
    train_seconds: float = 0.0
    path: Path | None = None
    calibration: Calibration | None = None


def _one_hot(codes: np.ndarray) -> torch.Tensor:
    return F.one_hot(torch.as_tensor(codes), num_classes=len(CONDITIONS)).float()


def train_timegan(
    market: MarketData, cfg: GANConfig, *, train_start, train_end, seed: int
) -> TrainedGAN:
    torch.set_num_threads(cfg.torch_threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    windows = build_windows(
        market, train_start, train_end,
        sequence_length=cfg.sequence_length, volume_window=cfg.volume_window,
        clip_sigma=cfg.clip_sigma,
    )  # fmt: skip
    data = torch.as_tensor(windows.windows)
    conditions = _one_hot(windows.conditions)
    model = TimeGAN(windows.dim, len(CONDITIONS), cfg.hidden_dim, cfg.num_layers, cfg.noise_dim)

    def adam(*modules):
        parameters = [p for module in modules for p in module.parameters()]
        return torch.optim.Adam(parameters, lr=cfg.learning_rate)

    opt_autoencoder = adam(model.embedder, model.recovery)
    opt_supervisor = adam(model.supervisor)
    opt_generator = adam(model.generator, model.supervisor)
    opt_embedder = adam(model.embedder, model.recovery)
    opt_discriminator = adam(model.discriminator)
    losses: dict[str, list[float]] = {
        "reconstruction": [], "supervised": [], "generator": [], "discriminator": [],
    }  # fmt: skip

    def batch():
        index = rng.integers(0, len(data), cfg.batch_size)
        return data[index], conditions[index]

    def noise(n: int) -> torch.Tensor:
        return torch.as_tensor(
            rng.random((n, cfg.sequence_length, cfg.noise_dim), dtype=np.float32)
        )

    def supervised_loss(h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(model.supervise(h, c)[:, :-1], h[:, 1:])

    started = time.perf_counter()
    model.train()

    for _ in range(cfg.embedding_iterations):
        x, _ = batch()
        loss = F.mse_loss(model.recover(model.embed(x)), x)
        opt_autoencoder.zero_grad()
        loss.backward()
        opt_autoencoder.step()
        losses["reconstruction"].append(loss.item())

    for _ in range(cfg.supervised_iterations):
        x, c = batch()
        loss = supervised_loss(model.embed(x).detach(), c)
        opt_supervisor.zero_grad()
        loss.backward()
        opt_supervisor.step()
        losses["supervised"].append(loss.item())

    ones = torch.ones(cfg.batch_size, cfg.sequence_length)
    zeros = torch.zeros(cfg.batch_size, cfg.sequence_length)
    for _ in range(cfg.joint_iterations):
        for _ in range(2):  # two generator updates per discriminator update
            x, c = batch()
            h = model.embed(x)
            e_hat = model.generate_latent(noise(cfg.batch_size), c)
            h_hat = model.supervise(e_hat, c)
            x_hat = model.recover(h_hat)
            adversarial = F.binary_cross_entropy_with_logits(model.discriminate(h_hat, c), ones)
            adversarial_e = F.binary_cross_entropy_with_logits(model.discriminate(e_hat, c), ones)
            moments = (x_hat.std(dim=0) - x.std(dim=0)).abs().mean() + (
                x_hat.mean(dim=0) - x.mean(dim=0)
            ).abs().mean()
            supervised = supervised_loss(h.detach(), c)
            generator_loss = (
                adversarial
                + cfg.gamma * adversarial_e
                + 100 * torch.sqrt(supervised)
                + 100 * moments
            )
            opt_generator.zero_grad()
            generator_loss.backward()
            opt_generator.step()

            h = model.embed(x)
            reconstruction = F.mse_loss(model.recover(h), x)
            embedder_loss = 10 * torch.sqrt(reconstruction) + 0.1 * supervised_loss(h, c)
            opt_embedder.zero_grad()
            embedder_loss.backward()
            opt_embedder.step()

        x, c = batch()
        with torch.no_grad():
            h = model.embed(x)
            e_hat = model.generate_latent(noise(cfg.batch_size), c)
            h_hat = model.supervise(e_hat, c)
        discriminator_loss = (
            F.binary_cross_entropy_with_logits(model.discriminate(h, c), ones)
            + F.binary_cross_entropy_with_logits(model.discriminate(h_hat, c), zeros)
            + cfg.gamma * F.binary_cross_entropy_with_logits(model.discriminate(e_hat, c), zeros)
        )
        if discriminator_loss.item() > 0.15:  # do not let the discriminator run away
            opt_discriminator.zero_grad()
            discriminator_loss.backward()
            opt_discriminator.step()
        losses["generator"].append(generator_loss.item())
        losses["discriminator"].append(discriminator_loss.item())

    model.eval()
    gan = TrainedGAN(
        model=model,
        windows=windows,
        cfg=cfg,
        seed=seed,
        train_start=str(train_start)[:10],
        train_end=str(train_end)[:10],
        losses=losses,
        train_seconds=time.perf_counter() - started,
    )
    if cfg.calibrate_moments:
        gan.calibration = fit_calibration(model, windows, cfg, seed)
    return gan


def save_gan(gan: TrainedGAN, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(gan.model.state_dict(), out_dir / "model.pt")
    w = gan.windows
    arrays = {"mean": w.mean, "std": w.std, "base_value": w.base_value, "last_close": w.last_close}
    if gan.calibration is not None:
        arrays["calibration_shift"] = gan.calibration.shift
        arrays["calibration_scale"] = gan.calibration.scale
    np.savez(out_dir / "scaler.npz", **arrays)
    meta = {
        "symbols": w.symbols,
        "excluded": w.excluded,
        "data_dim": w.dim,
        "config": gan.cfg.model_dump(mode="json"),
        "seed": gan.seed,
        "train_start": gan.train_start,
        "train_end": gan.train_end,
        "training_windows": int(len(w.windows)),
        "clipped_fraction": w.clipped_fraction,
        "train_seconds": round(gan.train_seconds, 1),
        "losses": {name: [round(v, 6) for v in values[:: max(len(values) // 200, 1)]]
                   for name, values in gan.losses.items()},
    }  # fmt: skip
    with open(out_dir / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    gan.path = out_dir
    return out_dir


def load_gan(path: Path, market: MarketData) -> TrainedGAN:
    """Reload a saved generator. The training windows are rebuilt from ``market``
    (the same dataset) so evaluation and memorisation checks remain possible."""
    path = Path(path)
    with open(path / "meta.json", encoding="utf-8") as handle:
        meta = json.load(handle)
    cfg = GANConfig.model_validate(meta["config"])
    windows = build_windows(
        market, meta["train_start"], meta["train_end"],
        sequence_length=cfg.sequence_length, volume_window=cfg.volume_window,
        clip_sigma=cfg.clip_sigma,
    )  # fmt: skip
    if windows.symbols != meta["symbols"]:
        raise ValueError("the dataset does not match the one this generator was trained on")
    model = TimeGAN(
        meta["data_dim"], len(CONDITIONS), cfg.hidden_dim, cfg.num_layers, cfg.noise_dim
    )
    model.load_state_dict(torch.load(path / "model.pt", weights_only=True))
    model.eval()
    saved = np.load(path / "scaler.npz")
    calibration = None
    if "calibration_scale" in saved:
        calibration = Calibration(saved["calibration_shift"], saved["calibration_scale"])
    return TrainedGAN(
        model, windows, cfg, meta["seed"], meta["train_start"], meta["train_end"], path=path,
        calibration=calibration,
    )  # fmt: skip
