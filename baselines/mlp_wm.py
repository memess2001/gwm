"""Plain (non-geometric) world models: the control group of every kill test (HANDOFF section 4).

    E  encoder     history window o_{t-h+1..t} -> z_t in R^d
                   vector obs: MLP on the flattened window; pixels: frames stacked as channels, stride-2 conv
                   stack 64 -> 32 -> 16 -> 8 -> 4 (channels c, 2c, 4c, 8c; kernel 4, padding 1) + MLP head
    F  transition  residual MLP  z' = z + g([z, a])   (autonomous worlds: z' = z + g(z)); the last layer of g
                   is zero-initialised, so F starts as the identity (stable K-step open-loop rollouts at init)
    Init: all other layers N(0, gain^2 / fan_in) with zero biases (ReLU gain in hidden layers, gain 1 for the
    layers that output z or the observation); see ``_init_layer`` for why the torch default fails on pixels.
    D  decoder     z -> current frame o_t (recon only); vector: MLP; pixels: mirrored deconv stack, LINEAR output
                   by default (``pixel_out="linear"``; "sigmoid" squashes to [0, 1], see below)
    Pixel output (measured 2026-11-04, pendulum pixels64, size S, training loss / val nmse only): with a sigmoid
    output the MSE stalled on the mean-image plateau (val nmse 0.97 after 800 steps; 1.14 after 250 steps at lr
    1e-3 and 3e-3), even with the init fix: the frames are dark except for rod and bob, the background logits
    saturate, and the sigmoid starts every pixel at 0.5. With the linear output the autoencoder reaches nmse 0.56
    in 300 steps, and the full recon loss (K = 8, lambda_lat 0.1, 500 trajectories) val nmse 0.14 after 1500
    steps. Decoded pixels may leave [0, 1] slightly; clip only where a valid image is needed, never in the loss.

Variants (identical E and F architectures and widths; the two jepa variants simply have no decoder):
    recon     sum_{j=0..K} err(D(zh_{t+j}), o_{t+j}) + lambda_lat * sum_{j=1..K} MSE(zh_{t+j}, sg(E(w_{t+j})))
              with zh_t = E(w_t), zh_{t+j} = F(zh_{t+j-1}, a_{t+j-1}) (open loop in latent space), w_s = window
              ending at s, err = MSE (default) or the circular loss 1 - cos(scale * (pred - target)) (angle_raw).
    jepa-sym  w_inv * sum_{j=1..K} MSE(zh_{t+j}, E(w_{t+j}))         (no stop-gradient: symmetric, VICReg style;
              + w_var * V(E(w_{t+j}), j = 0..K) + w_cov * C(...)      gradients flow through both branches)
              VICReg terms over ALL K+1 encoder outputs. "jepa" is an alias (the only jepa before the cell 0 split).
    jepa-sg   w_inv * sum_{j=1..K} MSE(zh_{t+j}, sg(E(w_{t+j})))     (stop-gradient prediction targets)
              + w_var * V(E(w_t)) + w_cov * C(E(w_t))                 VICReg terms on the ONLINE encoder output only
              The target encodings run under no_grad (no backward through K encoder passes: cheaper per step).
              This is the stop-grad JEPA of the cell 0 reviewer pilot (cells/0/prereg.md section 0b).
              Its loss is not monotone early on: the variance hinge first spreads the latents, which raises the
              absolute invariance error, and only the online branch can shrink it. It also starts slower (pixels
              S: after 1000 steps the k = 1 latent prediction was no better than copying z_t; after 3000 steps the
              k = 8 error was 0.0075 vs 0.96 for copying), so short smoke runs understate it.
    V(z) = mean_i relu(gamma - sqrt(Var_batch(z_i) + eps))            (variance hinge, per time slice)
    C(z) = sum_{i != k} Cov_batch(z)_{ik}^2 / d                       (decorrelation, per time slice)
    VICReg defaults (w_inv, w_var, w_cov) = (25, 25, 1), gamma = 1, eps = 1e-4.
    Reductions: each MSE is a mean over batch and features; the step terms are SUMMED over j (as specified), the
    VICReg regularisers are AVERAGED over the time slices they use (each slice is one VICReg "branch" statistic;
    for stationary data the K+1-slice average of jepa-sym and the single slice of jepa-sg have the same mean).
    jepa-sg vs jepa-sym is a pre-registered factor of cell 0: same architecture, same weights, same data; only the
    gradient path through the targets (and hence which encoder outputs the regulariser can act on) differs.

Size ladder: ``build_wm(spec, variant, size)`` with size in S / M / L / XL targets 0.1M / 0.5M / 2M / 8M TOTAL
parameters of the RECON model (+-15 %) by scaling all widths jointly with fixed depth: vector models scale the
MLP width w (2 hidden layers per MLP); pixel models scale c and w = 4c together. The jepa model of the same size
reuses the recon widths and therefore has fewer parameters (no decoder); ``describe`` reports both
("params_jepa" is the count of either jepa variant: they share one architecture).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields, replace
from typing import Optional

import torch
import torch.nn.functional as Fn
from torch import nn

from .data import action_dim, obs_kind

__all__ = ["SIZES", "SIZE_TOL", "VARIANTS", "VARIANT_ALIASES", "WMConfig", "WorldModel", "build_wm",
           "canonical_variant", "is_jepa", "solve_widths", "count_params", "describe", "train_flops_per_sample",
           "effective_rank", "latent_stats", "sliding_windows", "vicreg_terms"]

SIZES = {"S": 100_000, "M": 500_000, "L": 2_000_000, "XL": 8_000_000}
SIZE_TOL = 0.15
PIXEL_MLP_RATIO = 4          # pixel models: MLP width (head, transition) = 4 x base conv channels
VARIANTS = ("recon", "jepa-sym", "jepa-sg")
VARIANT_ALIASES = {"jepa": "jepa-sym"}     # backward compatibility: the original jepa was the symmetric loss
LOSSES = ("mse", "circular")
PIXEL_OUTS = ("linear", "sigmoid")


def canonical_variant(variant: str) -> str:
    """'jepa' -> 'jepa-sym'; other valid names unchanged; ValueError for unknown names."""
    v = VARIANT_ALIASES.get(variant, variant)
    if v not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS} (or an alias in {sorted(VARIANT_ALIASES)})")
    return v


def is_jepa(variant: str) -> bool:
    """True for every latent-prediction variant (no decoder)."""
    return canonical_variant(variant) != "recon"


@dataclass
class WMConfig:
    obs_shape: tuple
    obs_kind: str                         # "vector" | "image"
    action_dim: int = 0                   # 0 = autonomous world
    variant: str = "recon"                # "recon" | "jepa-sym" | "jepa-sg" ("jepa" = alias of "jepa-sym")
    size: Optional[str] = None            # ladder label (informational once widths are set)
    latent_dim: int = 8
    history: int = 2
    K: int = 8                            # training horizon (open-loop steps in the loss)
    width: int = 128                      # MLP hidden width (vector E / D, transition, pixel heads)
    channels: int = 0                     # base conv channels c (image only)
    depth: int = 2                        # hidden layers per MLP (vector E / D and transition)
    norm: str = "layer"                   # MLP hidden layers: "layer" (LayerNorm) | "none"; convs have no norm
    loss: str = "mse"                     # recon reconstruction error: "mse" | "circular"
    pixel_out: str = "linear"             # image decoder output: "linear" | "sigmoid" (checkpoints before
                                          # 2026-11-04 had no such key and used sigmoid; encode.load handles that)
    angle_scale: float = 1.0              # circular loss: physical angle per model unit (std if standardised)
    lambda_lat: float = 0.1               # recon latent-consistency weight
    vicreg: tuple = (25.0, 25.0, 1.0)     # jepa (invariance, variance, covariance) weights
    vic_gamma: float = 1.0                # jepa target std per latent dimension
    vic_eps: float = 1e-4

    def __post_init__(self):
        self.obs_shape = tuple(int(s) for s in self.obs_shape)
        self.vicreg = tuple(float(v) for v in self.vicreg)
        self.variant = canonical_variant(self.variant)        # old "jepa" checkpoints load as jepa-sym
        if self.loss not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES}")
        if self.loss == "circular" and (self.variant != "recon" or self.obs_kind != "vector"):
            raise ValueError("loss='circular' is a reconstruction loss for vector (angle) observations: recon only")
        if self.pixel_out not in PIXEL_OUTS:
            raise ValueError(f"pixel_out must be one of {PIXEL_OUTS}")
        if self.obs_kind not in ("vector", "image"):
            raise ValueError("obs_kind must be 'vector' or 'image'")
        if self.obs_kind == "image":
            if len(self.obs_shape) not in (2, 3) or self.obs_shape[-1] % 16 or self.obs_shape[-2] % 16:
                raise ValueError("image observations must be (H, W) or (C, H, W) with H, W divisible by 16")
            if self.channels < 1:
                raise ValueError("image models need channels >= 1")
        if self.history < 1 or self.K < 1 or self.latent_dim < 1 or self.width < 1 or self.depth < 1:
            raise ValueError("history, K, latent_dim, width, depth must be >= 1")
        if self.norm not in ("layer", "none"):
            raise ValueError("norm must be 'layer' or 'none'")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["obs_shape"], d["vicreg"] = list(self.obs_shape), list(self.vicreg)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "WMConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def _init_layer(m: nn.Module, gain: float = math.sqrt(2.0)) -> None:
    """Signal-preserving init: weights N(0, gain^2 / fan_in), biases 0 (ReLU gain for hidden layers).

    PyTorch's default (kaiming-uniform with a = sqrt(5), uniform biases) shrinks the input-dependent part of sparse
    [0, 1] pixel inputs layer by layer until the biases dominate: at init the pixel latents then varied across
    inputs by only ~0.01 (measured), and recon models sat on the mean-image plateau. fan_in of a stride-s transposed
    conv is in_channels * (k / s)^2 (inputs that reach one output pixel).
    """
    if isinstance(m, nn.ConvTranspose2d):
        k, st = m.kernel_size[0], m.stride[0]
        fan_in = m.in_channels * max(1, (k // st)) ** 2
    elif isinstance(m, (nn.Linear, nn.Conv2d)):
        fan_in = m.weight[0].numel()
    else:
        return
    nn.init.normal_(m.weight, 0.0, gain / math.sqrt(fan_in))
    if m.bias is not None:
        nn.init.zeros_(m.bias)


def _last_layer(m: nn.Module) -> nn.Module:
    return [x for x in m.modules() if isinstance(x, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d))][-1]


def mlp(n_in: int, width: int, n_out: int, depth: int, norm: str) -> nn.Sequential:
    """[Linear -> (LayerNorm) -> SiLU] x depth -> Linear."""
    layers, d = [], n_in
    for _ in range(depth):
        layers.append(nn.Linear(d, width))
        if norm == "layer":
            layers.append(nn.LayerNorm(width))
        layers.append(nn.SiLU())
        d = width
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


class ConvEncoder(nn.Module):
    """[B, C_in, H, W] -> 4 x (Conv 4x4 / stride 2 / pad 1 + SiLU), channels c, 2c, 4c, 8c -> MLP head -> [B, d]."""

    def __init__(self, in_ch: int, c: int, hw: tuple, width: int, d: int, norm: str):
        super().__init__()
        ch = [in_ch, c, 2 * c, 4 * c, 8 * c]
        layers = []
        for i in range(4):
            layers += [nn.Conv2d(ch[i], ch[i + 1], 4, 2, 1), nn.SiLU()]
        self.conv = nn.Sequential(*layers)
        self.head = mlp(8 * c * (hw[0] // 16) * (hw[1] // 16), width, d, 1, norm)
        self.channels_last = False

    def forward(self, x):
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        return self.head(self.conv(x).flatten(1))


class ConvDecoder(nn.Module):
    """[B, d] -> MLP head -> [B, 8c, H/16, W/16] -> 4 x ConvTranspose 4x4 / stride 2 (mirror) -> linear | sigmoid."""

    def __init__(self, out_ch: int, c: int, hw: tuple, width: int, d: int, norm: str, out: str = "linear"):
        super().__init__()
        self.out = out
        self.shape0 = (8 * c, hw[0] // 16, hw[1] // 16)
        self.head = mlp(d, width, math.prod(self.shape0), 1, norm)
        ch = [8 * c, 4 * c, 2 * c, c, out_ch]
        layers = []
        for i in range(4):
            layers.append(nn.ConvTranspose2d(ch[i], ch[i + 1], 4, 2, 1))
            if i < 3:
                layers.append(nn.SiLU())
        self.deconv = nn.Sequential(*layers)
        self.channels_last = False

    def forward(self, z):
        x = Fn.silu(self.head(z)).view(-1, *self.shape0)
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        y = self.deconv(x)
        return torch.sigmoid(y) if self.out == "sigmoid" else y


class WorldModel(nn.Module):
    def __init__(self, cfg: WMConfig):
        super().__init__()
        self.cfg = cfg
        d, w, h = cfg.latent_dim, cfg.width, cfg.history
        if cfg.obs_kind == "image":
            self.img_ch = 1 if len(cfg.obs_shape) == 2 else cfg.obs_shape[0]
            hw = cfg.obs_shape[-2:]
            self.encoder = ConvEncoder(h * self.img_ch, cfg.channels, hw, w, d, cfg.norm)
            self.decoder = (ConvDecoder(self.img_ch, cfg.channels, hw, w, d, cfg.norm, cfg.pixel_out)
                            if cfg.variant == "recon" else None)
        else:
            n_obs = math.prod(cfg.obs_shape)
            self.encoder = mlp(h * n_obs, w, d, cfg.depth, cfg.norm)
            self.decoder = mlp(d, w, n_obs, cfg.depth, cfg.norm) if cfg.variant == "recon" else None
        self.transition = mlp(d + cfg.action_dim, w, d, cfg.depth, cfg.norm)
        self.apply(_init_layer)
        for m in (self.encoder, self.decoder):          # layers that produce z / the observation: gain 1
            if m is not None:
                _init_layer(_last_layer(m), gain=1.0)
        nn.init.zeros_(self.transition[-1].weight)        # F starts as the identity
        nn.init.zeros_(self.transition[-1].bias)

    def set_channels_last(self, on: bool = True) -> "WorldModel":
        """Run the conv stacks in channels-last memory format (about 30 % faster on CPU; numerics differ from the
        default layout in the last bits, so keep one setting per experiment). No effect for vector models."""
        if self.cfg.obs_kind == "image":
            for m in (self.encoder, self.decoder):
                if m is not None:
                    m.channels_last = bool(on)
                    m.to(memory_format=torch.channels_last if on else torch.contiguous_format)
        return self

    # ------------------------------------------------------------------ pieces
    def encode(self, window: torch.Tensor) -> torch.Tensor:
        """window [B, h, *obs_shape] (float, model units) -> z [B, d]."""
        if self.cfg.obs_kind == "image":
            H, W = self.cfg.obs_shape[-2:]
            return self.encoder(window.reshape(window.shape[0], -1, H, W))
        return self.encoder(window.flatten(1))

    def step(self, z: torch.Tensor, a: Optional[torch.Tensor] = None) -> torch.Tensor:
        """One latent transition z' = z + g([z, a])."""
        if self.cfg.action_dim:
            if a is None:
                raise ValueError("this world model needs actions")
            return z + self.transition(torch.cat([z, a], -1))
        return z + self.transition(z)

    def rollout(self, z0: torch.Tensor, actions: Optional[torch.Tensor] = None, steps: Optional[int] = None):
        """Open-loop latent rollout: [B, steps+1, d] including z0 (actions [B, steps, A] if the world has them)."""
        if actions is not None:
            steps = actions.shape[1] if steps is None else steps
        if steps is None:
            raise ValueError("give actions or steps")
        zs = [z0]
        for j in range(steps):
            zs.append(self.step(zs[-1], None if actions is None else actions[:, j]))
        return torch.stack(zs, 1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z [B, d] -> predicted current frame [B, *obs_shape] (model units; pixels ~[0, 1], exactly so only with
        pixel_out="sigmoid")."""
        if self.decoder is None:
            raise RuntimeError(f"the {self.cfg.variant} variant has no decoder")
        out = self.decoder(z)
        return out.reshape(z.shape[0], *self.cfg.obs_shape)

    def windows(self, frames: torch.Tensor) -> torch.Tensor:
        """frames [B, h+K', *obs] -> sliding windows [B, K'+1, h, *obs] (window j ends at frames[:, h-1+j])."""
        return sliding_windows(frames, self.cfg.history, use_view=frames.device.type != "mps")

    def recon_error(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Elementwise reconstruction error in model units (MSE or 1 - cos(scale * delta))."""
        if self.cfg.loss == "circular":
            return 1.0 - torch.cos(self.cfg.angle_scale * (pred - target))
        return (pred - target) ** 2

    # ------------------------------------------------------------------ losses
    def loss(self, frames: torch.Tensor, actions: Optional[torch.Tensor] = None):
        """Training loss for a batch: frames [B, h+K, *obs] (o_{t-h+1}..o_{t+K}), actions [B, K, A] or None.

        Returns (loss, logs) where logs holds detached scalar tensors (and, for both jepa variants, the per-dim
        batch std of z_t = E(w_t) as "latent_std" [d]).
        """
        cfg = self.cfg
        B, h, K = frames.shape[0], cfg.history, frames.shape[1] - cfg.history
        if K < 1:
            raise ValueError("frames must hold history + K >= 1 future frames")
        win = self.windows(frames)                                         # [B, K+1, h, *obs]
        if cfg.variant == "recon":
            z0 = self.encode(win[:, 0])
            with torch.no_grad():                                          # stop-gradient latent targets
                z_tgt = self.encode(win[:, 1:].flatten(0, 1)).view(B, K, -1)
            Z = self.rollout(z0, actions, K)                               # [B, K+1, d]
            pred = self.decode(Z.flatten(0, 1)).view(B, K + 1, *cfg.obs_shape)
            per_step = self.recon_error(pred, frames[:, h - 1:]).flatten(2).mean((0, 2))    # [K+1]
            l_rec = per_step.sum()
            l_lat = ((Z[:, 1:] - z_tgt) ** 2).mean((0, 2)).sum()
            loss = l_rec + cfg.lambda_lat * l_lat
            logs = {"loss": loss.detach(), "recon": l_rec.detach(), "recon_j0": per_step[0].detach(),
                    "latent": l_lat.detach()}
            return loss, logs
        if B < 2:
            raise ValueError("jepa needs batch size >= 2 (batch statistics)")
        if cfg.variant == "jepa-sym":
            Zenc = self.encode(win.flatten(0, 1)).view(B, K + 1, -1)       # all K+1 encoder outputs, with grad
            z0, z_tgt, z_reg = Zenc[:, 0], Zenc[:, 1:], Zenc
        else:                                                              # jepa-sg
            z0 = self.encode(win[:, 0])                                    # online branch (with grad)
            with torch.no_grad():                                          # stop-gradient prediction targets
                z_tgt = self.encode(win[:, 1:].flatten(0, 1)).view(B, K, -1)
            z_reg = z0[:, None]                                            # VICReg on the online output only
        Zpred = self.rollout(z0, actions, K)[:, 1:]
        l_inv = ((Zpred - z_tgt) ** 2).mean((0, 2)).sum()
        l_var, l_cov, std = vicreg_terms(z_reg, cfg.vic_gamma, cfg.vic_eps)
        wi, wv, wc = cfg.vicreg
        loss = wi * l_inv + wv * l_var + wc * l_cov
        logs = {"loss": loss.detach(), "inv": l_inv.detach(), "var": l_var.detach(), "cov": l_cov.detach(),
                "latent_std": std[0].detach()}                             # per-dim batch std of z_t = E(w_t)
        return loss, logs


def sliding_windows(frames: torch.Tensor, h: int, use_view: bool = True) -> torch.Tensor:
    """[B, L, *obs] -> [B, L-h+1, h, *obs]. ``use_view``: zero-copy strided view (unfold; ~5-10 % faster training
    steps on CPU); otherwise one copy built from plain slices (used on MPS, where strided views have been fragile)."""
    if use_view:
        return frames.unfold(1, h, 1).movedim(-1, 2)
    return torch.stack([frames[:, j:j + h] for j in range(frames.shape[1] - h + 1)], 1)


def vicreg_terms(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4):
    """VICReg variance hinge and covariance penalty, statistics over the batch for each time slice.

    z [B, S, d] -> (variance term, covariance term, per-slice std [S, d]); both terms averaged over the S slices.
    """
    B, S, d = z.shape
    zc = z - z.mean(0, keepdim=True)
    var = zc.pow(2).sum(0) / (B - 1)                                       # [S, d]
    std = torch.sqrt(var + eps)
    l_var = Fn.relu(gamma - std).mean()
    cov = torch.einsum("bsi,bsj->sij", zc, zc) / (B - 1)                   # [S, d, d]
    off = cov - torch.diag_embed(torch.diagonal(cov, dim1=-2, dim2=-1))
    l_cov = off.pow(2).sum((-2, -1)).mean() / d
    return l_var, l_cov, std


@torch.no_grad()
def effective_rank(z: torch.Tensor) -> float:
    """Roy & Vetterli (2007) / RankMe effective rank exp(H(p)), p = singular values / their sum, of centred z [N, d].
    Computed on the CPU in float64 (MPS has no float64)."""
    z = z.detach().to("cpu", torch.float64)
    s = torch.linalg.svdvals(z - z.mean(0, keepdim=True))
    p = s / s.sum().clamp_min(1e-30)
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


@torch.no_grad()
def latent_stats(z: torch.Tensor) -> dict:
    """Collapse diagnostics for latents z [N, d]: per-dim std, min std, effective rank (no geometry beyond that)."""
    z = z.detach().to("cpu", torch.float64)
    std = z.std(0)
    return {"latent_std": [float(v) for v in std], "latent_std_min": float(std.min()),
            "effective_rank": effective_rank(z)}


# ---------------------------------------------------------------------- size ladder
def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _base_config(spec: dict, variant: str, latent_dim: int, history: int, K: int, **kw) -> WMConfig:
    kind = obs_kind(spec)
    return WMConfig(obs_shape=tuple(spec["obs_shape"]), obs_kind=kind, action_dim=action_dim(spec), variant=variant,
                    latent_dim=latent_dim, history=history, K=K, **kw)


def _meta_count(cfg: WMConfig) -> int:
    with torch.device("meta"):
        return count_params(WorldModel(cfg))


def solve_widths(spec: dict, size: str, latent_dim: int = 8, history: int = 2, depth: int = 2,
                 norm: str = "layer") -> dict:
    """Widths whose RECON model has the parameter count closest to ``SIZES[size]``.

    Vector observations: one MLP width w. Images: base channels c and MLP width w ~ PIXEL_MLP_RATIO * c, found by
    bisection on c (w = ratio * c) followed by a local search over (c +- 1, w within +-3 of ratio * c).
    Returns {"width", "channels", "params", "target", "rel_err"}.
    """
    if size not in SIZES:
        raise ValueError(f"size must be one of {list(SIZES)}")
    target = SIZES[size]
    base = _base_config(spec, "recon", latent_dim, history, 1, depth=depth, norm=norm,
                        channels=1 if obs_kind(spec) == "image" else 0)

    def n(width, channels=0):
        return _meta_count(replace(base, width=int(width), channels=int(channels)))

    def bisect(f, lo, hi):          # smallest x in [lo, hi] with f(x) >= target (f increasing)
        while lo < hi:
            mid = (lo + hi) // 2
            lo, hi = (mid + 1, hi) if f(mid) < target else (lo, mid)
        return lo

    if base.obs_kind == "vector":
        w = bisect(lambda x: n(x), 1, 1 << 15)
        cands = [(x, 0) for x in (w - 1, w) if x >= 1]
    else:
        r = PIXEL_MLP_RATIO
        c = bisect(lambda x: n(r * x, x), 1, 1024)
        cands = [(ww, cc) for cc in (c - 1, c, c + 1) if cc >= 1 for ww in range(max(1, r * cc - 3), r * cc + 4)]
    best = min(cands, key=lambda wc: (abs(n(*wc) - target), wc))
    p = n(*best)
    return {"width": best[0], "channels": best[1], "params": p, "target": target, "rel_err": p / target - 1.0}


def build_wm(spec: dict, variant: str = "recon", size: Optional[str] = "S", latent_dim: int = 8, history: int = 2,
             K: int = 8, width: Optional[int] = None, channels: Optional[int] = None, **kw) -> WorldModel:
    """World model for a zoo ``spec`` (``World.spec(obs_mode)``); widths from the ladder unless given explicitly.

    Extra keyword arguments go to ``WMConfig`` (loss, angle_scale, lambda_lat, vicreg, depth, norm, ...).
    """
    depth, norm = kw.get("depth", 2), kw.get("norm", "layer")
    if width is None or (obs_kind(spec) == "image" and channels is None):
        if size is None:
            raise ValueError("give a ladder size or explicit widths")
        sol = solve_widths(spec, size, latent_dim, history, depth, norm)
        width = sol["width"] if width is None else width
        channels = sol["channels"] if channels is None else channels
    cfg = _base_config(spec, variant, latent_dim, history, K, size=size, width=int(width),
                       channels=int(channels or 0), **kw)
    return WorldModel(cfg)


def describe(model: WorldModel) -> dict:
    """Parameter counts per module (and what the same-width model of the other variant would have)."""
    parts = {"encoder": count_params(model.encoder), "transition": count_params(model.transition),
             "decoder": count_params(model.decoder) if model.decoder is not None else 0}
    other = replace(model.cfg, variant="jepa-sym" if model.cfg.variant == "recon" else "recon", loss="mse")
    n_other = _meta_count(other)
    recon_n = count_params(model) if model.cfg.variant == "recon" else n_other
    jepa_n = count_params(model) if is_jepa(model.cfg.variant) else n_other
    return {"params": count_params(model), **{f"params_{k}": v for k, v in parts.items()},
            "params_recon": recon_n, "params_jepa": jepa_n, "width": model.cfg.width, "channels": model.cfg.channels}


def train_flops_per_sample(cfg: WMConfig, batch: int = 4) -> dict:
    """Approximate FLOPs per training sample (one K-step loss, forward + backward), counted with torch's
    FlopCounterMode (matmul / conv only; elementwise ops ignored). Also forward FLOPs per frame for E, F, D."""
    from torch.utils.flop_counter import FlopCounterMode

    torch_state = torch.random.get_rng_state()
    try:
        model = WorldModel(cfg)
        frames = torch.rand(batch, cfg.history + cfg.K, *cfg.obs_shape)
        acts = torch.rand(batch, cfg.K, cfg.action_dim) if cfg.action_dim else None
        with FlopCounterMode(display=False) as fc:
            loss, _ = model.loss(frames, acts)
            loss.backward()
        train = fc.get_total_flops() / batch
        with torch.no_grad():
            win = frames[:, :cfg.history]
            with FlopCounterMode(display=False) as fe:
                z = model.encode(win)
            with FlopCounterMode(display=False) as ft:
                model.step(z, None if acts is None else acts[:, 0])
            fd = None
            if model.decoder is not None:
                with FlopCounterMode(display=False) as fdm:
                    model.decode(z)
                fd = fdm.get_total_flops() / batch
    finally:
        torch.random.set_rng_state(torch_state)
    return {"train_flops_per_sample": train, "fwd_flops_encoder": fe.get_total_flops() / batch,
            "fwd_flops_transition": ft.get_total_flops() / batch, "fwd_flops_decoder": fd}
