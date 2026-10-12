"""Task A models: predict path / TS spectra from endpoint spectra only.

All models output a RESIDUAL R_theta; the final prediction is assembled with
the endpoint-exact form (done in train_taskA.combine):

    X_hat(s) = (1-s) X_R + s X_P + s(1-s) R_theta(X_R, X_P, comp, s)

so X_hat(0) == X_R and X_hat(1) == X_P exactly, in normalized space.

Variants:
  - PathMLP    (A2): endpoint-conditioned MLP trunk -> 3x200 residual.
  - PathResNet (A3): MLP encoder -> latent -> 1D ResNet decoder -> 3x200.
  - TSHead     (A2-TS): endpoint-conditioned MLP -> TS residual + s_hat_TS.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_EDGE = 3
SPEC_LEN = 200
COMP_DIM = 7
N_FOURIER = 4                      # sin/cos pairs for k = 1..4
S_FEAT_DIM = 1 + 2 * N_FOURIER     # 9
FLAT = N_EDGE * SPEC_LEN           # 600
PATH_IN_DIM = 2 * FLAT + COMP_DIM + S_FEAT_DIM   # 1216
TS_IN_DIM = 2 * FLAT + COMP_DIM                  # 1207


def s_features(s):
    """s: (B,) tensor -> (B, S_FEAT_DIM): [s, sin/cos(2 pi k s), k=1..4]."""
    feats = [s.unsqueeze(-1)]
    for k in range(1, N_FOURIER + 1):
        w = 2.0 * math.pi * k * s
        feats.append(torch.sin(w).unsqueeze(-1))
        feats.append(torch.cos(w).unsqueeze(-1))
    return torch.cat(feats, dim=-1)


def _mlp(in_dim, hidden, n_layers, out_dim):
    layers, d = [], in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden), nn.GELU()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class PathMLP(nn.Module):
    """A2: endpoint-conditioned MLP -> spectral residual."""

    def __init__(self, hidden=512, n_layers=2):
        super().__init__()
        self.trunk = _mlp(PATH_IN_DIM, hidden, n_layers, FLAT)

    def forward(self, xR, xP, comp, s):
        """xR/xP: (B,3,200) normalized, comp: (B,7), s: (B,). -> (B,3,200)."""
        inp = torch.cat([xR.flatten(1), xP.flatten(1), comp, s_features(s)], dim=1)
        return self.trunk(inp).view(-1, N_EDGE, SPEC_LEN)


class ResBlock1D(nn.Module):
    def __init__(self, ch, kernel=5):
        super().__init__()
        p = kernel // 2
        self.conv1 = nn.Conv1d(ch, ch, kernel, padding=p)
        self.conv2 = nn.Conv1d(ch, ch, kernel, padding=p)
        self.gn1 = nn.GroupNorm(8, ch)
        self.gn2 = nn.GroupNorm(8, ch)

    def forward(self, h):
        y = self.conv1(F.gelu(self.gn1(h)))
        y = self.conv2(F.gelu(self.gn2(y)))
        return h + y


class PathResNet(nn.Module):
    """A3: latent-conditioned 1D ResNet continuous decoder."""

    def __init__(self, hidden=512, latent=256, ch=64, len0=25, n_blocks=3):
        super().__init__()
        self.encoder = _mlp(PATH_IN_DIM, hidden, 2, latent)
        self.to_seq = nn.Linear(latent, ch * len0)
        self.ch, self.len0 = ch, len0
        # len0=25 -> 50 -> 100 -> 200
        self.blocks = nn.ModuleList([ResBlock1D(ch) for _ in range(n_blocks)])
        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv1d(ch, N_EDGE, 5, padding=2)

    def forward(self, xR, xP, comp, s):
        inp = torch.cat([xR.flatten(1), xP.flatten(1), comp, s_features(s)], dim=1)
        z = self.encoder(inp)
        h = self.to_seq(z).view(-1, self.ch, self.len0)
        for blk in self.blocks:
            h = F.interpolate(h, scale_factor=2, mode='linear',
                              align_corners=False)
            h = blk(h)
        h = self.out_conv(F.gelu(self.out_norm(h)))
        assert h.shape[-1] == SPEC_LEN, h.shape
        return h


class TSHead(nn.Module):
    """A2-TS: dedicated TS head -> TS residual + s_hat_TS (sigmoid).

    TS spectrum assembled with the same endpoint-exact form at s_hat_TS:
        X_hat_TS = (1-s)X_R + s X_P + s(1-s) R,  s = sigmoid(logit)
    """

    def __init__(self, hidden=512, n_layers=2):
        super().__init__()
        self.trunk = _mlp(TS_IN_DIM, hidden, n_layers, hidden)
        self.res_head = nn.Linear(hidden, FLAT)
        self.s_head = nn.Linear(hidden, 1)

    def forward(self, xR, xP, comp):
        inp = torch.cat([xR.flatten(1), xP.flatten(1), comp], dim=1)
        h = self.trunk(inp)
        resid = self.res_head(h).view(-1, N_EDGE, SPEC_LEN)
        s_hat = torch.sigmoid(self.s_head(h)).squeeze(-1)   # (B,)
        return resid, s_hat


def count_params(model):
    return sum(p.numel() for p in model.parameters())
