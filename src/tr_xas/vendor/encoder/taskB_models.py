"""Task B models: inverse problem (single-frame spectrum -> reaction coordinate).

Shared building blocks (per the Task B spec):
  - one shared 1D CNN edge encoder applied to each 200-point edge spectrum;
  - learned edge-identity embeddings, presence/absence embeddings;
  - learned role embeddings (R=0, current=1, P=2) for endpoint-conditioned models;
  - small composition MLP;
  - regression / classification heads.

All models take:
  x    : (B, n_spec, 3, 200) normalized spectra (n_spec = 1/2/3 depending on model)
  pres : (B, n_spec, 3)      edge presence flags
  comp : (B, 7)              composition vector
Forbidden inputs (frame index, other frames, current structure/energy) are
never passed to any model.
"""

import torch
import torch.nn as nn

N_BINS = 20
ROLE_R, ROLE_K, ROLE_P = 0, 1, 2


class EdgeEncoder(nn.Module):
    """1D CNN over one 200-point edge spectrum -> embedding."""

    def __init__(self, emb=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, padding=4), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=9, padding=4), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(4), nn.Flatten(),
            nn.Linear(64 * 4, emb), nn.GELU(),
        )

    def forward(self, x):  # x: (M, 200)
        return self.net(x.unsqueeze(1))


class SpectrumEmbedder(nn.Module):
    """Encode a (3,200) 3-edge spectrum + presence + role -> fixed vector."""

    def __init__(self, emb=128, fuse_dim=256):
        super().__init__()
        self.encoder = EdgeEncoder(emb)
        self.edge_emb = nn.Parameter(torch.zeros(3, emb))
        self.absent_emb = nn.Parameter(torch.zeros(3, emb))
        self.role_emb = nn.Parameter(torch.zeros(3, emb))
        self.fuse = nn.Sequential(
            nn.Linear(3 * emb, fuse_dim), nn.GELU(),
            nn.Linear(fuse_dim, fuse_dim), nn.GELU(),
        )
        for p in (self.edge_emb, self.absent_emb, self.role_emb):
            nn.init.normal_(p, std=0.02)

    def forward(self, x, pres, role):
        # x: (B, 3, 200), pres: (B, 3), role: int
        B = x.shape[0]
        h = self.encoder(x.reshape(B * 3, -1)).reshape(B, 3, -1)
        h = h + self.edge_emb[None] + self.role_emb[role][None, None]
        m = pres[..., None]
        h = m * h + (1.0 - m) * self.absent_emb[None]
        return self.fuse(h.reshape(B, -1))


class CompEncoder(nn.Module):
    def __init__(self, out=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, out), nn.GELU(), nn.Linear(out, out), nn.GELU())

    def forward(self, c):
        return self.net(c)


class _Base(nn.Module):
    """Shared trunk; subclasses define which spectra are embedded."""

    n_spec = 1

    def __init__(self, emb=128, fuse_dim=256, comp_dim=64, trunk=256, dropout=0.1):
        super().__init__()
        self.embed = SpectrumEmbedder(emb, fuse_dim)
        self.comp_enc = CompEncoder(comp_dim)
        self.trunk = nn.Sequential(
            nn.Linear(self.n_spec * fuse_dim + comp_dim, trunk), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(trunk, trunk), nn.GELU(),
        )
        self.trunk_dim = trunk

    def features(self, x, pres, comp, roles):
        feats = [self.embed(x[:, i], pres[:, i], r) for i, r in enumerate(roles)]
        return self.trunk(torch.cat(feats + [self.comp_enc(comp)], dim=1))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class B1Model(_Base):
    """Endpoint-only: X_R, X_P, composition -> conditional-mean point predictions.

    Heads: s (regression mean), abs_delta_s, d_TS, delta_E_R, near_TS logit.
    """

    n_spec = 2

    def __init__(self, **kw):
        super().__init__(**kw)
        t = self.trunk_dim
        self.head_s = nn.Linear(t, 1)
        self.head_abs = nn.Linear(t, 1)
        self.head_dts = nn.Linear(t, 1)
        self.head_dE = nn.Linear(t, 1)
        self.head_near = nn.Linear(t, 1)

    def forward(self, x, pres, comp):
        h = self.features(x, pres, comp, [ROLE_R, ROLE_P])
        return {
            's_mean': self.head_s(h).squeeze(-1),
            'abs_delta_s': self.head_abs(h).squeeze(-1),
            'd_TS': self.head_dts(h).squeeze(-1),
            'delta_E_R': self.head_dE(h).squeeze(-1),
            'near_logit': self.head_near(h).squeeze(-1),
        }


class B2Model(_Base):
    """Snapshot-only: X_k, composition -> binned p(s) + regressions + near-TS.

    Heads: s logits (N_BINS), abs_delta_s, d_TS, delta_E_R, near_TS logit.
    """

    n_spec = 1

    def __init__(self, n_bins=N_BINS, **kw):
        super().__init__(**kw)
        t = self.trunk_dim
        self.n_bins = n_bins
        self.head_s = nn.Linear(t, n_bins)
        self.head_abs = nn.Linear(t, 1)
        self.head_dts = nn.Linear(t, 1)
        self.head_dE = nn.Linear(t, 1)
        self.head_near = nn.Linear(t, 1)

    def forward(self, x, pres, comp):
        h = self.features(x, pres, comp, [ROLE_K])
        return {
            's_logits': self.head_s(h),
            'abs_delta_s': self.head_abs(h).squeeze(-1),
            'd_TS': self.head_dts(h).squeeze(-1),
            'delta_E_R': self.head_dE(h).squeeze(-1),
            'near_logit': self.head_near(h).squeeze(-1),
        }


class B3Model(_Base):
    """Endpoint-conditioned snapshot: X_R, X_k, X_P, composition (main model).

    Heads: s logits (N_BINS), signed_delta_s, d_TS, delta_E_R,
           pre_or_post logits (3), near_TS logit.
    """

    n_spec = 3

    def __init__(self, n_bins=N_BINS, **kw):
        super().__init__(**kw)
        t = self.trunk_dim
        self.n_bins = n_bins
        self.head_s = nn.Linear(t, n_bins)
        self.head_signed = nn.Linear(t, 1)
        self.head_dts = nn.Linear(t, 1)
        self.head_dE = nn.Linear(t, 1)
        self.head_prepost = nn.Linear(t, 3)
        self.head_near = nn.Linear(t, 1)

    def forward(self, x, pres, comp):
        h = self.features(x, pres, comp, [ROLE_R, ROLE_K, ROLE_P])
        return {
            's_logits': self.head_s(h),
            'signed_delta_s': self.head_signed(h).squeeze(-1),
            'd_TS': self.head_dts(h).squeeze(-1),
            'delta_E_R': self.head_dE(h).squeeze(-1),
            'prepost_logits': self.head_prepost(h),
            'near_logit': self.head_near(h).squeeze(-1),
        }
