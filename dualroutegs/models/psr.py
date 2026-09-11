"""Phase-Aware State Routing, manuscript Eqs. (8)-(27)."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .scan import affine_scan, compact_subspace_scan


def straight_through(hard, soft):
    return hard.to(soft.dtype).detach() - soft.detach() + soft


def scalar_embedding(value, dim):
    frequency = torch.exp(-math.log(10000) * torch.arange(dim // 2, device=value.device, dtype=torch.float32) / max(dim // 2 - 1, 1))
    phase = value.float()[:, None] * frequency[None]
    return torch.cat([phase.sin(), phase.cos()], -1)


class PhaseReliability(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_slope = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, log_snr):
        return torch.sigmoid(F.softplus(self.raw_slope) * log_snr + self.bias)


class PhaseAwareStateRouting(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d, h = cfg["dim"], cfg["state_dim"]
        self.groups = cfg["subspaces"]
        self.group_dim = h // self.groups
        self.heads = cfg["heads"]
        self.write_projection = nn.Linear(d, h)
        self.selective = nn.Linear(d, 2*h)
        self.dwconv = nn.Conv1d(2*h, 2*h, 3, padding=1, groups=2*h)
        self.A_log = nn.Parameter(torch.zeros(h))
        self.reliability = PhaseReliability()
        self.confidence_coeff = nn.Parameter(torch.tensor([1.0, 1.0, 0.2]))
        self.confidence_bias = nn.Parameter(torch.tensor(0.0))
        self.phase_bias = nn.Sequential(nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.phase_bias[-1].weight)
        nn.init.zeros_(self.phase_bias[-1].bias)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(h, d, bias=False)
        self.v = nn.Linear(h, d, bias=False)
        self.out = nn.Linear(d, d)

    def forward(self, x, log_snr, memory=None):
        cfg = self.cfg
        static = cfg["interaction"] == "static_state"
        # Keep ZOH, state norms, confidence and recurrence in fp32 under AMP.
        params = self.dwconv(self.selective(x).transpose(1, 2)).transpose(1, 2)
        projected = self.write_projection(x)
        with torch.autocast(x.device.type, enabled=False):
            delta, B = params.float().chunk(2, -1)
            delta = F.softplus(delta)
            A = -F.softplus(self.A_log.float()).clamp_min(1e-6)
            exponent = delta * A
            decay = torch.exp(exponent)
            B_bar = torch.expm1(exponent) / A * B
            candidate = B_bar * projected.float()  # diagonal B realization
            omega = torch.linalg.vector_norm(candidate, dim=-1)
            mean_write = candidate.detach().mean(1)
            previous = torch.zeros_like(mean_write) if memory is None else memory.detach()
            memory = cfg["memory_decay"] * previous + (1-cfg["memory_decay"]) * mean_write
            dot = (candidate * memory[:, None]).sum(-1, keepdim=True)
            projection = dot / (memory.square().sum(-1)[:, None, None] + 1e-8) * memory[:, None]
            innovation = torch.linalg.vector_norm(candidate - projection, dim=-1) / (omega + 1e-8)
            strength = (omega - omega.mean(1, keepdim=True)) / (omega.std(1, keepdim=True, correction=0) + 1e-6)
            chi = self.reliability(log_snr.float())
            if static or not cfg.get("phase_enabled", True):
                chi = torch.ones_like(chi)
            coeff = self.confidence_coeff.float()
            bias = self.confidence_bias + self.phase_bias(scalar_embedding(log_snr, 32)).flatten().float()
            if static or not cfg.get("phase_enabled", True):
                bias = self.confidence_bias.expand_as(bias)
            p = torch.sigmoid(coeff[0]*strength + chi[:, None]*(coeff[1]*innovation + coeff[2]*strength*innovation) - bias[:, None])
            minimum = cfg["g_min_early"] + chi * (cfg["g_min_late"] - cfg["g_min_early"])
            amplitude = minimum[:, None] + (1-minimum[:, None])*p
            if not cfg.get("amplitude_enabled", True):
                amplitude = torch.ones_like(amplitude)
            base = cfg["r_min"]
            thresholds = torch.arange(1, self.groups-base+1, device=x.device).float() / (self.groups-base+1)
            thresholds = (thresholds[None, None] + cfg["rank_phase_shift"]*(1-chi[:, None, None])).clamp(0, 1)
            extra_soft = torch.sigmoid(cfg["rank_sharpness"]*(p[..., None]-thresholds))
            extra_hard = p[..., None] >= thresholds
            ones = torch.ones(*p.shape, base, device=x.device)
            soft = torch.cat([ones, extra_soft], -1)
            hard = torch.cat([ones.bool(), extra_hard], -1)
            if not cfg.get("subspace_enabled", True):
                soft, hard = torch.ones_like(soft), torch.ones_like(hard)
            mask = straight_through(hard, soft).repeat_interleave(self.group_dim, -1)
            write = amplitude[..., None] * candidate
            if self.training or not cfg.get("compact_scan", True):
                purified = affine_scan(1+mask*(decay-1), mask*write)
            else:
                purified = compact_subspace_scan(decay, write, hard, self.group_dim)
            threshold_f = (cfg["query_threshold"] + cfg["query_phase_shift"]*(1-chi)).clamp(0, 1)
            query_soft = torch.sigmoid(cfg["query_sharpness"]*(p-threshold_f[:, None]))
            query_hard = p >= threshold_f[:, None]
            if not cfg.get("query_enabled", True):
                query_soft, query_hard = torch.ones_like(query_soft), torch.ones_like(query_hard)
            query_gate = straight_through(query_hard, query_soft)

        keys, values = self.k(purified.to(x.dtype)), self.v(purified.to(x.dtype))
        batch, tokens, width = x.shape
        head_dim = width // self.heads
        keys = keys.reshape(batch, tokens, self.heads, head_dim).transpose(1, 2)
        values = values.reshape(batch, tokens, self.heads, head_dim).transpose(1, 2)
        if self.training:
            # Dense straight-through relaxation at training, hard gather at inference.
            queries = self.q(x*query_gate[..., None].to(x.dtype))
            queries = queries.reshape(batch, tokens, self.heads, head_dim).transpose(1, 2)
            attention = F.scaled_dot_product_attention(queries, keys, values)
            retrieved = self.out(attention.transpose(1, 2).reshape(batch, tokens, width))
            output = query_gate[..., None]*retrieved + (1-query_gate[..., None])*x
        else:
            output = x.clone()
            for bi in range(batch):
                ids = query_hard[bi].nonzero(as_tuple=True)[0]
                if ids.numel() == 0:
                    continue  # Valid empty-query route: all tokens bypass retrieval.
                queries = self.q(x[bi:bi+1, ids]).reshape(1, len(ids), self.heads, head_dim).transpose(1, 2)
                attention = F.scaled_dot_product_attention(queries, keys[bi:bi+1], values[bi:bi+1])
                output[bi, ids] = self.out(attention.transpose(1, 2).reshape(1, len(ids), width))[0]
        diagnostics = {
            "confidence": p, "innovation": innovation,
            "reliability": chi, "active_subspaces": hard.sum(-1),
            "query_fraction": query_hard.float().mean(1),
        }
        return output.to(x.dtype), memory, diagnostics
