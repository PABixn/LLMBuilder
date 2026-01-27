import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools as it

from model.loader import *
from model.kvcache import KVCache

class ConfigurableGPT(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList()
        })

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.weight_tying:
            self.tie_weights()

        self.in_norm = LearnableRMSNorm(config.n_embd)
        self.out_norm = LearnableRMSNorm(config.n_embd)
        
        attn_idx = it.count()

        for lay in config.blocks:
            self.transformer.h.append(ConfigurableBlock(config.n_embd, attn_idx, lay))

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        x = self.transformer.wte(idx)
        x = self.in_norm(x)

        for block in self.transformer.h:
            x = block(x, kv_cache)

        x = self.out_norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            return logits

    def tie_weights(self):
        self.transformer.wte.weight = self.lm_head.weight


class ConfigurableBlock(nn.Module):
    def __init__(self, dim, attn_idx: it.count, config: Block):
        super().__init__()

        self.layer = nn.ModuleList()

        for lay in config.components:
            if isinstance(lay, AttentionComponent):
                self.layer.append(CausalSelfAttention(next(attn_idx), dim, lay.attention))

            elif isinstance(lay, MLPComponent):
                self.layer.append(ConfigurableMLP(dim, lay.mlp))

            elif isinstance(lay, ActivationComponent):
                self.layer.append(get_activation(lay.activation.type))

            elif isinstance(lay, NormComponent):
                self.layer.append(get_norm(lay.norm, dim))
            else:
                raise ValueError(f"Unknown Block layer type: {type(lay)}")

    def forward(self, x, kv_cache: KVCache):
        for layer in self.layer:
            if isinstance(layer, CausalSelfAttention):
                x = x + layer(x, kv_cache)
            elif isinstance(layer, ConfigurableMLP):
                x = x + layer(x)
            else:
                x = layer(x)
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, attn_idx: int, dim: int, config: Attention):
        super().__init__()

        self.attn_idx = attn_idx
        self.n_head: int = config.n_head
        self.n_kv_head: int = config.n_kv_head
        self.n_embd = dim

        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        self.head_dim = self.n_embd // self.n_head

        if self.head_dim % 2 != 0:
            raise ValueError("Rotary embeddings require an even head_dim.")

        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        self.q_norm = LearnableRMSNorm(self.head_dim)
        self.k_norm = LearnableRMSNorm(self.head_dim)
        self._rope_cache = {}

    def forward(self, x, kv_cache: KVCache):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos, sin = self._precompute_rotary_embeddings(T0 + T, x.device, x.dtype)
        cos, sin = cos[:, T0:T0 + T], sin[:, T0:T0 + T]

        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.attn_idx, k, v)

        Tq = q.size(2)
        Tk = k.size(2)

        enable_gqa = self.n_head != self.n_kv_head

        if kv_cache is None or Tq == Tk:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif Tq == 1:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)
            prefix_len = Tk - Tq
            attn_mask[:, :prefix_len] = True
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))

            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)

        return y

    def _precompute_rotary_embeddings(self, seq_len: int, device: torch.device, dtype: torch.dtype, base=10000):
        key = (seq_len, device.type, device.index, dtype)
        cached = self._rope_cache.get(key)
        if cached is not None:
            return cached

        channel_range = torch.arange(0, self.head_dim, 2, dtype=dtype, device=device)
        inv_freq = 1.0 / (base ** (channel_range / self.head_dim))

        t = torch.arange(seq_len, dtype=dtype, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]

        self._rope_cache[key] = (cos, sin)

        return cos, sin

class ConfigurableMLP(nn.Module):
    def __init__(self, dim: int, config: MLP):
        super().__init__()

        self.layer = nn.ModuleList()

        current_dim: int = dim

        for idx, lay in enumerate(config.sequence):
            if isinstance(lay, LinearStep):
                in_dim, out_dim = dim, dim

                if idx == 0:
                    out_dim *= config.multiplier
                elif idx == len(config.sequence) - 1:
                    in_dim *= config.multiplier
                else:
                    in_dim *= config.multiplier
                    out_dim *= config.multiplier
                self.layer.append(nn.Linear(int(in_dim), int(out_dim), bias=lay.linear.bias))
                current_dim = out_dim

            elif isinstance(lay, NormComponent):
                self.layer.append(get_norm(lay.norm, int(current_dim)))

            elif isinstance(lay, ActivationComponent):
                self.layer.append(get_activation(lay.activation.type))

            else:
                raise ValueError(f"Unknown MLP layer type: {type(lay)}")

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return x

class LearnableRMSNorm(nn.Module):
    def __init__(self, dim: int, eps=1e-6):
        super().__init__()

        self.dim = dim
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), weight=self.weight, eps=self.eps)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"

class StaticRMSNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()

        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)

    def extra_repr(self) -> str:
        return f"eps={self.eps}"

class SquaredReLU(nn.Module):
    def forward(self, x):
        return F.relu(x).square()


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4

    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)
    out = out.to(x.dtype)

    return out

def get_norm(norm: Norm, dim: int) -> nn.Module:
    if isinstance(norm, model_loader.RMSNorm):
        if norm.learnable_gamma:
            return LearnableRMSNorm(dim)
        else:
            return StaticRMSNorm()

    if isinstance(norm, LayerNorm):
        return nn.LayerNorm(dim)
    raise ValueError(f"Unknown norm config: {type(norm)}")

def get_activation(act: str) -> nn.Module:
    if act == "gelu":
        return nn.GELU(approximate="tanh")
    if act == "relu":
        return nn.ReLU()
    if act == "squared_relu":
        return SquaredReLU()
    if act == "silu":
        return nn.SiLU()
    if act == "tanh":
        return nn.Tanh()
    if act == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unknown activation config: {act}")
