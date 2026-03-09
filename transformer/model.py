# transformer/model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.data import PAD_IDX


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.h, self.dk = num_heads, d_model // num_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def _split(self, x):
        B, S, _ = x.size()
        return x.view(B, S, self.h, self.dk).transpose(1, 2)

    def forward(self, q, k, v, mask=None, return_weights=False):
        Q, K, V = self._split(self.Wq(q)), self._split(self.Wk(k)), self._split(self.Wv(v))
        sc = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dk)
        if mask is not None:
            sc = sc.masked_fill(mask == 0, float('-inf'))
        w = torch.nan_to_num(F.softmax(sc, dim=-1), nan=0.0)
        out = torch.matmul(w, V)
        B, _, S, _ = out.size()
        res = self.Wo(out.transpose(1, 2).contiguous().view(B, S, -1))
        return (res, w) if return_weights else res


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, x): return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nh, d_ff, drop):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, nh)
        self.ff = FeedForward(d_model, d_ff, drop)
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask):
        x = self.n1(x + self.drop(self.attn(x, x, x, mask)))
        return self.n2(x + self.drop(self.ff(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nh, d_ff, drop):
        super().__init__()
        self.sa = MultiHeadAttention(d_model, nh)
        self.ca = MultiHeadAttention(d_model, nh)
        self.ff = FeedForward(d_model, d_ff, drop)
        self.n1, self.n2, self.n3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(drop)

    def forward(self, x, enc, sm, tm, return_attention=False):
        x = self.n1(x + self.drop(self.sa(x, x, x, tm)))
        co, cw = self.ca(x, enc, enc, sm, return_weights=True)
        x = self.n2(x + self.drop(co))
        x = self.n3(x + self.drop(self.ff(x)))
        return (x, cw) if return_attention else x


class Encoder(nn.Module):
    def __init__(self, vsz, d_model, nh, d_ff, nl, drop):
        super().__init__()
        self.emb = nn.Embedding(vsz, d_model, padding_idx=PAD_IDX)
        self.pe = PositionalEncoding(d_model, drop)
        self.layers = nn.ModuleList([EncoderLayer(d_model, nh, d_ff, drop) for _ in range(nl)])
        self.norm = nn.LayerNorm(d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, src, mask):
        x = self.pe(self.emb(src) * self.scale)
        for l in self.layers: x = l(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vsz, d_model, nh, d_ff, nl, drop):
        super().__init__()
        self.emb = nn.Embedding(vsz, d_model, padding_idx=PAD_IDX)
        self.pe = PositionalEncoding(d_model, drop)
        self.layers = nn.ModuleList([DecoderLayer(d_model, nh, d_ff, drop) for _ in range(nl)])
        self.norm = nn.LayerNorm(d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, tgt, enc, sm, tm, return_attention=False):
        x = self.pe(self.emb(tgt) * self.scale)
        last_attn = None
        for l in self.layers:
            if return_attention:
                x, last_attn = l(x, enc, sm, tm, return_attention=True)
            else:
                x = l(x, enc, sm, tm)
        x = self.norm(x)
        return (x, last_attn) if return_attention else x


class Transformer(nn.Module):
    def __init__(self, src_vsz, tgt_vsz, d_model=128, num_heads=4, d_ff=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vsz, d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(tgt_vsz, d_model, num_heads, d_ff, num_layers, dropout)
        self.proj = nn.Linear(d_model, tgt_vsz)
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def make_src_mask(self, src):
        return (src != PAD_IDX).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        L = tgt.size(1)
        pm = (tgt != PAD_IDX).unsqueeze(1).unsqueeze(2)
        cm = torch.tril(torch.ones(L, L, device=tgt.device)).bool()
        return pm & cm

    def forward(self, src, tgt):
        sm, tm = self.make_src_mask(src), self.make_tgt_mask(tgt)
        enc = self.encoder(src, sm)
        dec = self.decoder(tgt, enc, sm, tm)
        return self.proj(dec)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, vsz, pad_idx, smoothing=0.05):
        super().__init__()
        self.kl = nn.KLDivLoss(reduction='sum')
        self.pad, self.sm, self.vsz = pad_idx, smoothing, vsz
        self.conf = 1.0 - smoothing

    def forward(self, logits, target):
        lp = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            d = torch.full_like(lp, self.sm / (self.vsz - 2))
            d.scatter_(1, target.unsqueeze(1), self.conf)
            d[:, self.pad] = 0
            mask = (target == self.pad)
            d[mask] = 0
        loss = self.kl(lp, d)
        n = (~mask).sum().item()
        return loss / n if n > 0 else loss


class WarmupScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=100):
        self.opt, self.d, self.ws, self.step_ = optimizer, d_model, warmup_steps, 0

    def step(self):
        self.step_ += 1
        lr = (self.d ** -0.5) * min(self.step_ ** -0.5, self.step_ * (self.ws ** -1.5))
        for pg in self.opt.param_groups: pg['lr'] = lr
        return lr