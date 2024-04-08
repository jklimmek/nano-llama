import einops
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class SwiGLU(nn.Module):
     
    def __init__(self, dim, ffwd_dropout):
        super().__init__()
        # Keep everything in multiples of 8 for potential efficiency.
        dim_proj = int(2/3 * 4 * dim // 8 * 8)
        self.W = nn.Linear(dim, dim_proj, bias=False)
        self.V = nn.Linear(dim, dim_proj, bias=False)
        self.W2 = nn.Linear(dim_proj, dim, bias=False)
        self.dropout = nn.Dropout(ffwd_dropout)
        
    def forward(self, x):
        xW = self.W(x)
        xV = self.V(x)
        WV = F.silu(xW) * xV
        WVW = self.W2(WV)
        out = self.dropout(WVW)
        return out


class RMSNorm(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))
        self.epsilon = 1e-5

    def forward(self, x):
        mean_square = (x ** 2).mean(dim=-1, keepdim=True).sqrt()
        x_hat = x / (mean_square + self.epsilon) * self.g
        return x_hat


class AlibiMultiHeadAttention(nn.Module):

    def __init__(self, context_size, dim, num_heads, attn_dropout, resid_dropout):
        super().__init__()
        self.w_qkv = nn.Linear(dim, 3*dim, bias=False)
        self.w_o = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(resid_dropout)
        self.num_heads = num_heads
        self.register_buffer("mask", torch.tril(torch.ones(1, 1, context_size, context_size)))
        self.register_buffer("m", self.get_alibi_slope(self.num_heads))

    def forward(self, x):
        _, T, C = x.shape
        q, k, v = self.w_qkv(x).chunk(3, dim=-1)

        q = einops.rearrange(q, "b n (h d_h) -> b h n d_h", h=self.num_heads)
        k = einops.rearrange(k, "b n (h d_h) -> b h n d_h", h=self.num_heads)
        v = einops.rearrange(v, "b n (h d_h) -> b h n d_h", h=self.num_heads)
        
        wei = einops.einsum(q, k, "b h n_q d_h, b h n_k d_h -> b h n_q n_k")
        alibi_bias = (self.m * self.get_relative_positions(T).to(x.device))
        scale = C ** 0.5
        scaled_wei = wei / scale + alibi_bias

        scores = scaled_wei.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = einops.einsum(attn, v, "b h n_q n_k, b h n_k d_h -> b h n_q d_h")
        concat = einops.rearrange(out, "b h n_q d_h -> b n_q (h d_h)")
        o_proj = self.w_o(concat)
        o_proj = self.resid_dropout(o_proj)
        return o_proj
    
    @staticmethod
    def get_relative_positions(seq_len):
        x = torch.arange(seq_len)[None, :]
        y = torch.arange(seq_len)[:, None]
        return x - y

    @staticmethod
    def get_alibi_slope(num_heads):
        x = (2 ** 8) ** (1 / num_heads)
        return (
            torch.tensor([1 / x ** (i + 1) for i in range(num_heads)])
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
    

class Block(nn.Module):
    
    def __init__(
            self, 
            context_size,
            dim, 
            num_heads, 
            attn_dropout,
            ffwd_dropout,
            resid_dropout
        ):
        super().__init__()
        self.rms_norm1 = RMSNorm(dim)
        self.self_attn = AlibiMultiHeadAttention(context_size, dim, num_heads, attn_dropout, resid_dropout)
        self.rms_norm2 = RMSNorm(dim)
        self.swiglu = SwiGLU(dim, ffwd_dropout)

    def forward(self, x):
        x_resid = x
        x = self.rms_norm1(x)
        x = self.self_attn(x)
        x = x + x_resid
        x_resid = x
        x = self.rms_norm2(x)
        x = self.swiglu(x)
        x = x + x_resid
        return x
    

class NanoLlama(nn.Module):
    """Llama 2 architecture, with replaced RoPE for ALiBi."""

    def __init__(
            self, 
            vocab_size, 
            context_size, 
            dim, 
            num_heads, 
            num_layers, 
            attn_dropout=0.0, 
            ffwd_dropout=0.0, 
            resid_dropout=0.0,
            init_mode="gpt-2"
        ):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, dim)
        self.core = nn.Sequential(
            *[Block(context_size, dim, num_heads, attn_dropout, ffwd_dropout, resid_dropout) for _ in range(num_layers)]
        )
        self.rms_norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.embeddings.weight = self.head.weight
        self._init_weights(init_mode)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.core(x)
        x = self.rms_norm(x)
        x = self.head(x)
        return x
    
    def configure_optimizer(self, weight_decay, lr, betas):
        """Configure AdamW optimizer."""
        # Weight decay on embedding layer improves model perplexity wtf.
        exclude_params = [] #"embeddings"
        params = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        decay_params = [p for pn, p in params.items() if p.dim() >= 2 and not any(param in pn for param in exclude_params)]
        # Excludes normalization and biases if they exist.
        no_decay_params = [p for _, p in params.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = optim.AdamW(optim_groups, lr=lr, betas=betas, eps=1e-5)
        return optimizer

    def _init_weights(self, init_mode):
        """Weight initialization inspired by different schemas."""
        num_layers = len(self.core)
        vocab, dim = self.embeddings.weight.shape
        swiglu_dim = self.core[0].swiglu.W.weight.shape[0]

        if init_mode == "gpt-2":
        # https://github.com/karpathy/nanoGPT/blob/master/model.py
            for pn, p in self.named_parameters():
                if any(n in pn for n in ["swiglu", "w_qkv", "embeddings"]):
                    nn.init.normal_(p, mean=0.0, std=0.02)
                elif pn.endswith('w_o.weight'):
                    nn.init.normal_(p, mean=0.0, std=0.02 / (2 * num_layers) ** 0.5)

        elif init_mode == "olmo":
        # https://github.com/allenai/OLMo/blob/main/olmo/initialization.py
            for pn, p in self.named_parameters():
                if "core" in pn:
                    std = 1.0 / dim ** 0.5
                    idx = int(pn.split(".")[1])
                    std = std / (2 * (idx + 1)) ** 0.5
                    if any(n in pn for n in ["swiglu", "w_qkv"]):
                        nn.init.trunc_normal_(p, mean=0.0, std=std, a=-3*std, b=3*std)
                    if pn.endswith('w_o.weight'):
                        std = std / (2 * num_layers) ** 0.5
                        nn.init.trunc_normal_(p, mean=0.0, std=std, a=-3*std, b=3*std)
                elif pn.startswith("embeddings"):
                    std = (2 / (vocab + dim)) ** 0.5
                    nn.init.trunc_normal_(p, mean=0.0, std=std, a=-3*std, b=3*std)

        elif init_mode == "small":
        # https://github.com/tnq177/transformers_without_tears/blob/master/layers.py
            for pn, p in self.named_parameters():
                if "self_attn" in pn:
                    std = 2 / (5 * dim) ** 0.5
                    nn.init.normal_(p, mean=0.0, std=std)
                elif "swiglu" in pn:
                    std = (2 / (swiglu_dim + dim)) ** 0.5
                    nn.init.normal_(p, mean=0.0, std=std)
                elif pn.startswith("embeddings"):
                    std = (2 / (vocab + dim)) ** 0.5
                    nn.init.normal_(p, mean=0.0, std=std)