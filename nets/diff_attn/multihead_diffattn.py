import math
import torch
import torch.nn.functional as F
from torch import nn
from .rotary import apply_rotary_emb
from .rms_norm import RMSNorm


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )


def lambda_init_fn(depth):
    return 1.0 - 0.65 * math.exp(-0.35 * depth)


class MultiheadDiffAttn(nn.Module):
    def __init__(
            self,
            decoder_kv_attention_heads,
            embed_dim,
            depth,
            num_heads,
    ):
        super().__init__()
        self.embed_dim = embed_dim  # 512
        self.num_heads = num_heads  # 1
        self.num_kv_heads = decoder_kv_attention_heads if decoder_kv_attention_heads is not None else num_heads  # 1
        self.n_rep = self.num_heads // self.num_kv_heads  # 1

        self.head_dim = embed_dim // num_heads // 2  # 256
        self.scaling = self.head_dim ** -0.5  # 0.0625

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, x, rel_pos, attn_mask=None):
        tgt_len, bsz, embed_dim = x.size() 
        src_len = tgt_len  

        q = self.q_proj(x)  
        k = self.k_proj(x)
        v = self.v_proj(x)
        # print(q.shape)

        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)  
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim) 

        q = apply_rotary_emb(q, *rel_pos, interleaved=True)
        k = apply_rotary_emb(k, *rel_pos, interleaved=True)

        offset = src_len - tgt_len  # 0
        q = q.transpose(1, 2)
        k = repeat_kv(k.transpose(1, 2), self.n_rep)  # 128 2 10 256
        v = repeat_kv(v.transpose(1, 2), self.n_rep)  # 128 1 10 256
        q *= self.scaling  # 128 2 10 256
        attn_weights = torch.matmul(q, k.transpose(-1, -2))  # 128 2 10 10
        if attn_mask is None:
            attn_mask = torch.triu(  # 10 10
                torch.zeros([tgt_len, src_len])
                .float()
                .fill_(float("-inf"))
                .type_as(attn_weights),
                1 + offset,
            )
        attn_weights = torch.nan_to_num(attn_weights)  # 128 2 10 10
        attn_weights += attn_mask  # 128 2 10 10
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(  # 128 2 10 10
            attn_weights
        )

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)  # 0.9462
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)  # 0.9402
        lambda_full = lambda_1 - lambda_2 + self.lambda_init  # 0.2419
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)  # 128 1 2 10 10
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]  # 128 1 10 10

        attn = torch.matmul(attn_weights, v)  # 128 1 10 512
        attn = self.subln(attn)  # RMSnorm  128 1 10 512
        attn = attn * (1 - self.lambda_init)  # 128 1 10 512
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)  # 128 10 512

        attn = self.out_proj(attn)  # 128 10 512
        return attn

