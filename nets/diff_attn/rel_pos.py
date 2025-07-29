import torch
import torch.nn as nn
import math


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super(RotaryPositionEmbedding, self).__init__()
        self.dim = dim
        self.max_len = max_len
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))  # 频率的倒数

        # 生成一个位置索引张量
        self.position = torch.arange(0, max_len).float().unsqueeze(1)  # (max_len, 1)
        self.freq = self.position * self.inv_freq  # 位置索引和频率的乘积

    def forward(self, seq_len: int):
        # 获取要处理的序列长度内的频率
        sin_cos_freq = torch.matmul(self.position[:seq_len], self.freq.unsqueeze(0))  # (seq_len, dim//2)

        # 使用sin和cos对频率编码
        sin_cos_freq = sin_cos_freq.unsqueeze(0).expand(1, -1, -1)  # (1, seq_len, dim//2)
        sin_cos_freq = torch.cat((sin_cos_freq.sin(), sin_cos_freq.cos()), dim=-1)  # (1, seq_len, dim)

        return sin_cos_freq


len = 10
dim = 128
max_len = 1
ro = RotaryPositionEmbedding(dim, max_len)
sc = ro(len)
print(sc)