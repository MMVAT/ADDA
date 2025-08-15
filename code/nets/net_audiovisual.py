import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from module.adha import ADHA
from module.diff_attn.multihead_diffattn import MultiheadDiffAttn

class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, d_model=512, nhead=1, dim_feedforward=512, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = []
        if encoder_layer == 'HANLayer':
            for i in range(num_layers):
                self.layers.append(HANLayer(d_model=d_model, nhead=nhead,
                                            dim_feedforward=dim_feedforward, dropout=dropout))
        else:
            raise ValueError('wrong encoder layer')
        self.layers = nn.ModuleList(self.layers)
        self.num_layers = num_layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm = norm

    def forward(self, src_a, src_v, mask=None, src_key_padding_mask=None, with_ca=True):
        output_a = src_a
        output_v = src_v

        for i in range(self.num_layers):
            output_a = self.layers[i](src_a, src_v, src_mask=mask,
                                      src_key_padding_mask=src_key_padding_mask, with_ca=with_ca)
            output_v = self.layers[i](src_v, src_a, src_mask=mask,
                                      src_key_padding_mask=src_key_padding_mask, with_ca=with_ca)
            src_a = output_a
            src_v = output_v

        if self.norm:
            output_a = self.norm1(output_a)
            output_v = self.norm2(output_v)

        return output_a, output_v


class HANLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(HANLayer, self).__init__()
        self.head_dim = d_model // (nhead * 2)
        self.nhead2 = 4

        self.self_attn = MultiheadDiffAttn(
            decoder_kv_attention_heads=8,
            embed_dim=d_model,
            depth=32,
            num_heads=nhead,
        )
        self.cm_attn = nn.MultiheadAttention(d_model, self.nhead2, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def get_rel_pos(self, seq_len, device):
        """计算RoPE位置编码，确保长度足够"""
        # 使用4倍序列长度来确保足够长
        max_seq_len = seq_len * 64  # 从2倍改为4倍

        position = torch.arange(max_seq_len, device=device).float()
        freqs = torch.exp(
            -torch.arange(0, self.head_dim, 2, device=device).float()
            * math.log(10000.0) / self.head_dim
        )

        emb = position[:, None] * freqs[None, :]
        cos_cache = torch.cos(emb)
        sin_cache = torch.sin(emb)

        # 确保返回的长度至少等于seq_len
        if cos_cache.size(0) < max_seq_len:
            # 如果长度仍然不够，进一步扩展
            pad_len = seq_len - cos_cache.size(0)
            cos_cache = F.pad(cos_cache, (0, 0, 0, pad_len), mode='replicate')
            sin_cache = F.pad(sin_cache, (0, 0, 0, pad_len), mode='replicate')

        return (cos_cache[:128], sin_cache[:128])

    def forward(self, src_q, src_v, src_mask=None, src_key_padding_mask=None, with_ca=True):
        """Pass the input through the encoder layer.

        Args:
            src_q: the sequence to the encoder layer (required).
            src_v: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            with_ca: whether to use audio-visual cross-attention
        Shape:
            see the docs in Transformer class.
        """
        bsz, seq_len, _ = src_q.size()
        device = src_q.device
        rel_pos = self.get_rel_pos(seq_len, device)  # 2* 512 128

        src_q = src_q.permute(1, 0, 2)
        src_v = src_v.permute(1, 0, 2)

        if with_ca:
            src1 = self.cm_attn(src_q, src_v, src_v, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)[0]
            src2 = self.self_attn(src_q, rel_pos, attn_mask=src_mask)
            src2 = src2.permute(1, 0, 2)
            src_q = src_q + self.dropout11(src1) + self.dropout12(src2)
            src_q = self.norm1(src_q)
        else:
            src2 = self.self_attn(src_q, rel_pos, attn_mask=src_mask)

            src_q = src_q + self.dropout12(src2)
            src_q = self.norm1(src_q)

        src_q = src_q + self.dropout2(self.linear2(self.dropout(F.relu(self.linear1(src_q)))))
        src_q = self.norm2(src_q)
        return src_q.permute(1, 0, 2)

class MMIL_Net(nn.Module):

    def __init__(self, num_layers=1, att_dropout=0.1, cls_dropout=0.5):
        super(MMIL_Net, self).__init__()

        self.fc_prob = nn.Linear(512, 25)

        self.fc_frame_att = nn.Linear(512, 25)
        self.fc_av_att = nn.Linear(512, 25)
        self.fc_a = nn.Linear(512, 512)
        self.fc_v = nn.Linear(512, 512)
        self.hat_encoder = Encoder('HANLayer', num_layers, norm=None, d_model=512,
                                   nhead=8, dim_feedforward=512, dropout=att_dropout)
        if cls_dropout != 0:
            self.dropout = nn.Dropout(p=cls_dropout)
        else:
            self.dropout = None

        self.aga = ADHA().cuda()

    def forward(self, audio, visual, with_ca=True):
        x1 = self.fc_a(audio)
        b,t,c = audio.shape

        # clip feature
        vids = visual.reshape(b,t,8,1,c)
        vid_s1 = self.fc_v(vids.float())
        x2 = self.aga(vid_s1, audio)

        x1, x2 = self.hat_encoder(x1, x2, with_ca=True)

        if self.dropout is not None:
            x2 = self.dropout(x2)
        x = torch.cat([x1.unsqueeze(-2), x2.unsqueeze(-2)], dim=-2)
        frame_prob = torch.sigmoid(self.fc_prob(x))
        # attentive MMIL pooling
        frame_att = torch.softmax(self.fc_frame_att(x), dim=1)
        av_att = torch.softmax(self.fc_av_att(x), dim=2)
        temporal_prob = frame_att * frame_prob
        global_prob = (temporal_prob * av_att).sum(dim=2).sum(dim=1)
        # frame-wise probability
        a_prob = temporal_prob[:, :, 0, :].sum(dim=1)
        v_prob = temporal_prob[:, :, 1, :].sum(dim=1)   
        return global_prob, a_prob, v_prob, frame_prob

        
