import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from .diff_attn.multihead_diffattn import MultiheadDiffAttn



class New_Audio_Guided_Attention(nn.Module):
    def __init__(self):
        super(New_Audio_Guided_Attention, self).__init__()
        self.hidden_size = 512
        self.relu = nn.ReLU()
        # channel attention
        self.affine_video_1 = nn.Linear(512, 512)
        self.affine_audio_1 = nn.Linear(128, 512)
        self.affine_bottleneck = nn.Linear(512, 256)
        self.affine_v_c_att = nn.Linear(256, 512)
        # spatial attention
        self.affine_video_2 = nn.Linear(512, 256)
        self.affine_audio_2 = nn.Linear(128, 256)
        self.affine_v_s_att = nn.Linear(256, 1)

        # video-guided audio attention
        self.affine_video_guided_1 = nn.Linear(512, 64)
        self.affine_video_guided_2 = nn.Linear(64, 128)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, video, audio):
        '''
        :param visual_feature: [batch, 10, 7, 7, 512]
        :param audio_feature:  [batch, 10, 128]
        :return: [batch, 10, 512]
        '''
        audio = audio.transpose(1, 0)
        batch, t_size, h, w, v_dim = video.size()
        a_dim = audio.size(-1)
        audio_feature = audio.reshape(batch * t_size, a_dim)
        visual_feature = video.reshape(batch, t_size, -1, v_dim)
        raw_visual_feature = visual_feature

        # ============================== Channel Attention ====================================
        audio_query_1 = self.relu(self.affine_audio_1(audio_feature)).unsqueeze(-2)
        video_query_1 = self.relu(self.affine_video_1(visual_feature)).reshape(batch * t_size, h * w, -1)
        audio_video_query_raw = (audio_query_1 * video_query_1).mean(-2)
        audio_video_query = self.relu(self.affine_bottleneck(audio_video_query_raw))
        channel_att_maps = self.affine_v_c_att(audio_video_query).sigmoid().reshape(batch, t_size, -1, v_dim)
        c_att_visual_feat = (raw_visual_feature * (channel_att_maps + 1))

        # ============================== Spatial Attention =====================================
        c_att_visual_feat = c_att_visual_feat.reshape(batch * t_size, -1, v_dim)
        c_att_visual_query = self.relu(self.affine_video_2(c_att_visual_feat))
        audio_query_2 = self.relu(self.affine_audio_2(audio_feature)).unsqueeze(-2)
        audio_video_query_2 = c_att_visual_query * audio_query_2
        spatial_att_maps = self.softmax(self.tanh(self.affine_v_s_att(audio_video_query_2)).transpose(2, 1))
        c_s_att_visual_feat = torch.bmm(spatial_att_maps, c_att_visual_feat).squeeze().reshape(batch, t_size, v_dim)

        return c_s_att_visual_feat

def exp_evidence(y, temp=0.8, v_min=5e-6):
    y = torch.clamp(y, -10, 10)
    evidence = torch.exp(y / temp)
    evidence = evidence + v_min
    return evidence

def compute_dirichlet_params(logits):
    evidence = exp_evidence(logits)
    alpha = evidence + 1.0
    return alpha

def get_p_and_u_from_logit(x):
    alpha = compute_dirichlet_params(x)
    S = torch.sum(alpha, dim=-1, keepdim=True)
    prob = alpha[..., 0] / S.squeeze(-1)
    K = alpha.size(-1)
    total_evidence = torch.sum(alpha - 1, dim=-1)
    uncertainty = K / (K + total_evidence)

    return prob, uncertainty

class LabelSmoothingNCELoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingNCELoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return -torch.mean(torch.log(torch.sum(true_dist * pred, dim=self.dim)))

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
        # 替换原有的self_attn和cm_attn
        self.d_model = d_model
        self.nhead = nhead
        self.nhead2 = 4
        self.head_dim = d_model // (nhead * 2)

        self.self_attn = MultiheadDiffAttn(
            decoder_kv_attention_heads=8,
            embed_dim=d_model,
            depth=24,
            num_heads=nhead,
        )

        # self.cm_attn = MultiheadDiffAttncs(
        #     decoder_kv_attention_heads=2,
        #     embed_dim=d_model,
        #     depth=24,
        #     num_heads=self.nhead2,
        # )

        # 保持原有的cm_attn不变
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
        # 准备位置编码
        bsz, seq_len, _ = src_q.size()
        device = src_q.device

        rel_pos = self.get_rel_pos(seq_len, device)  # 2* 512 128

        src_q = src_q.permute(1, 0, 2)
        src_v = src_v.permute(1, 0, 2)

        if with_ca:
            # src1 = self.cm_attn(src_q, src_v, rel_pos, attn_mask=src_mask)
            src1 = self.cm_attn(src_q, src_v, src_v, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
            
            src2 = self.self_attn(src_q, rel_pos, attn_mask=src_mask)
            src2 = src2.permute(1, 0, 2)
            # src1 = src1.permute(1, 0, 2)
            src_q = src_q + self.dropout11(src1) + self.dropout12(src2)
            src_q = self.norm1(src_q)
        else:
            # 仅使用自注意力
            src2 = self.self_attn(src_q, rel_pos, attn_mask=src_mask)
            src2 = src2.permute(1, 0, 2)
            src_q = src_q + self.dropout12(src2)
            src_q = self.norm1(src_q)

        # 前馈网络部分保持不变
        src_q = src_q + self.dropout2(self.linear2(self.dropout(F.relu(self.linear1(src_q)))))
        src_q = self.norm2(src_q)

        return src_q.permute(1, 0, 2)


class MMIL_Net(nn.Module):
    def __init__(self, num_layers=1, temperature=0.2, att_dropout=0.1, cls_dropout=0.5, nhead=1):
        super(MMIL_Net, self).__init__()

        self.fc_prob = nn.Linear(512, 25)
        self.fc_prob_beta = nn.Linear(512, 25)
        self.fc_frame_att = nn.Linear(512, 25 * 2)
        self.fc_global = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 25),
            nn.Sigmoid(),
        )
        self.fc_a = nn.Sequential(
            nn.Linear(128, 512),
            nn.ELU(),
        )
        self.fc_v = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ELU(),
        )
        self.fc_st = nn.Sequential(
            nn.Linear(512, 512),
            nn.ELU(),
        )
        self.fc_fusion = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ELU(),
        )

        self.norm = nn.LayerNorm(512)
        self.hat_encoder = Encoder('HANLayer', num_layers, norm=None, d_model=512,
                                   nhead=nhead, dim_feedforward=512, dropout=att_dropout)

        self.v2a = nn.Sequential(
            nn.Linear(512, 512),
            # nn.Dropout(0.2),
            nn.LeakyReLU(0.2),
        )

        self.a2v = nn.Sequential(
            nn.Linear(512, 512),
            # nn.Dropout(0.2),
            nn.LeakyReLU(0.2),
        )

        self.classifier1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
        )

        self.classifier2 = nn.Linear(512, 25)

        self.temp = temperature
        if cls_dropout != 0:
            self.dropout = nn.Dropout(p=cls_dropout)
        else:
            self.dropout = None

        # self.spatial_channel_att = Visual_Attention().cuda()
        # self.avf_att = Audio_Visual_Fusion_Attention().cuda()
        self.aga = New_Audio_Guided_Attention().cuda()
        # self.aga = New_Audio_Guided_Attentions(beta=0.5).cuda()

    def forward(self, audio, visual, visual_st, with_ca=True):
        b, t, d = visual_st.size()
        x1 = self.fc_a(audio)

        # 2d and 3d visual feature fusion
        vid_s = visual.reshape(b, t, 8, 1, 2048)
        vid_s1 = self.fc_v(vid_s)
        vid_s = self.aga(vid_s1, audio)
        vid_st = self.fc_st(visual_st)
        x2 = torch.cat((vid_s, vid_st), dim=-1)
        x2 = self.fc_fusion(x2)

        # Mutual Learning
        x1_embed = self.classifier1(x1)
        x2_embed = self.classifier1(x2)
        ori_logit = self.classifier2(x1_embed + x2_embed)
        ori_alpha = exp_evidence(ori_logit.mean(dim=1)).add(1.0)
        global_uct = 25 / torch.sum(ori_alpha, dim=-1)
        ori_prob = nn.Sigmoid()(ori_logit)
        global_prob = ori_prob.mean(dim=1)

        # HAN
        # x1, x2 = self.hat_encoder(x1, x2, with_ca=with_ca)
        if not with_ca:
            x1, x2 = self.hat_encoder(x1, x2, with_ca=False)
        else:
            x1_woca, x2_woca = self.hat_encoder(x1, x2, with_ca=False)
            x1_ca, x2_ca = self.hat_encoder(x1, x2, with_ca=True)
            ratio = 0.8
            x1 = x1_ca * ratio + x1_woca * (1 - ratio)
            x2 = x2_ca * ratio + x2_woca * (1 - ratio)

        # noise contrastive
        # please refer to https://github.com/Yu-Wu/Modaily-Aware-Audio-Visual-Video-Parsing
        xx2_after = F.normalize(x2, p=2, dim=-1)
        xx1_after = F.normalize(x1, p=2, dim=-1)
        sims_after = xx1_after.bmm(xx2_after.permute(0, 2, 1)).squeeze(1) / self.temp
        sims_after = sims_after.reshape(-1, 10)
        mask_after = torch.zeros(b, 10)
        mask_after = mask_after.long()
        for i in range(10):
            mask_after[:, i] = i
        mask_after = mask_after.cuda()
        mask_after = mask_after.reshape(-1)

        # prediction
        if self.dropout is not None:
            x2 = self.dropout(x2)
        x = torch.cat([x1.unsqueeze(-2), x2.unsqueeze(-2)], dim=-2)

        c = 25
        b, t, m, _ = x.shape  # 128, 10, 2, 512

        frame_alpha_logit = self.fc_prob(x)  # 128, 10, 2, 25
        x_a = x[:, :, 0]  # 128, 10, 512
        x_v = x[:, :, 1]  # 128, 10, 512
        x_v2a = x_a + self.v2a(x_a + x_v)
        x_a2v = x_v + self.a2v(x_a + x_v)
        frame_beta_a = self.fc_prob_beta(x_v2a)  # 128, 10, 25
        frame_beta_v = self.fc_prob_beta(x_a2v)
        frame_beta_logit = torch.stack((frame_beta_a, frame_beta_v), dim=2)
        frame_logit = torch.stack((frame_alpha_logit, frame_beta_logit), dim=-1)

        # attentive MMIL pooling
        frame_att_before_softmax = self.fc_frame_att(x).reshape(b, t, m, c, 2)
        frame_att = torch.softmax(frame_att_before_softmax, dim=1)
        temporal_logit = frame_att * frame_logit
        # frame-wise probability
        a_logit = temporal_logit[:, :, 0, :, :].sum(dim=1)
        v_logit = temporal_logit[:, :, 1, :, :].sum(dim=1)

        a_prob, a_uct = get_p_and_u_from_logit(a_logit)
        v_prob, v_uct = get_p_and_u_from_logit(v_logit)
        frame_prob, frame_uct = get_p_and_u_from_logit(frame_logit)

        return global_prob, a_prob, v_prob, frame_prob, sims_after, mask_after, global_uct, a_uct, v_uct, frame_uct