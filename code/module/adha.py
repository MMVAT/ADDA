import torch
import torch.nn as nn


class ADHA(nn.Module):
    def __init__(self):
        super(ADHA, self).__init__()
        self.hidden_size = 512
        self.relu = nn.ReLU()
        self.affine_video_1 = nn.Linear(512, 512)
        self.affine_audio_1 = nn.Linear(512, 512)
        self.affine_bottleneck = nn.Linear(512, 256)
        self.affine_v_c_att = nn.Linear(256, 512)
        self.affine_video_2 = nn.Linear(512, 256)
        self.affine_audio_2 = nn.Linear(512, 256)
        self.affine_v_s_att = nn.Linear(256, 1)

        self.affine_video_guided_1 = nn.Linear(512, 64)
        self.affine_video_guided_2 = nn.Linear(64, 128)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, video, audio):
        audio = audio.transpose(1, 0)
        batch, t_size, h, w, v_dim = video.size()
        a_dim = audio.size(-1)
        audio_feature = audio.reshape(batch * t_size, a_dim)
        visual_feature = video.reshape(batch, t_size, -1, v_dim)
        raw_visual_feature = visual_feature
        # ============================== Audio-Guided Channel Attention ====================================
        audio_query_1 = self.relu(self.affine_audio_1(audio_feature)).unsqueeze(
            -2)
        video_query_1 = self.relu(self.affine_video_1(visual_feature)).reshape(batch * t_size, h * w,
                                                                               -1)
        audio_video_query_raw = (audio_query_1 * video_query_1).mean(-2)
        audio_video_query = self.relu(self.affine_bottleneck(audio_video_query_raw))
        channel_att_maps = self.affine_v_c_att(audio_video_query).sigmoid().reshape(batch, t_size, -1,
                                                                                    v_dim)
        c_att_visual_feat = (
                    raw_visual_feature * (channel_att_maps + 1))

        # ============================== Event-Aware Spatial Localization =====================================
        c_att_visual_feat = c_att_visual_feat.reshape(batch * t_size, -1, v_dim)
        c_att_visual_query = self.relu(self.affine_video_2(c_att_visual_feat))
        audio_query_2 = self.relu(self.affine_audio_2(audio_feature)).unsqueeze(
            -2)
        audio_video_query_2 = c_att_visual_query * audio_query_2
        spatial_att_maps = self.softmax(self.tanh(self.affine_v_s_att(audio_video_query_2)).transpose(2, 1))
        c_s_att_visual_feat_bmm = torch.bmm(spatial_att_maps,
                                            c_att_visual_feat)
        c_s_att_visual_feat = c_s_att_visual_feat_bmm.squeeze().reshape(batch, t_size, v_dim)

        return c_s_att_visual_feat

