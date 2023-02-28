import torch
import torch.nn as nn
from layers.ConvLSTM import ConvLSTM2D

class Encoder_wrf_model(nn.Module):
    def __init__(self, tra_frames, channels, config_dict):
        super(Encoder_wrf_model, self).__init__()
        self.config_dict = config_dict
        self.tra_frames = tra_frames
        self.wrf_encoder_conv2d = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.wrf_encoder_convLSTM2D = ConvLSTM2D(64, 32, kernel_size=5, img_rowcol=(config_dict['GridRowColNum']//2)//2)

    def forward(self, wrf):
        # wrf : [frames, batch_size, channels, x, y]
        batch_size = wrf.shape[1]
        wrf_conv = [None] * self.tra_frames
        for i in range(self.tra_frames):
            wrf_conv[i] = self.wrf_encoder_conv2d(wrf[i])
        wrf_h = torch.zeros([batch_size, 32, wrf_conv[0].shape[2], wrf_conv[0].shape[3]], dtype=torch.float32).to(wrf.device)
        wrf_c = torch.zeros([batch_size, 32, wrf_conv[0].shape[2], wrf_conv[0].shape[3]], dtype=torch.float32).to(wrf.device)

        for i in range(self.tra_frames):
            wrf_h, wrf_c = self.wrf_encoder_convLSTM2D(wrf_conv[i], wrf_h, wrf_c)

        return wrf_h, wrf_c


class Encoder_obs_model(nn.Module):
    def __init__(self, tra_frames, channels, config_dict):
        super(Encoder_obs_model, self).__init__()
        self.config_dict = config_dict
        self.tra_frames = tra_frames
        self.obs_encoder_conv2d = nn.Sequential(
                nn.Conv2d(channels, 4, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(4, 4, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        self.obs_encoder_convLSTM2D = ConvLSTM2D(4, 8, kernel_size=5, img_rowcol=(config_dict['GridRowColNum']//2)//2)

    def forward(self, obs):
        # obs : [frames, batch_size, channels, x, y]
        batch_size = obs.shape[1]
        obs_conv = [None] * self.tra_frames
        for i in range(self.tra_frames):
            obs_conv[i] = self.obs_encoder_conv2d(obs[i])
        obs_h = torch.zeros([batch_size, 8, obs_conv[0].shape[2], obs_conv[0].shape[3]], dtype=torch.float32).to(obs.device)
        obs_c = torch.zeros([batch_size, 8, obs_conv[0].shape[2], obs_conv[0].shape[3]], dtype=torch.float32).to(obs.device)

        for i in range(self.tra_frames):
            obs_h, obs_c = self.obs_encoder_convLSTM2D(obs_conv[i], obs_h, obs_c)

        return obs_h, obs_c


class Fusion_model(nn.Module):
    def __init__(self, filters_list_input, filters_list_output, config_dict):
        super(Fusion_model, self).__init__()
        self.config_dict = config_dict
        self.filters_list_output = filters_list_output
        fusion_conv_h = [None] * len(self.filters_list_output)
        fusion_conv_c = [None] * len(self.filters_list_output)
        # fusion_conv_encoder = [None] * len(self.filters_list)
        for i in range(len(self.filters_list_output)):
            fusion_conv_h[i] = nn.Sequential(
                nn.Conv2d(filters_list_input[i], self.filters_list_output[i], kernel_size=1, stride=1),
                nn.ReLU()
            )
            fusion_conv_c[i] = nn.Sequential(
                nn.Conv2d(filters_list_input[i], self.filters_list_output[i], kernel_size=1, stride=1),
                nn.ReLU()
            )
        self.fusion_conv_h = nn.ModuleList(fusion_conv_h)
        self.fusion_conv_c = nn.ModuleList(fusion_conv_c)

    def forward(self, h_list, c_list):
        h_concat = [None] * len(self.filters_list_output)
        c_concat = [None] * len(self.filters_list_output)
        encoder_concat = [None] * len(self.filters_list_output)
        for i in range(len(self.filters_list_output)):
            h_concat[i] = self.fusion_conv_h[i](h_list[i])
            c_concat[i] = self.fusion_conv_c[i](c_list[i])
            # encoder_concat[i] = self.fusion_conv_encoder[i](encoder_output_list[i])
        h_concat = torch.cat(h_concat, dim=1)
        c_concat = torch.cat(c_concat, dim=1)
        # encoder_concat = torch.cat(encoder_concat, dim=1)
        # fusion_output = torch.cat([h_concat, c_concat, encoder_concat], dim=1)
        return h_concat, c_concat


class Decoder_model(nn.Module):
    def __init__(self, pre_frames, ConvLSTM2D_filters, config_dict):
        super(Decoder_model, self).__init__()
        self.config_dict = config_dict
        self.pre_frames = pre_frames
        self.decoder_conv2D = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.decoder_convLSTM2D = ConvLSTM2D(4, ConvLSTM2D_filters, kernel_size=5, img_rowcol=(config_dict['GridRowColNum']//2)//2)
        self.decoder_transconv2D = nn.Sequential(
            nn.ConvTranspose2d(ConvLSTM2D_filters, ConvLSTM2D_filters, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(ConvLSTM2D_filters, ConvLSTM2D_filters, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(ConvLSTM2D_filters, 1, kernel_size=1, stride=1)
        )

    def forward(self, frame, h, c):
        pre_frames = [None] * self.pre_frames
        for i in range(self.pre_frames):
            frame = self.decoder_conv2D(frame)
            h, c = self.decoder_convLSTM2D(frame, h, c)
            pre_frames[i] = self.decoder_transconv2D(h)
            frame = torch.sigmoid(pre_frames[i])

        pre_frames = torch.stack(pre_frames, dim=0)
        return pre_frames


class LightNet_Model(nn.Module):
    def __init__(self, obs_tra_frames, obs_channels, wrf_tra_frames, wrf_channels, config_dict):
        super(LightNet_Model, self).__init__()
        self.config_dict = config_dict
        pre_frames = wrf_tra_frames
        self.encoder_obs_model = Encoder_obs_model(obs_tra_frames, obs_channels, config_dict)
        self.encoder_wrf_model = Encoder_wrf_model(wrf_tra_frames, wrf_channels, config_dict)
        filters_list_output = [16, 8]
        self.fusion_model = Fusion_model([8, 32], filters_list_output, config_dict)   # [obs, wrf]
        self.decoder_model = Decoder_model(pre_frames, sum(filters_list_output), config_dict)

    def forward(self, wrf, obs):
        # encoder
        h_list = []
        c_list = []
        # obs : [batch_size, frames, x, y, channels] -> [frames, batch_size, channels, x, y]
        obs = obs.permute(1, 0, 4, 2, 3).contiguous()
        obs_h, obs_c = self.encoder_obs_model(obs)
        h_list.append(obs_h)
        c_list.append(obs_c)

        # wrf : [batch_size, frames, x, y, channels] -> [frames, batch_size, channels, x, y]
        wrf = wrf.permute(1, 0, 4, 2, 3).contiguous()
        wrf_h, wrf_c = self.encoder_wrf_model(wrf)
        h_list.append(wrf_h)
        c_list.append(wrf_c)

        # fusion
        fusion_h, fusion_c = self.fusion_model(h_list, c_list)

        # decoder
        pre_frames = self.decoder_model(obs[-1], fusion_h, fusion_c)

        # pre_frames = pre_frames.squeeze(dim=2)
        # pre_frames = pre_frames.reshape([pre_frames.shape[0], pre_frames.shape[1], pre_frames.shape[2]*pre_frames.shape[3], -1])
        pre_frames = pre_frames.permute(1, 0, 3, 4, 2).contiguous()
        # pre_frames = torch.sigmoid(pre_frames)
        return pre_frames
