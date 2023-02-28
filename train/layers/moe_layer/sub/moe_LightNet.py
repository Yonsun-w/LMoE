import torch
import torch.nn as nn
from layers.ConvLSTM import ConvLSTM2D

class Encoder_wrf_model(nn.Module):
    def __init__(self, tra_frames, channels, row_col):
        super(Encoder_wrf_model, self).__init__()
        self.tra_frames = tra_frames
        self.wrf_encoder_conv2d = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.wrf_encoder_convLSTM2D = ConvLSTM2D(64, 32, kernel_size=5, img_rowcol=(row_col//2)//2)

    def forward(self, wrf):
        # wrf : [frames, batch_size, channels, x, y]
        batch_size = wrf.shape[1]
        wrf_conv = [None] * self.tra_frames
        for i in range(self.tra_frames):
            wrf_conv[i] = self.wrf_encoder_conv2d(wrf[i])
            # wrf[i]=torch.Size([4, 29, 159, 159]),wrf_conv[i]=torch.Size([4, 64, 39, 39])
            # print('wrf[i]={},wrf_conv[i]={}'.format(wrf[i].shape, wrf_conv[i].shape))
        wrf_h = torch.zeros([batch_size, 32, wrf_conv[0].shape[2], wrf_conv[0].shape[3]], dtype=torch.float32).to(wrf.device)
        wrf_c = torch.zeros([batch_size, 32, wrf_conv[0].shape[2], wrf_conv[0].shape[3]], dtype=torch.float32).to(wrf.device)

        for i in range(self.tra_frames):
            wrf_h, wrf_c = self.wrf_encoder_convLSTM2D(wrf_conv[i], wrf_h, wrf_c)

        return wrf_h, wrf_c

class Encoder_obs_model(nn.Module):
    def __init__(self, tra_frames, channels, row_col):
        super(Encoder_obs_model, self).__init__()
        self.tra_frames = tra_frames
        self.obs_encoder_conv2d = nn.Sequential(
                nn.Conv2d(channels, 4, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(4, 4, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        self.obs_encoder_convLSTM2D = ConvLSTM2D(4, 8, kernel_size=5, img_rowcol=(row_col//2)//2)

    def forward(self, obs):
        # obs : [frames, batch_size, channels, x, y]
        batch_size = obs.shape[1]
        obs_conv = [None] * self.tra_frames
        for i in range(self.tra_frames):
            obs_conv[i] = self.obs_encoder_conv2d(obs[i])
            # obs_conv[i].shape=torch.Size([4, 4, 39, 39])

        obs_h = torch.zeros([batch_size, 8, obs_conv[0].shape[2], obs_conv[0].shape[3]], dtype=torch.float32).to(obs.device)
        obs_c = torch.zeros([batch_size, 8, obs_conv[0].shape[2], obs_conv[0].shape[3]], dtype=torch.float32).to(obs.device)

        for i in range(self.tra_frames):
            obs_h, obs_c = self.obs_encoder_convLSTM2D(obs_conv[i], obs_h, obs_c)

        return obs_h, obs_c

class Fusion_model(nn.Module):
    def __init__(self, filters_list_input, filters_list_output):
        super(Fusion_model, self).__init__()
        self.filters_list_output = filters_list_output
        fusion_conv_h = [None] * len(self.filters_list_output)
        fusion_conv_c = [None] * len(self.filters_list_output)

        # filters_list_input=[8, 32],filters_list_output=[16, 8]
        # print('filters_list_input={},filters_list_output={}'.format(filters_list_input, filters_list_output))
        # fusion_conv_encoder = [None] * len(self.filters_list)
        # len(self.filters_list_output)=2
        # print('len(self.filters_list_output)={}'.format(len(self.filters_list_output)))

        for i in range(len(self.filters_list_output)):
            fusion_conv_h[i] = nn.Sequential(
                nn.Conv2d(filters_list_input[i], self.filters_list_output[i], kernel_size=1, stride=1),
                nn.BatchNorm2d(self.filters_list_output[i]),
                nn.ReLU()
            )
            fusion_conv_c[i] = nn.Sequential(
                nn.Conv2d(filters_list_input[i], self.filters_list_output[i], kernel_size=1, stride=1),
                nn.BatchNorm2d(self.filters_list_output[i]),
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

            #h_concat[0].shape=torch.Size([4, 16, 39, 39]), c_concat[0].shape=torch.Size([4, 16, 39, 39])
            #h_concat[1].shape=torch.Size([4, 8, 39, 39]), c_concat[1].shape=torch.Size([4, 8, 39, 39])
            # print('h_concat[{}].shape={}, c_concat[i].shape={}'.format(i,h_concat[i].shape, c_concat[i].shape))
            # encoder_concat[i] = self.fusion_conv_encoder[i](encoder_output_list[i])

        h_concat = torch.cat(h_concat, dim=1)
        c_concat = torch.cat(c_concat, dim=1)



        # h_concat.shape=torch.Size([4, 24, 39, 39]), c_concat.shape=torch.Size([4, 24, 39, 39])


        # encoder_concat = torch.cat(encoder_concat, dim=1)
        # fusion_output = torch.cat([h_concat, c_concat, encoder_concat], dim=1)
        return h_concat, c_concat

class Decoder_model(nn.Module):
    def __init__(self, pre_frames, ConvLSTM2D_filters, row_col):
        super(Decoder_model, self).__init__()
        self.pre_frames = pre_frames
        self.decoder_conv2D = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.decoder_convLSTM2D = ConvLSTM2D(4, ConvLSTM2D_filters, kernel_size=5, img_rowcol=(row_col//2)//2)

        self.decoder_transconv2D = nn.Sequential(
            nn.ConvTranspose2d(ConvLSTM2D_filters, ConvLSTM2D_filters, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(ConvLSTM2D_filters, ConvLSTM2D_filters, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(ConvLSTM2D_filters, 1, kernel_size=1, stride=1),
        )



    def forward(self, frame, h, c):
        pre_frames = [None] * self.pre_frames
        h_frames = [None] * self.pre_frames
        for i in range(self.pre_frames):
            frame = self.decoder_conv2D(frame)
            h, c = self.decoder_convLSTM2D(frame, h, c)
            h_frames[i] = h.detach()
            pre_frames[i] = self.decoder_transconv2D(h)
            frame = torch.sigmoid(pre_frames[i])
            # pre_frames[i].shape =  ([4, 1, 159, 159])
            # frame.shape=torch.Size([4, 1, 159, 159])

        # 此时pre是个list 长度为预测时间
        pre_frames = torch.stack(pre_frames, dim=0)
        # torch.Size([3, 4, 1, 159, 159])

        h_frames = torch.stack(h_frames, dim=0)
        # ([12, 4, 24, 39, 39])


        return pre_frames ,h_frames


class LightNet_Model(nn.Module):
     # obs_tra_frames = TruthHistoryHourNum   wrf_tra_frames=config_dict['ForecastHourNum']
     #  wrf_channels=config_dict['WRFChannelNum'],
    def __init__(self, obs_tra_frames, obs_channels, wrf_tra_frames, wrf_channels, row_col):
        super(LightNet_Model, self).__init__()
        pre_frames = wrf_tra_frames
        self.encoder_obs_model = Encoder_obs_model(obs_tra_frames, obs_channels, row_col)
        self.encoder_wrf_model = Encoder_wrf_model(wrf_tra_frames, wrf_channels, row_col)
        filters_list_output = [16, 8]
        self.fusion_model = Fusion_model([8, 32], filters_list_output)   # [obs, wrf]
        self.decoder_model = Decoder_model(pre_frames, sum(filters_list_output), row_col)

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
        #  wrf_h.shape=torch.Size([4, 32, 39, 39]),wrf_c.shape=torch.Size([4, 32, 39, 39]),obs_h.shape=torch.Size([4, 8, 39, 39]),obs_c.shape=torch.Size([4, 8, 39, 39])

        h_list.append(wrf_h)
        c_list.append(wrf_c)

        # fusion
        fusion_h, fusion_c = self.fusion_model(h_list, c_list)

        #  h_list=2 c_list=2
        # print('h_list={}c_list={}'.format(len(h_list), len(c_list)))

        # fusion_h[0].shape=torch.Size([24, 39, 39]), fusion_c[0].shape=torch.Size([24, 39, 39])
        # print('fusion_h[0].shape={}, fusion_c[0].shape={}'.format(fusion_h[0].shape, fusion_c[0].shape))



        # decoder torch.Size([3, 4, 1, 159, 159])
        pre_frames, h_frames = self.decoder_model(obs[-1], fusion_h, fusion_c)

        #h_frames.shape = ([12, 4, 24, 39, 39])

        pre_frames = pre_frames.permute(1, 0, 3, 4, 2).contiguous()
        # pre_frames = torch.sigmoid(pre_frames)
        # ([4, 3, 159, 159, 1])

        return pre_frames, h_frames
