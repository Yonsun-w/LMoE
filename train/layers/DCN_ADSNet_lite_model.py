import torch
import torch.nn as nn
from layers.ConvLSTM import ConvLSTM2D
from deformable_convolution.modules import ModulatedDeformConvPack

class LiteEncoder(nn.Module):
    def __init__(self, config_dict):
        super(LiteEncoder, self).__init__()
        mn = (config_dict['GridRowColNum'] // 2) // 2
        self.conv2d_qice = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2d_qsnow = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2d_qgroup = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2d_w = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2d_rain = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layernorm = nn.LayerNorm([5, mn, mn], elementwise_affine=True)
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )


    def forward(self, wrf):
        wrf_qice = wrf[:, 3:7]
        wrf_qsnow = wrf[:, 12:16]
        wrf_qgroup = wrf[:, 21:25]
        wrf_w = wrf[:, 27:28]
        wrf_rain = wrf[:, 28:29]

        wrf_qice = self.conv2d_qice(wrf_qice)
        wrf_qsnow = self.conv2d_qsnow(wrf_qsnow)
        wrf_qgroup = self.conv2d_qgroup(wrf_qgroup)
        wrf_w = self.conv2d_w(wrf_w)
        wrf_rain = self.conv2d_rain(wrf_rain)

        wrf_enc = torch.cat([wrf_qice, wrf_qsnow, wrf_qgroup, wrf_w, wrf_rain], dim=1)
        wrf_enc = self.layernorm(wrf_enc)
        wrf_enc = self.encoder(wrf_enc)
        return wrf_enc


class DCNADSNet_lite_Model(nn.Module):
    def __init__(self, obs_tra_frames, obs_channels, wrf_tra_frames, wrf_channels, config_dict):
        super(DCNADSNet_lite_Model, self).__init__()
        self.config_dict = config_dict
        self.obs_tra_frames = obs_tra_frames
        self.wrf_tra_frames = wrf_tra_frames
        mn = (config_dict['GridRowColNum'] // 2) // 2
        self.CNN_module1 = nn.Sequential(
            nn.Conv2d(obs_channels, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 8, kernel_size=5, stride=1, padding=2),
            nn.LayerNorm([8, mn, mn], elementwise_affine=True)
        )
        self.encoder_ConvLSTM = ConvLSTM2D(8, 8, kernel_size=5, img_rowcol=mn)
        self.encoder_h = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        self.encoder_c = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        self.CNN_module2 = LiteEncoder(config_dict=config_dict)
        self.decoder_ConvLSTM = ConvLSTM2D(8, 8, kernel_size=5, img_rowcol=(config_dict['GridRowColNum']//2)//2)
        self.CNN_module3 = nn.Sequential(
            ModulatedDeformConvPack(8, 8, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1, stride=1)
        )
        # self.attention = Attention_model(wrf_channels)

    def forward(self, wrf, obs):
        # obs : [batch_size, frames, x, y, channels] -> [frames, batch_size, channels, x, y]
        obs = obs.permute(1, 0, 4, 2, 3).contiguous()
        # wrf : [batch_size, frames, x, y, channels] -> [frames, batch_size, channels, x, y]
        wrf = wrf.permute(1, 0, 4, 2, 3).contiguous()

        batch_size = obs.shape[1]
        pre_frames = torch.zeros([self.wrf_tra_frames, batch_size, 1, wrf.shape[3], wrf.shape[4]]).to(wrf.device)

        h = torch.zeros([batch_size, 8, (self.config_dict['GridRowColNum']//2)//2, (self.config_dict['GridRowColNum']//2)//2], dtype=torch.float32).to(obs.device)
        c = torch.zeros([batch_size, 8, (self.config_dict['GridRowColNum']//2)//2, (self.config_dict['GridRowColNum']//2)//2], dtype=torch.float32).to(obs.device)
        for t in range(self.obs_tra_frames):
            obs_encoder = self.CNN_module1(obs[t])
            h, c = self.encoder_ConvLSTM(obs_encoder, h, c)
        h = self.encoder_h(h)
        c = self.encoder_c(c)
        for t in range(self.wrf_tra_frames):
            wrf_encoder = self.CNN_module2(wrf[t])
            h, c = self.decoder_ConvLSTM(wrf_encoder, h, c)
            pre = self.CNN_module3(h)
            pre_frames[t] = pre
        pre_frames = pre_frames.permute(1, 0, 3, 4, 2).contiguous()
        return pre_frames

if __name__ == "__main__":
    # from config import read_config
    # config_dict = read_config()
    # wrf = torch.zeros(1, 12, config_dict['GridRowColNum'], config_dict['GridRowColNum'], 29)
    # obs = torch.zeros(1, 3, config_dict['GridRowColNum'], config_dict['GridRowColNum'], 1)
    config_dict = {}
    config_dict['GridRowColNum'] = 159
    model = DCNADSNet_lite_Model(obs_tra_frames=3, obs_channels=1, wrf_tra_frames=12, wrf_channels=29, config_dict=config_dict)
    print('# generator parameters:', sum(param.numel() for param in model.parameters()))

