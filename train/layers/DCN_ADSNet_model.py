import torch
import torch.nn as nn
from layers.ConvLSTM import ConvLSTM2D
from deformable_convolution.modules import ModulatedDeformConvPack

class Attention_model(nn.Module):
    def __init__(self, channels):
        super(Attention_model, self).__init__()
        self.channels = channels
        self.DWCNN_att = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=2, groups=channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=2, groups=channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, groups=channels),
        )
        self.CNN_hw = nn.Conv2d(32 * 2, 1, kernel_size=1, stride=1)


    def forward(self, wrf, h, c):
        att_wrf = self.DWCNN_att(wrf)
        att_hc = self.CNN_hw(torch.cat([h, c], dim=1))
        att = torch.mul(att_hc, att_wrf)
        e = torch.div(torch.sum(att, dim=[2, 3], keepdim=True), att_wrf.shape[3])
        alpha = torch.softmax(e, dim=1)
        att_res = torch.mul(wrf, alpha)
        return att_res



class DCNADSNet_Model(nn.Module):
    def __init__(self, obs_tra_frames, obs_channels, wrf_tra_frames, wrf_channels, config_dict):
        super(DCNADSNet_Model, self).__init__()
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
            nn.Conv2d(8, 32, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        self.encoder_c = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        self.CNN_module2 = nn.Sequential(
            nn.Conv2d(wrf_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.LayerNorm([32, mn, mn], elementwise_affine=True)
        )
        self.decoder_ConvLSTM = ConvLSTM2D(32, 32, kernel_size=5, img_rowcol=(config_dict['GridRowColNum']//2)//2)
        self.CNN_module3 = nn.Sequential(
            ModulatedDeformConvPack(32, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1)
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
            # wrf_att = self.attention(wrf[t], h, c)
            wrf_encoder = self.CNN_module2(wrf[t])
            h, c = self.decoder_ConvLSTM(wrf_encoder, h, c)
            pre = self.CNN_module3(h)
            pre_frames[t] = pre
        pre_frames = pre_frames.permute(1, 0, 3, 4, 2).contiguous()
        # print(pre_frames.shape)
        return pre_frames

if __name__ == "__main__":
    config_dict = {}
    config_dict['GridRowColNum'] = 159
    model = DCNADSNet_Model(obs_tra_frames=3, obs_channels=1, wrf_tra_frames=12, wrf_channels=29,
                                 config_dict=config_dict)
    print('# generator parameters:', sum(param.numel() for param in model.parameters()))

