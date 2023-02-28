import torch
import torch.nn as nn
from layers.ConvLSTM import ConvLSTM2D
from layers.moe_layer.sub.moe_ADSNet import ADSNet_Model
from layers.moe_layer.sub.moe_LightNet import LightNet_Model
from layers.moe_layer.transformer import Moe_Atten

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1



        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, query, key):
        # 在channel维度上进行拼接
        query = torch.cat([query, key], dim=1)
        # 感觉在这里应该加一层卷积
        avg_out = torch.mean(query, dim=1, keepdim=True)
        max_out, _ = torch.max(query, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        out = torch.sigmoid(out)
        # scale = key * torch.sigmoid(out)
        # out就是权重
        return torch.sum(out)


class Attention_model(nn.Module):
    # obs = 1 h = 64
    def __init__(self, obs_channels, h_channels):
        super(Attention_model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(obs_channels, h_channels, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4)
        )
        self.att_weight1 = nn.Sequential(
            nn.Conv2d(h_channels, h_channels, kernel_size=5, stride=1, padding=2)
        )
        self.att_weight2 = nn.Sequential(
            nn.Conv2d(h_channels, h_channels, kernel_size=5, stride=1, padding=2)
        )
        self.CNN_hw = nn.Conv2d(128 * 2, 1, kernel_size=1, stride=1)

      # h_old 64  39 39      obs 1  159 159
    def forward(self, a_h_old, l_h_old, obs):

        # 对obs进行放缩 也变成64 39 39
        obs = self.conv1(obs)

        # 首先计算channel的注意力
        # 用obs 点× 专家输入 然后压缩
        a_h_c = torch.mul(obs, a_h_old)
        l_h_c = torch.mul(obs, l_h_old)
        # 这是每个专家通道注意力
        a_h_c = torch.sum(a_h_c, [2,3], keepdim=True).unsqueeze(dim=0)
        l_h_c = torch.sum(l_h_c, [2, 3], keepdim=True).unsqueeze(dim=0)
        channel_att = torch.cat([a_h_c, l_h_c], dim=0)
        # 通道注意力
        channel_att = torch.softmax(channel_att, dim=0)



        # 对a_h l_h进行采样
        a_h_old = self.att_weight1(a_h_old)
        l_h_old = self.att_weight2(l_h_old)

        # obs 分别乘 a_h_old
        a_h_old = torch.mul(obs, a_h_old).unsqueeze(dim=0)
        l_h_old = torch.mul(obs, l_h_old).unsqueeze(dim=0)

        # 专家数 64 39 39 分别代表每个专家每个点的权重
        res = torch.cat([a_h_old, l_h_old], dim=0)
        res = torch.softmax(res, dim=0)

        return res


class MOE_Model(nn.Module):
     # obs_tra_frames = TruthHistoryHourNum   wrf_tra_frames=config_dict['ForecastHourNum']
     #  wrf_channels=config_dict['WRFChannelNum'],

    def __init__(self, truth_history_hour_num,forecast_hour_num, row_col,wrf_channels, obs_channel, ads_net_model_path = '/data/wenjiahua/light_data/ADSNet_testdata/gyRecod/ads/ADSNet_model_maxETS.pkl',
                 light_net_model_path='/home/wenjiahua/windows_project/gy_train/layers/moe_layer/LightNet_model_maxETS.pkl'):
        super(MOE_Model, self).__init__()

        self.truth_history_hour_num = truth_history_hour_num
        self.forecast_hour_num = forecast_hour_num

        self.wrf_batchnorm = nn.BatchNorm3d(wrf_channels)
        self.obs_batchnorm = nn.BatchNorm3d(obs_channel)


        self.fusion_conv = nn.Sequential(
            nn.Conv2d(wrf_channels + obs_channel,  (wrf_channels + obs_channel)//2, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d( (wrf_channels + obs_channel)//2,  (wrf_channels + obs_channel)//2, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fusion_convLSTM2D = ConvLSTM2D((wrf_channels + obs_channel)//2, 8, kernel_size=5, img_rowcol=(row_col//2)//2)

        # self.ads_encoder_convLSTM2D = ConvLSTM2D(128, 8, kernel_size=5, img_rowcol=(row_col//2)//2)
        # self.light_encoder_convLSTM2D = ConvLSTM2D(24, 8, kernel_size=5, img_rowcol=(row_col//2)//2)


        #定义transformer
        self.transformer = Moe_Atten()



        ADSNet_expert = ADSNet_Model(obs_tra_frames=truth_history_hour_num, obs_channels=obs_channel,
                             wrf_tra_frames=forecast_hour_num,
                             wrf_channels=wrf_channels, row_col=row_col).to('cuda')

        LightNet_expert = LightNet_Model(obs_tra_frames=truth_history_hour_num, obs_channels=obs_channel,
                             wrf_tra_frames=forecast_hour_num,
                             wrf_channels=wrf_channels, row_col=row_col).to('cuda')
        ads_model_file = torch.load(ads_net_model_path, map_location=torch.device('cuda'))
        ADSNet_expert.load_state_dict(ads_model_file)

        light_model_file = torch.load(light_net_model_path, map_location=torch.device('cuda'))
        LightNet_expert.load_state_dict(light_model_file)
        ADSNet_expert.requires_grad = False
        LightNet_expert.requires_grad = False

        self.row_col = row_col

        self.expert_list = [ADSNet_expert, LightNet_expert]

        self.CNN_module3 = nn.Sequential(
            nn.ConvTranspose2d(128 + 24, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1)
        )
        self.attention_expert = nn.ModuleList([Attention_model(obs_channels=1, h_channels=64).to('cuda') for i in range(self.forecast_hour_num)])


        self.ads_norm = nn.BatchNorm3d(128)
        self.light_norm = nn.BatchNorm3d(24)

        self.spatial_att = SpatialAttention()


         # 前两个参数是当前的输入 为了预测之后的, 中间两个是为了获取在当前时间到预测时间长度之前的专家模拟输入
         # 最后一个参数是当前到预测长度之前的真实输入
    def forward(self, wrf, obs, wrf_old):

        batch_size = wrf.size(0)
        # 首先得到不同专家模型的结果
        ads_pre, ads_pre_h = self.expert_list[0](wrf, obs)
        light_pre, light_pre_h = self.expert_list[1](wrf, obs)

        # h 不用动 已经是 [frames, batch_size, channels, x, y] ,但是为归一，所以也不得不先移动一下
        #  [frames, batch_size, channels, x, y] - > [batch, channels, frames, x,y]
        ads_pre_h = ads_pre_h.permute(1, 2, 0, 3, 4)
        light_pre_h = light_pre_h.permute(1, 2, 0, 3, 4)
        #ads_pre_h = self.ads_norm(ads_pre_h)
        #light_pre_h = self.light_norm(light_pre_h)
        # 改回来
        ads_pre_h = ads_pre_h.permute(2,0,1,3,4).contiguous()
        light_pre_h = light_pre_h.permute(2,0,1,3,4).contiguous()


        # 归一 [batch,trame,h,w,channels] - > [batch,chanel,trame,h,w]
        wrf_old = wrf_old.permute(0, 4, 1, 2, 3)
        wrf_old = self.wrf_batchnorm(wrf_old)

        wrf_old = wrf_old.permute(2,0,1,3,4)
        obs = obs.permute(1,0,4,2,3)

        tru_hour = wrf_old.shape[0]


        # 拼接 在channel维度上
        fusion = torch.cat([obs, wrf_old], dim=2)


        obs,wrf_old = self.transformer(obs,wrf_old)
        # 这里应该是39
        h = w = (self.row_col//2)//2

        fusion_h = torch.zeros([batch_size, 8, h, w], dtype=torch.float32).to(obs.device)
        fusion_c = torch.zeros([batch_size, 8, h, w], dtype=torch.float32).to(obs.device)

        # fusion 应该是 frames,batch,channel 合并, 159,159
        for i in range(tru_hour):
            fusion_tmp = self.fusion_conv(fusion[i])
            fusion_h, fusion_c = self.fusion_convLSTM2D(fusion_tmp, fusion_h, fusion_c)

        # 我们得到了一个fusion_h 这个h没有时间纬度 但是包含了时序讯息 他将作为query 查询

        # 专家的输出也应该在时间维度上叠加 然后得到一个整体的
        # 然后在空间纬度计算注意力
        ads_h = torch.sum(ads_pre_h,dim=0)
        light_h = torch.sum(light_pre_h,dim=0)

        ads_weight = self.spatial_att(fusion_h, ads_h).unsqueeze(dim=0)
        light_weight = self.spatial_att(fusion_h, light_h).unsqueeze(dim=0)

        weights = torch.cat([ads_weight,light_weight], dim=0)
        weights = torch.softmax(weights,dim=0)
        ads_weight = weights[0].contiguous()
        light_weight = weights[1].contiguous()

        ads_pre_h = ads_pre_h * ads_weight
        light_pre_h = light_pre_h * light_weight

        res = [None] * self.forecast_hour_num
        for i in range(self.forecast_hour_num):
            #  4, 128, 39, 39
            a_h = ads_pre_h[i]
            #  4, 24, 39, 39
            l_h = light_pre_h[i]

            a_l_r = torch.cat([a_h,l_h],dim=1).contiguous()

            a_l_r = self.CNN_module3(a_l_r)
            # 此时得到的是真是的
            res[i] = a_l_r

        res = torch.stack(res, dim=0)

        #  batch hour  1  159 159
        res = res.permute(1, 0, 3, 4, 2)

        return res

if __name__ == '__main__':
    TruthHistoryHourNum = 3
    ForecastHourNum = 12
    GridRowColNum = 159
    WRFChannelNum = 29
    batch = 1
    wrf = torch.randn(batch, ForecastHourNum, GridRowColNum, GridRowColNum, WRFChannelNum).to('cuda')
    obs = torch.randn(batch, TruthHistoryHourNum, GridRowColNum, GridRowColNum, 1).to('cuda')
    wrf_old = torch.randn(batch,  TruthHistoryHourNum , GridRowColNum, GridRowColNum, WRFChannelNum).to('cuda')

    model = MOE_Model(truth_history_hour_num=TruthHistoryHourNum,
                      forecast_hour_num=ForecastHourNum,
                      row_col=GridRowColNum, wrf_channels=WRFChannelNum, obs_channel=1).to('cuda')

    x = model(wrf,obs,wrf_old)
