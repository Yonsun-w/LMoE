import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.CNN_module = nn.Sequential(
            nn.Conv2d(29, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


    def forward(self, wrf):
        pass

class LiteEncoder(nn.Module):
    def __init__(self):
        super(LiteEncoder, self).__init__()
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
        self.encoder = nn.Conv2d(5, 8, kernel_size=5, stride=1, padding=2)


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
        wrf_enc = self.encoder(wrf_enc)
        return wrf_enc

class LiteEncoder_SSIM(nn.Module):
    def __init__(self):
        super(LiteEncoder_SSIM, self).__init__()
        self.conv2d_qice = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2d_qsnow = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2d_qgroup = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=5, stride=1, padding=2),
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
        self.encoder = nn.Conv2d(5, 8, kernel_size=5, stride=1, padding=2)


    def forward(self, wrf):
        wrf_qice = wrf[:, 0:9]
        wrf_qsnow = wrf[:, 9:18]
        wrf_qgroup = wrf[:, 18:27]
        wrf_w = wrf[:, 27:28]
        wrf_rain = wrf[:, 28:29]

        wrf_qice = self.conv2d_qice(wrf_qice)
        wrf_qsnow = self.conv2d_qsnow(wrf_qsnow)
        wrf_qgroup = self.conv2d_qgroup(wrf_qgroup)
        wrf_w = self.conv2d_w(wrf_w)
        wrf_rain = self.conv2d_rain(wrf_rain)

        wrf_enc = torch.cat([wrf_qice, wrf_qsnow, wrf_qgroup, wrf_w, wrf_rain], dim=1)
        wrf_enc = self.encoder(wrf_enc)
        return wrf_enc

if __name__ == '__main__':
    batchsize = 2
    channels = 29
    wrf = torch.zeros([batchsize, channels, 159, 159], device='cuda')
    model = LiteEncoder()
    wrf_enc = model(wrf)
    print(wrf_enc.shape)
    print('# generator parameters:', sum(param.numel() for param in model.parameters()))
    print('# generator parameters:', sum(param.numel() for param in Encoder().parameters()))

