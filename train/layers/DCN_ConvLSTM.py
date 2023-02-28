import torch
import torch.nn as nn
from deformable_convolution.modules import ModulatedDeformConv

class DCN_ConvLSTM2D(nn.Module):
    def __init__(self, channels, filters, kernel_size, img_rowcol):
        super(DCN_ConvLSTM2D, self).__init__()
        # self.channels = channels
        self.filters = filters
        self.padding = kernel_size // 2
        # self.conv_x = nn.Conv2d(channels, filters * 4, kernel_size=kernel_size, stride=1, padding=self.padding, bias=True)
        # self.conv_h = nn.Conv2d(filters, filters * 4, kernel_size=kernel_size, stride=1, padding=self.padding, bias=False)
        self.conv_x = ModulatedDeformConv(channels, filters * 4, kernel_size=kernel_size, stride=1, padding=self.padding, bias=True)
        self.conv_h = nn.Conv2d(filters, filters * 4, kernel_size=kernel_size, stride=1, padding=self.padding, bias=False)
        self.mul_c = nn.Parameter(torch.zeros([1, filters * 3, img_rowcol, img_rowcol], dtype=torch.float32))
        self.conv_offset_mask_x = nn.Conv2d(channels,
                                          out_channels=3 * kernel_size * kernel_size,
                                          kernel_size=kernel_size,
                                          stride=1,
                                          padding=self.padding,
                                          bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask_x.weight.data.zero_()
        self.conv_offset_mask_x.bias.data.zero_()

    def forward(self, x, h, c):
        # x -> [batch_size, channels, x, y]
        out = self.conv_offset_mask_x(x)
        o1, o2, mask_x = torch.chunk(out, 3, dim=1)
        offset_x = torch.cat((o1, o2), dim=1)
        mask_x = torch.sigmoid(mask_x)

        x_concat = self.conv_x(x, offset_x, mask_x)
        h_concat = self.conv_h(h)
        i_x, f_x, c_x, o_x = torch.split(x_concat, self.filters, dim=1)
        i_h, f_h, c_h, o_h = torch.split(h_concat, self.filters, dim=1)
        i_c, f_c, o_c = torch.split(self.mul_c, self.filters, dim=1)
        i_t = torch.sigmoid(i_x + i_h + i_c * c)
        f_t = torch.sigmoid(f_x + f_h + f_c * c)
        c_t = torch.relu(c_x + c_h)
        c_next = i_t * c_t + f_t * c
        o_t = torch.sigmoid(o_x + o_h + o_c * c_next)
        h_next = o_t * torch.relu(c_next)
        return h_next, c_next
