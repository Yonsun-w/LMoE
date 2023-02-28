import torch
import torch.nn as nn

class ConvLSTM2D_Attn(nn.Module):
    def __init__(self, channels, filters, kernel_size, img_rowcol):
        super(ConvLSTM2D_Attn, self).__init__()
        # self.channels = channels
        self.filters = filters
        self.padding = kernel_size // 2
        # self.kernel_size = kernel_size
        # self.strides = strides
        self.conv_x = nn.Conv2d(channels, filters * 4, kernel_size=kernel_size, stride=1, padding=self.padding, bias=True)
        self.conv_h = nn.Conv2d(filters, filters * 4, kernel_size=kernel_size, stride=1, padding=self.padding, bias=False)
        self.mul_c = nn.Parameter(torch.zeros([1, filters * 3, img_rowcol, img_rowcol], dtype=torch.float32))
        self.self_attn = SelfAttention(channels=filters, layers=1)


    def forward(self, x, h, c):
        # x -> [batch_size, channels, x, y]
        x_concat = self.conv_x(x)
        h_concat = self.conv_h(h)
        i_x, f_x, c_x, o_x = torch.split(x_concat, self.filters, dim=1)
        i_h, f_h, c_h, o_h = torch.split(h_concat, self.filters, dim=1)
        i_c, f_c, o_c = torch.split(self.mul_c, self.filters, dim=1)
        i_t = torch.sigmoid(i_x + i_h + i_c * c)
        f_t = torch.sigmoid(f_x + f_h + f_c * c)
        c_t = torch.tanh(c_x + c_h)
        c_next = i_t * c_t + f_t * c
        o_t = torch.sigmoid(o_x + o_h + o_c * c_next)
        h_next = o_t * torch.tanh(c_next)
        h_next = self.self_attn(h_next)
        return h_next, c_next

class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead=2, dim_feedforward=256, dropout=0.0, normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos, src_mask=None, src_key_padding_mask=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class SelfAttention(nn.Module):
    def __init__(self, channels, layers):
        super().__init__()
        self.self_attn_stack = nn.ModuleList([SelfAttentionLayer(channels) for _ in range(layers)])

    def forward(self, src):
        batch_size, channels, x, y = src.shape
        # pos = self.pos_emb(src)
        pos = None
        src = src.flatten(2).permute(2, 0, 1)
        # pos = pos.flatten(2).permute(2, 0, 1)
        for layer in self.self_attn_stack:
            src = layer(src, pos)
        src = src.reshape([x, y, batch_size, channels]).permute(2, 3, 0, 1)
        return src
