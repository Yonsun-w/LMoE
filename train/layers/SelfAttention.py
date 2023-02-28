import torch.nn as nn


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

if __name__ == "__main__":
    # from config import read_config
    # config_dict = read_config()
    # wrf = torch.zeros(1, 12, config_dict['GridRowColNum'], config_dict['GridRowColNum'], 29)
    # obs = torch.zeros(1, 3, config_dict['GridRowColNum'], config_dict['GridRowColNum'], 1)
    config_dict = {}
    config_dict['GridRowColNum'] = 159
    model = SelfAttention(channels=8, layers=1)
    print('# generator parameters:', sum(param.numel() for param in model.parameters()))
