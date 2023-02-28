from functools import reduce
import copy
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F

class E3DLSTM_Model(nn.Module):
    def __init__(self, obs_tra_frames, obs_channels, wrf_tra_frames, wrf_channels, config_dict):
        super().__init__()
        self.config_dict = config_dict
        self.window_size = 2
        self.hidden_size = 32
        kernel = (2, 5, 5)
        lstm_layers = 1

        mn = (config_dict['GridRowColNum'] // 2) // 2
        self.obs_subsampling_module = nn.Sequential(
            nn.Conv2d(obs_channels, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 8, kernel_size=5, stride=1, padding=2),
            nn.LayerNorm([8, mn, mn], elementwise_affine=True)
        )
        self.wrf_subsampling_module = LiteEncoder(config_dict=config_dict)
        input_shape_obs = (8, config_dict['TruthHistoryHourNum'], mn, mn)
        self.encoder_obs = E3DLSTM(
            input_shape_obs, self.hidden_size, lstm_layers, kernel
        )
        input_shape_wrf = (8, self.window_size, mn, mn)
        self.encoder_wrf = E3DLSTM(
            input_shape_wrf, self.hidden_size, lstm_layers, kernel
        )
        self.decoder = nn.Conv3d(
            self.hidden_size * 2, 8, kernel, padding=(0, 2, 2)
        )
        self.upsampling_module = nn.Sequential(
            nn.ConvTranspose2d(8, 8, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1, stride=1)
        )

    def window(self, seq, size=2, stride=1):
        """Returns a sliding window (of width n) over data from the iterable
           E.g., s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...  
        """
        it = iter(seq)
        result = []
        for elem in it:
            result.append(elem)
            if len(result) == size:
                yield result
                result = result[stride:]

    def forward(self, wrf, obs):
        # obs : [batch_size, frames, x, y, channels] -> [batch_size, channels, frames, x, y]
        obs = obs.permute(0, 4, 1, 2, 3).contiguous()
        # wrf : [batch_size, frames, x, y, channels] -> [batch_size, channels, frames, x, y]
        wrf = wrf.permute(0, 4, 1, 2, 3).contiguous()

        # subsampling
        obs_subsampling, wrf_subsampling = [], []
        for i in range(obs.shape[2]):
            obs_subsampling.append(self.obs_subsampling_module(obs[:, :, i]))
        obs = torch.stack(obs_subsampling, dim=2)
        for i in range(wrf.shape[2]):
            wrf_subsampling.append(self.wrf_subsampling_module(wrf[:, :, i]))
        wrf = torch.stack(wrf_subsampling, dim=2)

        cat_shape = (wrf.shape[0], wrf.shape[1], self.window_size - 1, wrf.shape[3], wrf.shape[4])
        wrf = torch.cat([wrf, torch.zeros(cat_shape, device=self.config_dict['Device'])], dim=2)

        pre_frames = []
        his_frames_encoder = torch.zeros([wrf.shape[0],
                                          self.hidden_size,
                                          self.config_dict['TruthHistoryHourNum'] + self.config_dict['ForecastHourNum'],
                                          wrf.shape[3],
                                          wrf.shape[4]]).to(wrf.device)
        his_frames_encoder[:, :, :self.config_dict['TruthHistoryHourNum']] = self.encoder_obs(obs.unsqueeze(dim=0))

        his_bias = self.config_dict['TruthHistoryHourNum'] - self.window_size
        for indices in self.window(range(self.config_dict['ForecastHourNum'] + self.window_size - 1), self.window_size):
            # frames_seq.append(wrf[:, :, indices[0]: indices[-1] + 1])
            input_wrf = wrf[:, :, indices[0]: indices[-1] + 1].unsqueeze(dim=0)
            wrf_enc = self.encoder_wrf(input_wrf)
            his_enc = his_frames_encoder[:, :, indices[0] + his_bias: indices[-1] + 1 + his_bias]
            enc = torch.cat([his_enc, wrf_enc], dim=1)
            dec = self.decoder(enc)
            dec = self.upsampling_module(dec[:, :, 0])
            pre_frames.append(dec)
        pre_frames = torch.stack(pre_frames, dim=2)
        pre_frames = pre_frames.permute(0, 2, 3, 4, 1).contiguous()
        return pre_frames


class E3DLSTM(nn.Module):
    def __init__(self, input_shape, hidden_size, num_layers, kernel_size, tau=2):
        super().__init__()

        self._tau = tau
        self._cells = []

        input_shape = list(input_shape)
        for i in range(num_layers):
            cell = E3DLSTMCell(input_shape, hidden_size, kernel_size)
            # NOTE hidden state becomes input to the next cell
            input_shape[0] = hidden_size
            self._cells.append(cell)
            # Hook to register submodule
            setattr(self, "cell{}".format(i), cell)

    def forward(self, input):
        # NOTE (seq_len, batch, input_shape)
        batch_size = input.size(1)
        c_history_states = []
        h_states = []
        outputs = []

        for step, x in enumerate(input):
            for cell_idx, cell in enumerate(self._cells):
                if step == 0:
                    c_history, m, h = self._cells[cell_idx].init_hidden(
                        batch_size, self._tau, input.device
                    )
                    c_history_states.append(c_history)
                    h_states.append(h)

                # NOTE c_history and h are coming from the previous time stamp, but we iterate over cells
                c_history, m, h = cell(
                    x, c_history_states[cell_idx], m, h_states[cell_idx]
                )
                c_history_states[cell_idx] = c_history
                h_states[cell_idx] = h
                # NOTE hidden state of previous LSTM is passed as input to the next one
                x = h

            outputs.append(h)

        # NOTE Concat along the channels
        return torch.cat(outputs, dim=1)


class E3DLSTMCell(nn.Module):
    def __init__(self, input_shape, hidden_size, kernel_size):
        super().__init__()

        in_channels = input_shape[0]
        self._input_shape = input_shape
        self._hidden_size = hidden_size

        # memory gates: input, cell(input modulation), forget
        self.weight_xi = ConvDeconv3d(in_channels, hidden_size, kernel_size)
        self.weight_hi = ConvDeconv3d(hidden_size, hidden_size, kernel_size, bias=False)

        self.weight_xg = copy.deepcopy(self.weight_xi)
        self.weight_hg = copy.deepcopy(self.weight_hi)

        self.weight_xr = copy.deepcopy(self.weight_xi)
        self.weight_hr = copy.deepcopy(self.weight_hi)

        memory_shape = list(input_shape)
        memory_shape[0] = hidden_size

        self.layer_norm = nn.LayerNorm(memory_shape)

        # for spatiotemporal memory
        self.weight_xi_prime = copy.deepcopy(self.weight_xi)
        self.weight_mi_prime = copy.deepcopy(self.weight_hi)

        self.weight_xg_prime = copy.deepcopy(self.weight_xi)
        self.weight_mg_prime = copy.deepcopy(self.weight_hi)

        self.weight_xf_prime = copy.deepcopy(self.weight_xi)
        self.weight_mf_prime = copy.deepcopy(self.weight_hi)

        self.weight_xo = copy.deepcopy(self.weight_xi)
        self.weight_ho = copy.deepcopy(self.weight_hi)
        self.weight_co = copy.deepcopy(self.weight_hi)
        self.weight_mo = copy.deepcopy(self.weight_hi)

        self.weight_111 = nn.Conv3d(hidden_size + hidden_size, hidden_size, 1)

    def self_attention(self, r, c_history):
        batch_size = r.size(0)
        channels = r.size(1)
        r_flatten = r.view(batch_size, -1, channels)
        # BxtaoTHWxC
        c_history_flatten = c_history.view(batch_size, -1, channels)

        # Attention mechanism
        # BxTHWxC x BxtaoTHWxC' = B x THW x taoTHW
        scores = torch.einsum("bxc,byc->bxy", r_flatten, c_history_flatten)
        attention = F.softmax(scores, dim=2)

        return torch.einsum("bxy,byc->bxc", attention, c_history_flatten).view(*r.shape)

    def self_attention_fast(self, r, c_history):
        # Scaled Dot-Product but for tensors
        # instead of dot-product we do matrix contraction on twh dimensions
        scaling_factor = 1 / (reduce(operator.mul, r.shape[-3:], 1) ** 0.5)
        scores = torch.einsum("bctwh,lbctwh->bl", r, c_history) * scaling_factor

        attention = F.softmax(scores, dim=0)
        return torch.einsum("bl,lbctwh->bctwh", attention, c_history)

    def forward(self, x, c_history, m, h):
        # Normalized shape for LayerNorm is CxT×H×W
        normalized_shape = list(h.shape[-3:])

        def LR(input):
            return F.layer_norm(input, normalized_shape)

        # R is CxT×H×W
        r = torch.sigmoid(LR(self.weight_xr(x) + self.weight_hr(h)))
        i = torch.sigmoid(LR(self.weight_xi(x) + self.weight_hi(h)))
        g = torch.tanh(LR(self.weight_xg(x) + self.weight_hg(h)))

        recall = self.self_attention_fast(r, c_history)

        c = i * g + self.layer_norm(c_history[-1] + recall)

        i_prime = torch.sigmoid(LR(self.weight_xi_prime(x) + self.weight_mi_prime(m)))
        g_prime = torch.tanh(LR(self.weight_xg_prime(x) + self.weight_mg_prime(m)))
        f_prime = torch.sigmoid(LR(self.weight_xf_prime(x) + self.weight_mf_prime(m)))

        m = i_prime * g_prime + f_prime * m
        o = torch.sigmoid(
            LR(
                self.weight_xo(x)
                + self.weight_ho(h)
                + self.weight_co(c)
                + self.weight_mo(m)
            )
        )
        h = o * torch.tanh(self.weight_111(torch.cat([c, m], dim=1)))

        # TODO is it correct FIFO?
        c_history = torch.cat([c_history[1:], c[None, :]], dim=0)
        # nice_print(**locals())

        return (c_history, m, h)

    def init_hidden(self, batch_size, tau, device=None):
        memory_shape = list(self._input_shape)
        memory_shape[0] = self._hidden_size
        c_history = torch.zeros(tau, batch_size, *memory_shape, device=device)
        m = torch.zeros(batch_size, *memory_shape, device=device)
        h = torch.zeros(batch_size, *memory_shape, device=device)

        return (c_history, m, h)


class ConvDeconv3d(nn.Module):
    def __init__(self, in_channels, out_channels, *vargs, **kwargs):
        super().__init__()

        self.conv3d = nn.Conv3d(in_channels, out_channels, *vargs, **kwargs)
        # self.conv_transpose3d = nn.ConvTranspose3d(out_channels, out_channels, *vargs, **kwargs)

    def forward(self, input):
        # print(self.conv3d(input).shape, input.shape)
        # return self.conv_transpose3d(self.conv3d(input))
        return F.interpolate(self.conv3d(input), size=input.shape[-3:], mode="nearest")


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
        wrf_qice = torch.layer_norm(wrf_qice, normalized_shape=tuple(wrf_qice[0, 0].shape), eps=1e-30)
        wrf_qsnow = wrf[:, 12:16]
        wrf_qsnow = torch.layer_norm(wrf_qsnow, normalized_shape=tuple(wrf_qsnow[0, 0].shape), eps=1e-30)
        wrf_qgroup = wrf[:, 21:25]
        wrf_qgroup = torch.layer_norm(wrf_qgroup, normalized_shape=tuple(wrf_qgroup[0, 0].shape), eps=1e-30)
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
