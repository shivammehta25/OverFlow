"""
Glow-TTS Code from https://github.com/jaywalnut310/glow-tts
"""
import torch.nn as nn

import src.model.DecoderComponents.flows as flows
from src.utilities.functions import get_mask_from_len, squeeze, unsqueeze


class FlowSpecDecoder(nn.Module):
    def __init__(self, hparams):

        super().__init__()
        self.in_channels = hparams.n_mel_channels
        self.hidden_channels = hparams.flow_hidden_channels
        self.kernel_size = hparams.kernel_size_dec
        self.dilation_rate = hparams.dilation_rate
        self.n_blocks = hparams.n_blocks_dec
        self.n_layers = hparams.n_block_layers
        self.p_dropout = hparams.p_dropout_dec
        self.n_split = hparams.n_split
        self.n_sqz = hparams.n_sqz
        self.sigmoid_scale = hparams.sigmoid_scale
        self.gin_channels = hparams.gin_channels

        self.flows = nn.ModuleList()
        for b in range(hparams.n_blocks_dec):
            self.flows.append(flows.ActNorm(channels=hparams.n_mel_channels * hparams.n_sqz))
            self.flows.append(flows.InvConvNear(channels=hparams.n_mel_channels * self.n_sqz, n_split=hparams.n_split))
            self.flows.append(
                flows.CouplingBlock(
                    hparams.n_mel_channels * hparams.n_sqz,
                    hparams.flow_hidden_channels,
                    kernel_size=hparams.kernel_size_dec,
                    dilation_rate=hparams.dilation_rate,
                    n_layers=hparams.n_blocks_dec,
                    gin_channels=hparams.gin_channels,
                    p_dropout=hparams.p_dropout_dec,
                    sigmoid_scale=hparams.sigmoid_scale,
                )
            )

    def forward(self, x, x_lengths, g=None, reverse=False):
        """Calls Glow-TTS decoder

        Args:
            x (torch.FloatTensor): Input tensor (batch_len, n_mel_channels, T_max)
            x_lengths (torch.IntTensor): lens of mel spectrograms (batch_len)
            g (_type_, optional): _description_. Defaults to None.
            reverse (bool, optional): True when synthesising. Defaults to False.

        Returns:
            _type_: _description_
        """
        x, x_lengths, x_max_length = self.preprocess(x, x_lengths, x_lengths.max())

        x_mask = get_mask_from_len(x_lengths, x_max_length, device=x_lengths.device).float().unsqueeze(1)

        if not reverse:
            flows = self.flows
            logdet_tot = 0
        else:
            flows = reversed(self.flows)
            logdet_tot = None

        if self.n_sqz > 1:
            x, x_mask = squeeze(x, x_mask, self.n_sqz)
        for f in flows:
            if not reverse:
                x, logdet = f(x, x_mask, g=g, reverse=reverse)
                logdet_tot += logdet
            else:
                x, logdet = f(x, x_mask, g=g, reverse=reverse)
        if self.n_sqz > 1:
            x, x_mask = unsqueeze(x, x_mask, self.n_sqz)
        return x, x_lengths, logdet_tot

    def store_inverse(self):
        for f in self.flows:
            f.store_inverse()

    def preprocess(self, y, y_lengths, y_max_length):
        if y_max_length is not None:
            y_max_length = (y_max_length // self.n_sqz) * self.n_sqz
            y = y[:, :, :y_max_length]
        y_lengths = (y_lengths // self.n_sqz) * self.n_sqz
        return y, y_lengths, y_max_length
