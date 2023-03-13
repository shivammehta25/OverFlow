"""
Glow-TTS Code from https://github.com/jaywalnut310/glow-tts
"""
import numpy as np
import torch
import torch.nn as nn

import src.model.DecoderComponents.flows as flows
from src.model.layers import LinearNorm
from src.model.transformer import Conformer, FFTransformer
from src.model.wavegrad import WaveGrad
from src.utilities.functions import get_mask_from_len, squeeze, unsqueeze


class FlowSpecDecoder(nn.Module):
    def __init__(self, hparams, in_channels, p_dropout=0.05):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hparams.flow_hidden_channels
        self.kernel_size = hparams.kernel_size_dec
        self.dilation_rate = hparams.dilation_rate
        self.n_blocks = hparams.n_blocks_dec
        self.n_layers = hparams.n_block_layers
        self.p_dropout = p_dropout
        self.n_split = hparams.n_split
        self.n_sqz = hparams.n_sqz
        self.sigmoid_scale = hparams.sigmoid_scale
        self.gin_channels = hparams.gin_channels

        self.flows = nn.ModuleList()
        for b in range(hparams.n_blocks_dec):
            self.flows.append(flows.ActNorm(channels=self.in_channels * hparams.n_sqz))
            self.flows.append(flows.InvConvNear(channels=self.in_channels * hparams.n_sqz, n_split=hparams.n_split))
            self.flows.append(
                flows.CouplingBlock(
                    self.in_channels * hparams.n_sqz,
                    hparams.flow_hidden_channels,
                    kernel_size=hparams.kernel_size_dec,
                    dilation_rate=hparams.dilation_rate,
                    n_layers=hparams.n_block_layers,
                    gin_channels=hparams.gin_channels,
                    p_dropout=self.p_dropout,
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
        assert x.shape == (
            x_lengths.shape[0],
            self.in_channels,
            x_lengths.max(),
        ), f"The shape of the  \
            input should be (batch_dim, n_mel_channels, T_max) but received {x.shape}"
        x, x_lengths, x_max_length = self.preprocess(x, x_lengths, x_lengths.max())

        x_mask = get_mask_from_len(x_lengths, x_max_length, device=x.device, dtype=x.dtype).unsqueeze(1)

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
            y_max_length = torch.div(y_max_length, self.n_sqz, rounding_mode="floor") * self.n_sqz
            y = y[:, :, :y_max_length]
        y_lengths = torch.div(y_lengths, self.n_sqz, rounding_mode="floor") * self.n_sqz
        return y, y_lengths, y_max_length


class DiffusionDecoder(nn.Module):
    def __init__(self, steps, noise_schedule) -> None:
        super().__init__()
        self.steps = steps
        self.model = WaveGrad()
        self.loss = nn.L1Loss()
        self.set_noise_schedule(noise_schedule)

    def set_noise_schedule(self, noise_schedule):
        self.beta = noise_schedule
        self.alpha = 1 - self.beta
        self.alpha_cum = np.cumprod(self.alpha)

    def forward(self, motion, motion_lens, z, z_lens):
        b, N, T = motion.shape

        s = torch.randint(1, self.steps + 1, [N], device=motion.device)
        l_a, l_b = self.beta[s - 1], self.beta[s]
        noise_scale = l_a + torch.rand(N, device=motion.device) * (l_b - l_a)
        noise_scale = noise_scale.unsqueeze(1)
        noise = torch.randn_like(motion)
        noisy_audio = noise_scale * motion + (1.0 - noise_scale**2) ** 0.5 * noise
        predicted = self.model(noisy_audio, z, noise_scale.squeeze(1))
        return self.loss(noise, predicted.squeeze(1))

    def sample(self, z):
        """

        Args:
            z (torch.FloatTensor): shape: (batch, n_motion_vectors, T)
        """
        audio = torch.randn(*z.shape, device=z.device)
        noise_scale = torch.from_numpy(self.alpha_cum**0.5).float().unsqueeze(1).to(z.device)

        for n in range(len(self.alpha) - 1, -1, -1):
            c1 = 1 / self.alpha[n] ** 0.5
            c2 = (1 - self.alpha[n]) / (1 - self.alpha_cum[n]) ** 0.5
            audio = c1 * (audio - c2 * self.model(audio, z, noise_scale[n]).squeeze(1))
            if n > 0:
                noise = torch.randn_like(audio)
                sigma = ((1.0 - self.alpha_cum[n - 1]) / (1.0 - self.alpha_cum[n]) * self.beta[n]) ** 0.5
                audio += sigma * noise
        return audio


class MotionDecoder(nn.Module):
    def __init__(self, hparams, decoder_type="transformer"):
        super().__init__()
        self.frame_rate_reduction_factor = hparams.frame_rate_reduction_factor
        self.downsampling_proj = nn.Conv1d(
            hparams.n_mel_channels,
            hparams.motion_decoder_param[decoder_type]["hidden_channels"],
            kernel_size=hparams.frame_rate_reduction_factor,
            stride=hparams.frame_rate_reduction_factor,
        )
        self.decoder_type = decoder_type
        if decoder_type == "transformer":
            self.encoder = FFTransformer(**hparams.motion_decoder_param[decoder_type])
        elif decoder_type == "conformer":
            self.encoder = Conformer(**hparams.motion_decoder_param[decoder_type])
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")
        self.out_proj = LinearNorm(
            hparams.motion_decoder_param[decoder_type]["hidden_channels"], hparams.n_motion_joints
        )

    def forward(self, x, input_lengths):
        """

        Args:
            x : (b, c, T_mel)
            input_lengths: (b)

        Returns:
            x: (b, T_mel, c)
        """
        x = self.downsampling_proj(x)
        input_lengths = torch.div(input_lengths, self.frame_rate_reduction_factor, rounding_mode="floor")
        x, enc_mask = self.encoder(x, seq_lens=input_lengths)
        x = self.out_proj(x) * enc_mask
        return x, input_lengths
