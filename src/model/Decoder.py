"""
Glow-TTS Code from https://github.com/jaywalnut310/glow-tts
"""
from argparse import Namespace

import numpy as np
import torch
import torch.distributions as tdist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import src.model.DecoderComponents.flows as flows
from src.model.layers import ConvNorm, LinearNorm
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
        self.decoder_type = decoder_type
        self.frame_rate_reduction_factor = hparams.frame_rate_reduction_factor
        self.n_motion_joints = hparams.n_motion_joints
        self.in_proj = ConvNorm(
            hparams.n_mel_channels,
            hparams.motion_decoder_param[decoder_type]["hidden_channels"],
            kernel_size=5,
        )
        if decoder_type == "transformer":
            self.encoder = FFTransformer(**hparams.motion_decoder_param[decoder_type])
        elif decoder_type == "conformer":
            self.encoder = Conformer(**hparams.motion_decoder_param[decoder_type])
        elif decoder_type == "flow":
            self.base_dist = tdist.normal.Normal
            hparams_decoder = Namespace(**hparams.motion_decoder_param[decoder_type])
            self.latent_proj = LinearNorm(hparams_decoder.hidden_channels, hparams.n_motion_joints * 2)
            self.encoder = FlowSpecDecoder(hparams_decoder, hparams.n_motion_joints + 3)
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")
        self.out_proj = LinearNorm(
            hparams.motion_decoder_param[decoder_type]["hidden_channels"], hparams.n_motion_joints
        )

    def forward(self, x, input_lengths, motion_target=None, reverse=False, sampling_temp=1.0):
        """

        Args:
            x : (b, c, T_mel)
            input_lengths: (b)
            output: (b, n_joints, T_mel)

        Returns:
            x: (b, T_mel // frame_rate_reduction_factor, c)
            input_lengths: (b)  # reduced by frame_rate_reduction_factor
            motion: (b, T_mel // frame_rate_reduction_factor, c)
        """
        assert input_lengths is not None, "Placeholder for future use"
        if reverse is False:
            assert motion_target is not None
            motion_target = motion_target[:, :, :: self.frame_rate_reduction_factor].transpose(1, 2)
        else:
            motion_target = None

        x = x[:, :, :: self.frame_rate_reduction_factor]
        output_lengths = torch.round(
            torch.where(input_lengths % 4 == 0, input_lengths, input_lengths + 2) / self.frame_rate_reduction_factor
        ).int()
        x = self.in_proj(x)
        if self.decoder_type in ["transformer", "conformer"]:
            x, enc_mask = self.encoder(x, seq_lens=output_lengths)
            x = self.out_proj(x) * enc_mask
            if motion_target is not None:
                motion_target = motion_target * enc_mask

        elif self.decoder_type == "flow":
            random_element = torch.randn(x.shape[0], x.shape[-1], 3, device=x.device, dtype=x.dtype)
            x = self.latent_proj(rearrange(x, "b c t -> b t c"))
            x_m = rearrange(x[:, :, : self.n_motion_joints], "b t c -> b c t")
            x_s = torch.clamp_min(rearrange(F.softplus(x[:, :, self.n_motion_joints :]), "b t c -> b c t"), 1e-3)

            if reverse is False:
                motion_target = rearrange(torch.concat([motion_target, random_element], dim=-1), "b t c -> b c t")
                z, z_lengths, logdet = self.encoder(motion_target, output_lengths)
                z = z[:, : self.n_motion_joints]
                x_m, *_ = self.encoder.preprocess(x_m, z_lengths, z_lengths.max())
                x_s, *_ = self.encoder.preprocess(x_s, z_lengths, z_lengths.max())
                mask = get_mask_from_len(z_lengths, device=z_lengths.device, dtype=z_lengths.dtype)
                prob = self.base_dist(x_m, x_s).log_prob(z) * rearrange(mask, "b t -> b () t")
                prob = prob.sum([1, 2]) / (z_lengths.sum() * prob.shape[1])  # Averaging across time and channels
                loss = prob + logdet
            else:
                z = self.base_dist(x_m, x_s).sample() if sampling_temp > 0 else x_m
                z = torch.concat([z, rearrange(random_element, "b t c -> b c t")], dim=1)
                z, output_lengths, _ = self.encoder(z, output_lengths, reverse=True)
                x = rearrange(z[:, : self.n_motion_joints], "b c t -> b t c")
                loss = None

        return {"generated": x, "generated_lengths": output_lengths, "target": motion_target, "loss": loss}
