"""
Glow-TTS Code from https://github.com/jaywalnut310/glow-tts
"""
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

import src.model.DecoderComponents.flows as flows
from src.model.diffusion import Diffusion as GradTTSDiffusion
from src.model.diffusion import MyDiffusion
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
        x, x_lengths, x_max_length = self.preprocess(x, x_lengths, x_lengths.max(), self.n_sqz)

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

    @staticmethod
    def preprocess(y, y_lengths, y_max_length, n_sqz):
        if y_max_length is not None:
            y_max_length = torch.div(y_max_length, n_sqz, rounding_mode="floor") * n_sqz
            y = y[:, :, :y_max_length]
        y_lengths = torch.div(y_lengths, n_sqz, rounding_mode="floor") * n_sqz
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


class RNNDecoder(nn.Module):
    def __init__(self, hidden_channels, n_layer, dropout, bidirectional=True):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            num_layers=n_layer,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.hidden_channels = hidden_channels
        self.num_layers = n_layer
        self.dropout = dropout
        self.bidirectional = bidirectional

    def forward(self, x, seq_lens):
        x = rearrange(x, "b c t -> b t c")
        x = nn.utils.rnn.pack_padded_sequence(x, seq_lens.cpu(), enforce_sorted=False, batch_first=True)
        x, _ = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        x = rearrange(x, "t b (x c) -> b t c x", c=self.hidden_channels).mean(-1)
        return x, get_mask_from_len(seq_lens, device=seq_lens.device, dtype=seq_lens.dtype).unsqueeze(-1)


class MotionDecoder(nn.Module):
    _FORWARD_DECODERS = ["transformer", "conformer", "rnn"]
    _DIFFUSION_DECODERS = ["gradtts", "mydiffusion"]

    def __init__(self, hparams, decoder_type="transformer"):
        super().__init__()
        self.decoder_type = decoder_type
        self.n_sqz = hparams.n_sqz
        self.n_motion_joints = hparams.n_motion_joints
        self.in_proj = LinearNorm(hparams.n_mel_channels, hparams.motion_decoder_param[decoder_type]["hidden_channels"])
        self.out_proj = LinearNorm(
            hparams.motion_decoder_param[decoder_type]["hidden_channels"], hparams.n_motion_joints
        )
        self.motion_loss = nn.MSELoss()
        if decoder_type == "transformer":
            self.encoder = FFTransformer(**hparams.motion_decoder_param[decoder_type])
        elif decoder_type == "conformer":
            self.encoder = Conformer(**hparams.motion_decoder_param[decoder_type])
        elif decoder_type == "rnn":
            self.encoder = RNNDecoder(**hparams.motion_decoder_param[decoder_type])
        elif decoder_type == "gradtts":
            self.encoder = GradTTSDiffusion(
                n_feats=hparams.n_motion_joints, **hparams.motion_decoder_param[decoder_type]
            )
            self.motion_loss = None  # The loss will be returned by the decoder
            self.in_proj = nn.Conv1d(hparams.n_mel_channels, hparams.n_motion_joints, 1)
            self.out_proj = None
        elif decoder_type == "mydiffusion":
            decoder_params = deepcopy(hparams.motion_decoder_param[decoder_type])
            del decoder_params["hidden_channels"]
            self.encoder = MyDiffusion(hparams.n_motion_joints, hparams.n_mel_channels, **decoder_params)
            self.in_proj = nn.Identity()
            self.out_proj = None
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")

    def forward_forward(self, x, input_lengths):
        """

        Args:
            x : (b, c, T_mel)
            input_lengths: (b)

        Returns:
            x: (b, T_mel, c)
        """
        x = self.in_proj(x.transpose(1, 2)).transpose(1, 2)
        x, enc_mask = self.encoder(x, seq_lens=input_lengths)
        x = self.out_proj(x) * enc_mask
        output = {
            "motions": x,
            "motion_lengths": input_lengths,
            "enc_mask": enc_mask,
        }
        return output

    @staticmethod
    def _validate_inputs(target_motions, reverse):
        if reverse:
            assert target_motions is None
        else:
            assert target_motions is not None

    def forward(self, x, input_lengths, target_motions=None, reverse=False):
        self._validate_inputs(target_motions, reverse)

        if target_motions is not None:
            target_motions, _, _ = FlowSpecDecoder.preprocess(
                target_motions, input_lengths, input_lengths.max(), self.n_sqz
            )

        if self.decoder_type in self._FORWARD_DECODERS:
            if reverse:
                return self.forward_forward(x, input_lengths)
            else:
                decoder_output = self.forward_forward(x, input_lengths)
                # Assure that the target motions are padded to the same length as the input motions
                decoder_output["loss"] = self.motion_loss(decoder_output["motions"], target_motions.transpose(1, 2))
                return decoder_output

        elif self.decoder_type in self._DIFFUSION_DECODERS:
            inputs_mask = get_mask_from_len(
                input_lengths, input_lengths.max(), device=input_lengths.device, dtype=input_lengths.dtype
            ).unsqueeze(1)

            x = self.in_proj(x * inputs_mask)
            if reverse:
                # Reverse diffusion
                output = self.encoder(x, inputs_mask)
                return {
                    "motions": rearrange(output, "b c t -> b t c"),
                    "motion_lengths": input_lengths,
                    "enc_mask": inputs_mask.squeeze(1).unsqueeze(-1),
                }
            else:
                # loss computation
                loss, xt = self.encoder.compute_loss(target_motions, inputs_mask, x)
                return {
                    "loss": loss,
                    "motions": rearrange(xt, "b c t -> b t c"),  # Noisy image at timestep t
                    "motion_lengths": input_lengths,
                }
        else:
            raise ValueError(f"Unknown decoder type: {self.decoder_type}")
