# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from math import log as ln

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(*args, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x, mask):
        return self.conv(x * mask)


class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, noise_level):
        """
        Arguments:
          x:
              (shape: [N,C,T], dtype: float32)
          noise_level:
              (shape: [N], dtype: float32)

        Returns:
          noise_level:
              (shape: [N,C,T], dtype: float32)
        """
        return x + self._build_encoding(noise_level)[:, :, None]

    def _build_encoding(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-ln(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FiLM(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.encoding = PositionalEncoding(input_size)
        self.input_conv = nn.Conv1d(input_size, input_size, 3, padding=1)
        self.output_conv = nn.Conv1d(input_size, output_size * 2, 3, padding=1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_conv.weight)
        nn.init.xavier_uniform_(self.output_conv.weight)
        nn.init.zeros_(self.input_conv.bias)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, x, mask, noise_scale):
        x = self.input_conv(x * mask)
        x = F.leaky_relu(x, 0.2)
        x = self.encoding(x, noise_scale)
        shift, scale = torch.chunk(self.output_conv(x * mask), 2, dim=1)
        return shift * mask, scale * mask


class UBlock(nn.Module):
    def __init__(self, input_size, hidden_size, factor, dilation):
        super().__init__()
        assert isinstance(dilation, (list, tuple))
        assert len(dilation) == 4

        self.factor = factor
        self.block1 = Conv1d(input_size, hidden_size, 1)
        self.block2 = nn.ModuleList(
            [
                Conv1d(input_size, hidden_size, 3, dilation=dilation[0], padding=dilation[0]),
                Conv1d(hidden_size, hidden_size, 3, dilation=dilation[1], padding=dilation[1]),
            ]
        )
        self.block3 = nn.ModuleList(
            [
                Conv1d(hidden_size, hidden_size, 3, dilation=dilation[2], padding=dilation[2]),
                Conv1d(hidden_size, hidden_size, 3, dilation=dilation[3], padding=dilation[3]),
            ]
        )

    def forward(self, x, mask, film_shift, film_scale):
        block1 = F.interpolate(x, size=x.shape[-1] * self.factor)
        block1 = self.block1(block1, mask)

        block2 = F.leaky_relu(x, 0.2)
        block2 = F.interpolate(block2, size=x.shape[-1] * self.factor)
        block2 = self.block2[0](block2, mask)
        block2 = film_shift + film_scale * block2
        block2 = F.leaky_relu(block2, 0.2)
        block2 = self.block2[1](block2, mask)

        x = block1 + block2

        block3 = film_shift + film_scale * x
        block3 = F.leaky_relu(block3, 0.2)
        block3 = self.block3[0](block3, mask)
        block3 = film_shift + film_scale * block3
        block3 = F.leaky_relu(block3, 0.2)
        block3 = self.block3[1](block3, mask)

        x = x + block3
        return x * mask


class DBlock(nn.Module):
    def __init__(self, input_size, hidden_size, factor):
        super().__init__()
        self.factor = factor
        self.residual_dense = Conv1d(input_size, hidden_size, 1)
        self.conv = nn.ModuleList(
            [
                Conv1d(input_size, hidden_size, 3, dilation=1, padding=1),
                Conv1d(hidden_size, hidden_size, 3, dilation=2, padding=2),
                Conv1d(hidden_size, hidden_size, 3, dilation=4, padding=4),
            ]
        )

    def forward(self, x, mask):
        size = x.shape[-1] // self.factor

        residual = self.residual_dense(x, mask)
        residual = F.interpolate(residual, size=size) * mask

        x = F.interpolate(x, size=size)
        for layer in self.conv:
            x = F.leaky_relu(x, 0.2)
            x = layer(x, mask)

        return (x + residual) * mask


class WaveGrad(nn.Module):
    def __init__(self, motion_in_channels=45, latent_in_channels=80):
        super().__init__()
        self.downsample = nn.ModuleList(
            [
                Conv1d(motion_in_channels, 64, 5, padding=2),
                DBlock(64, 128, 1),
                DBlock(128, 128, 1),
                DBlock(128, 256, 1),
                DBlock(256, 512, 1),
            ]
        )
        self.film = nn.ModuleList(
            [
                FiLM(64, 128),
                FiLM(128, 128),
                FiLM(128, 256),
                FiLM(256, 512),
                FiLM(512, 512),
            ]
        )
        self.upsample = nn.ModuleList(
            [
                UBlock(768, 512, 1, [1, 2, 1, 2]),
                UBlock(512, 512, 1, [1, 2, 1, 2]),
                UBlock(512, 256, 1, [1, 2, 4, 8]),
                UBlock(256, 128, 1, [1, 2, 4, 8]),
                UBlock(128, 128, 1, [1, 2, 4, 8]),
            ]
        )
        self.first_conv = Conv1d(latent_in_channels, 768, 3, padding=1)
        self.last_conv = Conv1d(128, motion_in_channels, 3, padding=1)

    def forward(self, motion, mask, z, timestep):
        downsampled = []
        x = motion
        for film, layer in zip(self.film, self.downsample):
            x = layer(x, mask)
            downsampled.append(film(x, mask, timestep))

        x = self.first_conv(z, mask)
        for layer, (film_shift, film_scale) in zip(self.upsample, reversed(downsampled)):
            x = layer(x, mask, film_shift, film_scale)
        x = self.last_conv(x, mask)
        return x
