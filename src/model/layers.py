"""layers.py.

Layer modules used in the model design
"""
import numpy as np
import torch
import torch.nn as nn
from librosa.filters import mel as librosa_mel_fn

from src.utilities.audio import dynamic_range_compression, dynamic_range_decompression
from src.utilities.stft import STFT


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
        super().__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class LinearReluInitNorm(nn.Module):
    r"""
    Contains a Linear Layer with Relu activation and a dropout
    Args:
        inp (tensor): size of input to the linear layer
        out (tensor): size of output from the linear layer
        init (bool): model initialisation with xavier initialisation
            default: False
        w_init_gain (str): gain based on the activation function used
            default: relu
    """

    def __init__(self, inp, out, init=True, w_init_gain="relu", bias=True):
        super().__init__()

        self.w_init_gain = w_init_gain
        self.linear = nn.Sequential(nn.Linear(inp, out, bias=bias), nn.ReLU())

        if init:
            self.linear.apply(self._weights_init)

    def _weights_init(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight.data, gain=torch.nn.init.calculate_gain(self.w_init_gain))

    def forward(self, x):
        return self.linear(x)


class ConvNorm(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):
        super().__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        torch.nn.init.xavier_uniform_(self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class TacotronSTFT(torch.nn.Module):
    """Short Time Fourier Transformation."""

    def __init__(
        self,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        sampling_rate=22050,
        mel_fmin=0.0,
        mel_fmax=8000.0,
    ):
        super().__init__()
        self.n_mel_channels = n_mel_channels  # 80
        self.sampling_rate = sampling_rate  # 22050
        self.stft_fn = STFT(filter_length, hop_length, win_length)  # default values
        # """This produces a linear transformation matrix to project FFT bins onto Mel-frequency bins."""
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax
        )  # all default values

        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("inv_mel_basis", torch.linalg.pinv(self.mel_basis))

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert torch.min(y.data) >= -1
        assert torch.max(y.data) <= 1

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output

    def mel_to_linear(self, mel_spec):
        """Project a melspectrogram to a full scale spectrogram.

        Args:
            mel_spec (Float[torch.Tensor, "B n_mel_channel T_mel"]): Melspectrogram.

        Returns:
            Float[torch.Tensor, "B n_spec_channel T_mel"]: Full scale spectrogram.
        """
        return torch.max(mel_spec.new_tensor(1e-10), torch.matmul(self.inv_mel_basis, mel_spec))

    def griffin_lim(self, mel_spec, n_iters=15):
        """Applies Griffin-Lim's raw to reconstruct phase.

        Args:
            audio (Union[Float[torch.Tensor, "B n_mel_channels T_mel"], str]):
                audio filename or torch tensor of the mel spectrogram
            mel (bool, optional):
                True when input is a mel spectrogram otherwise its a linear spectrogram. Defaults to True.

        Returns:
            waveform: reconstructed waveform
                -shape (1, T_wav)
            sampling_rate: sampling rate of the audio file
        """
        assert mel_spec.ndim == 3, "input shape must be batch, n_mel_channels, T_mel"
        mel_spec = mel_spec.cpu()
        mel_spec = self.spectral_de_normalize(mel_spec)
        magnitudes = self.mel_to_linear(mel_spec)

        angles = np.angle(np.exp(2j * np.pi * np.random.rand(*magnitudes.size())))
        angles = angles.astype(np.float32)
        angles = torch.autograd.Variable(torch.from_numpy(angles))
        signal = self.stft_fn.inverse(magnitudes, angles).squeeze(1)

        for _ in range(n_iters):
            _, angles = self.stft_fn.transform(signal)
            signal = self.stft_fn.inverse(magnitudes, angles).squeeze(1)
        return signal.cpu().numpy(), self.sampling_rate
