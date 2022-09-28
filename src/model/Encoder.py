import torch
from torch import nn
from torch.nn import functional as F

from src.model.layers import FFN, ConvNorm, ConvReluNorm, LayerNorm, MultiHeadAttention
from src.utilities.functions import get_mask_from_len


class Encoder_T2(nn.Module):
    """Encoder module:

    - Three 1-d convolution banks
    - Bidirectional LSTM
    """

    def __init__(self, hparams):
        super().__init__()

        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.state_per_phone = hparams.state_per_phone

        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(
                    hparams.encoder_embedding_dim,
                    hparams.encoder_embedding_dim,
                    kernel_size=hparams.encoder_kernel_size,
                    stride=1,
                    padding=int((hparams.encoder_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="relu",
                ),
                nn.BatchNorm1d(hparams.encoder_embedding_dim),
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(
            hparams.encoder_embedding_dim,
            int(hparams.encoder_embedding_dim / 2) * hparams.state_per_phone,
            1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x, input_lengths):
        r"""
        Takes embeddings as inputs and returns encoder representation of them
        Args:
            x (torch.float) : input shape (32, 512, 139)
            input_lengths (torch.int) : (32)

        Returns:
            outputs (torch.float):
                shape (batch, text_len * phone_per_state, encoder_embedding_dim)
            input_lengths (torch.float): shape (batch)
        """

        batch_size = x.shape[0]
        t_len = x.shape[2]

        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths_np = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths_np, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)  # We do not use the hidden or cell states

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        outputs = outputs.reshape(batch_size, t_len * self.state_per_phone, self.encoder_embedding_dim)
        input_lengths = input_lengths * self.state_per_phone

        return outputs, input_lengths  # (32, 139, 519)


class Encoder(nn.Module):
    """Glow TTS encoder with custom modifications"""

    def __init__(self, hparams):
        super().__init__()

        # hparams.n_symbols,
        # hparams.encoder_embedding_dim,
        # hparams.encoder_hidden_channels,
        # hparams.encoder_filter_channels,
        # hparams.encoder_filter_channels_dp,
        # hparams.encoder_n_heads,
        # hparams.encoder_n_layers,
        # hparams.encoder_kernel_size,
        # hparams.encoder_p_dropout,
        # hparams.encoder_window_size,
        # hparams.encoder_block_length,
        # hparams.encoder_prenet,

        self.prenet = hparams.encoder_prenet
        self.state_per_phone = hparams.state_per_phone
        self.encoder_embedding_dim = hparams.encoder_embedding_dim

        if hparams.encoder_prenet:
            self.pre = ConvReluNorm(
                hparams.encoder_hidden_channels,
                hparams.encoder_hidden_channels,
                hparams.encoder_hidden_channels,
                kernel_size=5,
                n_layers=3,
                p_dropout=0.5,
            )

        self.encoder = TransformerEncoder(
            hparams.encoder_hidden_channels,
            hparams.encoder_filter_channels,
            hparams.encoder_n_heads,
            hparams.encoder_n_layers,
            hparams.encoder_kernel_size,
            hparams.encoder_p_dropout,
            window_size=hparams.encoder_window_size,
            block_length=hparams.encoder_block_length,
        )

        self.proj = nn.Conv1d(
            hparams.encoder_hidden_channels, hparams.encoder_embedding_dim * hparams.state_per_phone, 1
        )

    def forward(self, x, x_lengths):
        """
        Take text embeddings and text lengths as inputs and returns encoder representation of them

        Args:
            x (torch.FloatTensor): input shape (batch, text_len, hidden_channels)
            x_lengths (torch.IntTensor): shape (batch)

        Returns:
            outputs: encoder representation of text embeddings
            x_lengths: text lengths
        """
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(get_mask_from_len(x_lengths, x.size(2), device=x.device), 1).to(x.dtype)

        if self.prenet:
            x = self.pre(x, x_mask)
        x = self.encoder(x, x_mask)

        outputs = self.proj(x) * x_mask

        outputs = outputs.reshape(x.shape[0], self.encoder_embedding_dim, x.shape[2] * self.state_per_phone)
        x_lengths = x_lengths * self.state_per_phone
        return outputs.transpose(1, 2), x_lengths


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size=1,
        p_dropout=0.0,
        window_size=None,
        block_length=None,
        **kwargs
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.block_length = block_length

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for i in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    window_size=window_size,
                    p_dropout=p_dropout,
                    block_length=block_length,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout)
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        for i in range(self.n_layers):
            x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x
