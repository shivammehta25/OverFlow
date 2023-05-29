from typing import Tuple

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from transformers import T5Config, T5EncoderModel

from src.model.layers import ConvNorm
from src.model.transformer import FFTransformer
from src.utilities.functions import get_mask_from_len


class Tacotron2Encoder(nn.Module):
    """Encoder module:

    - Three 1-d convolution banks
    - Bidirectional LSTM
    """

    def __init__(self, hparams, name="conv"):
        super().__init__()
        encoder_params = hparams.encoder_params[name]
        self.encoder_embedding_dim = encoder_params["hidden_channels"]
        self.state_per_phone = encoder_params["state_per_phone"]

        convolutions = []
        for _ in range(encoder_params["n_convolutions"]):
            conv_layer = nn.Sequential(
                ConvNorm(
                    encoder_params["hidden_channels"],
                    encoder_params["hidden_channels"],
                    kernel_size=encoder_params["kernel_size"],
                    stride=1,
                    padding=int((encoder_params["kernel_size"] - 1) / 2),
                    dilation=1,
                    w_init_gain="relu",
                ),
                nn.BatchNorm1d(encoder_params["hidden_channels"]),
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(
            encoder_params["hidden_channels"],
            int(encoder_params["hidden_channels"] / 2) * encoder_params["state_per_phone"],
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


class TransformerEncoder(nn.Module):
    def __init__(self, hparams, name="transformer"):
        super().__init__()

        encoder_params = hparams.encoder_params[name]

        self.encoder = FFTransformer(**encoder_params)

    def forward(self, x, input_lengths):
        x, enc_mask = self.encoder(x, seq_lens=input_lengths)
        return x, input_lengths


class HFT5Encoder(nn.Module):
    def __init__(self, hparams, name="hfT5") -> None:
        super().__init__()

        config = T5Config()
        encoder_params = hparams.encoder_params[name]
        config.is_encoder_decoder = False
        config.d_model = encoder_params["hidden_channels"]
        config.num_layers = encoder_params["n_layer"]
        config.vocab_size = 1  # will not be used
        config.num_heads = encoder_params["d_head"]
        config.d_ff = encoder_params["d_inner"]
        config.num_heads = encoder_params["n_head"]
        config.dropout_rate = encoder_params["dropout"]
        config.feed_forward_proj = encoder_params["feed_forward_proj"]
        print(config)
        self.encoder = T5EncoderModel(config)
        self.encoder.shared.weight.requires_grad = False

    def forward(self, x, input_lengths):
        attention_mask = get_mask_from_len(input_lengths, device=x.device, dtype=x.dtype)
        x = rearrange(x, "b c t -> b t c")
        t5output = self.encoder(inputs_embeds=x, attention_mask=attention_mask)
        return t5output.last_hidden_state, input_lengths


class Encoder(nn.Module):
    """Wrapper for encoders"""

    def __init__(self, hparams) -> None:
        super().__init__()
        if hparams.encoder_type == "conv":
            self.encoder = Tacotron2Encoder(hparams)
        elif hparams.encoder_type == "transformer":
            self.encoder = TransformerEncoder(hparams)
        elif hparams.encoder_type == "hfT5":
            self.encoder = HFT5Encoder(hparams)
        else:
            raise ValueError(f"Unknown encoder type: {hparams.encoder_type}")

    def forward(self, x: torch.FloatTensor, x_len: torch.LongTensor) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        return self.encoder(x, x_len)

    def inference(self, *args, **kwargs):
        return self.encoder.inference(*args, **kwargs)
