from math import sqrt

import torch
from torch import nn

from src.model.Encoder import Encoder
from src.model.FlowDecoder import FlowSpecDecoder
from src.model.HMM import HMM


class NeuralHMM(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.embedding = nn.Embedding(hparams.n_symbols, hparams.symbols_embedding_dim)
        if hparams.warm_start or hparams.checkpoint_path:
            # If warm start or resuming training do not re-initialize embeddings
            std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
            val = sqrt(3.0) * std  # uniform bounds for std
            self.embedding.weight.data.uniform_(-val, val)

        # Data Properties
        self.normaliser = hparams.normaliser

        self.encoder = Encoder(hparams)
        self.hmm = HMM(hparams)
        self.decoder = FlowSpecDecoder(hparams)
        self.logger = hparams.logger

    def parse_batch(self, batch):
        """
        Takes batch as an input and returns all the tensor to GPU
        Args:
            batch:

        Returns:

        """
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch
        text_padded = text_padded.long()
        input_lengths = input_lengths.long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = mel_padded.float()
        gate_padded = gate_padded.float()
        output_lengths = output_lengths.long()

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths),
            (mel_padded, gate_padded),
        )

    def forward(self, inputs):
        text_inputs, text_lengths, mels, max_len, mel_lengths = inputs
        text_lengths, mel_lengths = text_lengths.data, mel_lengths.data
        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        encoder_outputs, text_lengths = self.encoder(embedded_inputs, text_lengths)

        z, z_lengths, logdet = self.decoder(mels, mel_lengths)

        log_probs = self.hmm(encoder_outputs, text_lengths, z, z_lengths)
        return log_probs + logdet

    @torch.inference_mode()
    def sample(self, text_inputs, text_lengths=None):
        r"""
        Sampling mel spectrogram based on text inputs
        Args:
            text_inputs (int tensor) : shape ([x]) where x is the phoneme input
            text_lengths (int tensor, Optional):  single value scalar with length of input (x)

        Returns:
            mel_outputs (list): list of len of the output of mel spectrogram
                    each containing n_mel_channels channels
                shape: (len, n_mel_channels)
            states_travelled (list): list of phoneme travelled at each time step t
                shape: (len)
        """
        if text_inputs.ndim > 1:
            text_inputs = text_inputs.squeeze(0)

        if text_lengths is None:
            text_lengths = text_inputs.new_tensor(text_inputs.shape[0])

        text_inputs, text_lengths = text_inputs.unsqueeze(0), text_lengths.unsqueeze(0)
        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        encoder_outputs, text_lengths = self.encoder(embedded_inputs, text_lengths)

        (
            mel_latent,
            states_travelled,
            input_parameters,
            output_parameters,
        ) = self.hmm.sample(encoder_outputs)

        mel_output, mel_lengths, _ = self.decoder(
            mel_latent.unsqueeze(0).transpose(1, 2), text_lengths.new_tensor([mel_latent.shape[0]]), reverse=True
        )

        if self.normaliser:
            mel_output = self.normaliser.inverse_normalise(mel_output)

        return mel_output.transpose(1, 2), states_travelled, input_parameters, output_parameters

    def store_inverse(self):
        self.decoder.store_inverse()
