import torch
from torch import nn

from src.model.Encoder import Encoder
from src.model.FlowDecoder import FlowSpecDecoder
from src.model.HMM import HMM


class OverFlow(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.embedding = nn.Embedding(
            hparams.n_symbols, hparams.encoder_params[hparams.encoder_type]["hidden_channels"]
        )

        # Data Properties
        self.normaliser = hparams.normaliser

        if hparams.num_speakers > 1:
            self.speaker_embedding = nn.Embedding(
                hparams.num_speakers, hparams.encoder_params[hparams.encoder_type]["hidden_channels"]
            )

        self.encoder = Encoder(hparams)
        # self.encoder = Tacotron2Encoder(hparams)
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
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths, speaker_id = batch
        text_padded = text_padded.long()
        input_lengths = input_lengths.long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = mel_padded.float()
        gate_padded = gate_padded.float()
        output_lengths = output_lengths.long()

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths, speaker_id),
            (mel_padded, gate_padded),
        )

    def forward(self, inputs):
        text_inputs, text_lengths, mels, max_len, mel_lengths, speaker_id = inputs
        text_lengths, mel_lengths = text_lengths.data, mel_lengths.data
        if self.hparams.num_speakers > 1:
            speaker_embeddings = self.speaker_embedding(speaker_id).unsqueeze(1)
        else:
            speaker_embeddings = 0
        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        encoder_outputs, text_lengths = self.encoder(embedded_inputs + speaker_embeddings, text_lengths)
        z, z_lengths, logdet = self.decoder(
            mels, mel_lengths, g=speaker_embeddings if self.hparams.num_speakers > 1 else None
        )
        log_probs = self.hmm(encoder_outputs, text_lengths, z, z_lengths)
        loss = (log_probs + logdet) / (text_lengths.sum() + mel_lengths.sum())
        return loss

    @torch.inference_mode()
    def sample(self, text_inputs, text_lengths=None, speaker_id=None, sampling_temp=1.0):
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

        if self.hparams.num_speakers > 1:
            if speaker_id is None:
                speaker_id = text_inputs.new_zeros(1, 1)
            speaker_embeddings = self.speaker_embedding(speaker_id).unsqueeze(1)
        else:
            speaker_id = 0

        encoder_outputs, text_lengths = self.encoder(embedded_inputs + speaker_id, text_lengths)

        (
            mel_latent,
            states_travelled,
            input_parameters,
            output_parameters,
        ) = self.hmm.sample(encoder_outputs, sampling_temp=sampling_temp)

        mel_output, mel_lengths, _ = self.decoder(
            mel_latent.unsqueeze(0).transpose(1, 2),
            text_lengths.new_tensor([mel_latent.shape[0]]),
            reverse=True,
            g=speaker_embeddings if self.hparams.num_speakers > 1 else None,
        )

        if self.normaliser:
            mel_output = self.normaliser.inverse_normalise(mel_output)

        return mel_output.transpose(1, 2), states_travelled, input_parameters, output_parameters

    def store_inverse(self):
        self.decoder.store_inverse()
