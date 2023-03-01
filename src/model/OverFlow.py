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
        self.n_motion_joints = hparams.n_motion_joints
        self.embedding = nn.Embedding(
            hparams.n_symbols, hparams.encoder_params[hparams.encoder_type]["hidden_channels"]
        )

        # Data Properties
        self.mel_normaliser = hparams.mel_normaliser
        self.motion_normaliser = hparams.motion_normaliser

        self.encoder = Encoder(hparams)
        # self.encoder = Tacotron2Encoder(hparams)
        self.hmm = HMM(hparams)
        self.decoder_mel = FlowSpecDecoder(hparams, hparams.n_mel_channels, hparams.p_dropout_dec_mel)
        self.decoder_motion = FlowSpecDecoder(hparams, hparams.n_motion_joints, hparams.p_dropout_dec_motion)
        self.logger = hparams.logger

    def parse_batch(self, batch):
        """
        Takes batch as an input and returns all the tensor to GPU
        Args:
            batch:

        Returns:

        """
        text_padded, input_lengths, mel_padded, motion_padded, output_lengths = batch
        text_padded = text_padded.long()
        input_lengths = input_lengths.long()
        mel_padded = mel_padded.float()
        motion_padded = motion_padded.float()
        output_lengths = output_lengths.long()

        return (
            (text_padded, input_lengths, mel_padded, motion_padded, output_lengths),
            (mel_padded, motion_padded),
        )

    def forward(self, inputs):
        text_inputs, text_lengths, mels, motions, mel_lengths = inputs
        text_lengths, mel_lengths = text_lengths.data, mel_lengths.data
        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        encoder_outputs, text_lengths = self.encoder(embedded_inputs, text_lengths)
        z_mel, z_lengths, logdet_mel = self.decoder_mel(mels, mel_lengths)
        z_motion, z_lengths, logdet_motion = self.decoder_motion(motions, mel_lengths)
        z = torch.concat([z_mel, z_motion], dim=1)
        logdet = logdet_mel + logdet_motion
        log_probs = self.hmm(encoder_outputs, text_lengths, z, z_lengths)
        loss = (log_probs + logdet) / (text_lengths.sum() + mel_lengths.sum())
        return loss

    @torch.inference_mode()
    def sample(self, text_inputs, text_lengths=None, sampling_temp=1.0):
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
            mel_motion_latent,
            states_travelled,
            input_parameters,
            output_parameters,
        ) = self.hmm.sample(encoder_outputs, sampling_temp=sampling_temp)

        mel_latent, motion_latent = (
            mel_motion_latent[:, : self.n_mel_channels],
            mel_motion_latent[:, self.n_mel_channels :],
        )

        mel_output, mel_lengths, _ = self.decoder_mel(
            mel_latent.unsqueeze(0).transpose(1, 2), text_lengths.new_tensor([mel_latent.shape[0]]), reverse=True
        )

        motion_output, _, _ = self.decoder_motion(
            motion_latent.unsqueeze(0).transpose(1, 2), text_lengths.new_tensor([motion_latent.shape[0]]), reverse=True
        )

        if self.mel_normaliser:
            mel_output = self.mel_normaliser.inverse_normalise(mel_output)
        if self.motion_normaliser:
            motion_output = self.motion_normaliser.inverse_normalise(motion_output)

        return (
            mel_output.transpose(1, 2),
            motion_output.transpose(1, 2),
            states_travelled,
            input_parameters,
            output_parameters,
        )

    def store_inverse(self):
        self.decoder_mel.store_inverse()
