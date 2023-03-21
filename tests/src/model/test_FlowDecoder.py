"""Since it is taken from Glow-TTS, I am not implementing test for individual elements."""
import pytest
import torch

from src.model.Decoder import FlowSpecDecoder, MotionDecoder
from src.utilities.functions import get_mask_from_len
from tests.test_utilities import reset_all_weights


@pytest.mark.parametrize(
    "flow_hidden_channels, dilation_rate, n_block_dec, n_block_layers, n_split, n_sqz",
    [(192, 1, 4, 2, 2, 2), (256, 2, 12, 4, 8, 1)],
)
def test_FlowDecoder(
    hparams,
    dummy_data,
    test_batch_size,
    flow_hidden_channels,
    dilation_rate,
    n_block_dec,
    n_block_layers,
    n_split,
    n_sqz,
):
    """Test the FlowDecoder class."""
    hparams.flow_hidden_channels = flow_hidden_channels
    hparams.dilation_rate = dilation_rate
    hparams.n_blocks_dec = n_block_dec
    hparams.n_block_layers = n_block_layers
    hparams.n_split = n_split
    hparams.n_sqz = n_sqz

    hparams.p_dropout_dec = 0.0  # Turn off dropout to check invertibility

    decoder = FlowSpecDecoder(hparams, hparams.n_mel_channels, hparams.p_dropout_dec)

    reset_all_weights(decoder)

    _, _, mel_padded, motion_padded, output_lengths = dummy_data
    z, z_lengths, logdet = decoder(mel_padded, output_lengths)
    assert logdet.shape[0] == test_batch_size
    assert z.shape[1] == hparams.n_mel_channels
    assert (z.shape[2] == mel_padded.shape[2]) or (z.shape[2] == (mel_padded.shape[2] - 1)), "Output format matches"

    mel_, _, logdet_ = decoder(z, z_lengths, reverse=True)
    len_mask = get_mask_from_len(z_lengths, device=z_lengths.device).unsqueeze(1)
    mel_padded = mel_padded[:, :, : z.shape[2]] * len_mask
    assert torch.isclose(mel_padded, mel_, atol=1e-5).all(), "Invertible"
    assert logdet_ is None


@pytest.mark.parametrize("motion_decoder_type", ["conformer", "transformer", "rnn", "gradtts"])
def test_MotionDecoder(hparams, dummy_data, test_batch_size, motion_decoder_type):
    """Test the MotionDecoder class."""
    decoder_motion = MotionDecoder(hparams, motion_decoder_type)
    _, _, mel_padded, motion_padded, output_lengths = dummy_data
    mel_padded, output_lengths, _ = FlowSpecDecoder.preprocess(
        mel_padded, output_lengths, output_lengths.max(), hparams.n_sqz
    )
    decoder_output = decoder_motion(mel_padded, output_lengths, motion_padded)
    motion_output = decoder_output["motions"]
    motion_len = decoder_output["motion_lengths"]
    assert motion_output.shape[-1] == hparams.n_motion_joints
    assert motion_output.shape[0] == test_batch_size
    assert motion_len.shape[0] == test_batch_size
