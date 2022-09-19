"""Since it is taken from Glow-TTS, I am not implementing test for individual elements."""
from src.model.FlowDecoder import FlowSpecDecoder


def test_FlowDecoder(hparams, dummy_data, test_batch_size):
    """Test the FlowDecoder class."""
    decoder = FlowSpecDecoder(hparams)
    _, _, mel_padded, _, output_lengths = dummy_data
    z, z_length, logdet = decoder(mel_padded, output_lengths)
    assert logdet.shape[0] == test_batch_size
    assert z.shape[1] == hparams.n_mel_channels
    assert (z.shape[2] == mel_padded.shape[2]) or z.shape == mel_padded.shape[2]

    mel_, output_lengths_ = decoder(z, z_length, reverse=True)
    # TODO: check the invertibility of the decoder
