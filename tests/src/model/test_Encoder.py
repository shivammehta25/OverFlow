import pytest
import torch

from src.model.Encoder import Encoder, Tacotron2Encoder


@pytest.mark.parametrize("state_per_phone", [1, 2, 3])
def test_Encoder_forward(
    hparams,
    dummy_data,
    state_per_phone,
):
    hparams.encoder_type = "conv"
    hparams.encoder_params["conv"]["state_per_phone"] = state_per_phone
    emb_dim = hparams.encoder_params[hparams.encoder_type]["hidden_channels"]
    encoder = Tacotron2Encoder(hparams)
    text_padded, input_lengths, _, _, _ = dummy_data
    emb = torch.nn.Embedding(hparams.n_symbols, emb_dim)(text_padded)
    encoder_outputs, text_lengths_post_enc = encoder(emb.transpose(1, 2), input_lengths)
    assert encoder_outputs.shape[1] == (emb.shape[1] * state_per_phone)
    assert (text_lengths_post_enc == (input_lengths * state_per_phone)).all()


@pytest.mark.parametrize("rel_attention", [True, False])
def test_FPEncoder(hparams, dummy_data, rel_attention):
    hparams.encoder_type = "transformer"
    hparams.encoder_params["transformer"]["rel_attention"] = rel_attention
    encoder = Encoder(hparams)
    text_padded, input_lengths, _, _, _ = dummy_data
    emb_dim = hparams.encoder_params[hparams.encoder_type]["hidden_channels"]
    emb = torch.nn.Embedding(hparams.n_symbols, emb_dim)(text_padded)
    encoder_outputs, text_lengths_post_enc = encoder(emb.transpose(1, 2), input_lengths)
    assert encoder_outputs.shape[1] == (emb.shape[1])
    assert (text_lengths_post_enc == (input_lengths)).all()
