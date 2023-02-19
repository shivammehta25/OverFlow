import pytest
import torch

from src.model.OverFlow import OverFlow


def test_parse_batch(hparams, dummy_data):
    neural_hmm = OverFlow(hparams)
    parsed_batch = neural_hmm.parse_batch(dummy_data)
    text_padded, input_lengths, mel_padded, motion_padded, mel_lengths = parsed_batch[0]
    mel_padded, _ = parsed_batch[1]
    assert text_padded.shape[1] == max(input_lengths).item()
    assert mel_padded.shape[2] == torch.max(mel_lengths).item()
    assert motion_padded.shape[2] == torch.max(mel_lengths).item()


def test_forward(hparams, dummy_data, test_batch_size):
    neural_hmm = OverFlow(hparams)
    log_probs = neural_hmm.forward(dummy_data)
    assert log_probs.shape == (test_batch_size,)


@pytest.mark.parametrize("send_len", [True, False])
def test_sample(hparams, dummy_data_uncollated, send_len):
    neural_hmm = OverFlow(hparams)
    text = dummy_data_uncollated[0][0]
    (
        mel_output,
        motion_output,
        states_travelled,
        input_parameters,
        output_parameters,
    ) = (
        neural_hmm.sample(text, torch.tensor(len(text))) if send_len else neural_hmm.sample(text)
    )
    assert mel_output.shape[2] == hparams.n_mel_channels
    assert motion_output.shape[2] == hparams.n_motion_joints
    assert input_parameters[0][0].shape[-1] == (hparams.n_mel_channels + hparams.n_motion_joints)
    assert output_parameters[0][0].shape[-1] == (hparams.n_mel_channels + hparams.n_motion_joints)
    assert output_parameters[0][1].shape[-1] == (hparams.n_mel_channels + hparams.n_motion_joints)
