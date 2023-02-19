r"""
The file creates a pickle file where the values needed for loading of dataset is stored and the model can load it
when needed.

Parameters from hparam.py will be used
"""
import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.hparams import create_hparams
from src.utilities.data import TextMelLoader, TextMelMotionCollate


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return x


def parse_batch(batch):
    r"""
    Takes batch as an input and returns all the tensor to GPU
    Args:
        batch: batch of data
    """
    text_padded, input_lengths, mel_padded, motion_padded, output_lengths = batch
    text_padded = to_gpu(text_padded).long()
    input_lengths = to_gpu(input_lengths).long()
    mel_padded = to_gpu(mel_padded).float()
    motion_padded = to_gpu(motion_padded).float()
    output_lengths = to_gpu(output_lengths).long()

    return (
        (text_padded, input_lengths, mel_padded, motion_padded, output_lengths),
        (mel_padded, motion_padded),
    )


def get_data_parameters_for_flat_start(train_loader, hparams):
    r"""

    Args:
        dataloader (torch.util.data.DataLoader) : datalaoder containing text, mel
        hparams (hparam.py): hyperparemters

    Returns:
        mean (single value float tensor): Mean of the dataset
        std (single value float tensor): Standard deviation of the dataset
        total_sum (single value float tensor): Sum of all values in the observations
        total_observations_all_timesteps (single value float tensor): Sum of all length of observations
    """

    # N related information, useful for transition prob init
    total_state_len = torch.zeros(1).cuda().type(torch.double)
    total_mel_len = torch.zeros(1).cuda().type(torch.double)

    # Useful for data mean and data std
    total_mel_sum = torch.zeros(1).cuda().type(torch.double)
    total_mel_sq_sum = torch.zeros(1).cuda().type(torch.double)

    # For motion
    total_motion_sum = torch.zeros(1).cuda().type(torch.double)
    total_motion_sq_sum = torch.zeros(1).cuda().type(torch.double)

    # For go token
    sum_first_observation = torch.zeros(hparams.n_mel_channels + hparams.n_motion_joints).cuda().type(torch.double)

    print("For exact calculation we would do it with two loops")
    print("We first get the mean:")
    start = time.perf_counter()

    for i, batch in enumerate(tqdm(train_loader)):
        (text_inputs, text_lengths, mels, motions, mel_lengths), (
            _,
            gate_padded,
        ) = parse_batch(batch)

        total_state_len += torch.sum(text_lengths)
        total_mel_len += torch.sum(mel_lengths)

        total_mel_sum += torch.sum(mels)
        total_mel_sq_sum += torch.sum(torch.pow(mels, 2))

        total_motion_sum += torch.sum(motions)
        total_motion_sq_sum += torch.sum(torch.pow(motions, 2))

        sum_first_observation += torch.sum(torch.concat([mels[:, :, 0], motions[:, :, 0]], dim=1), dim=0)

    mel_mean = total_mel_sum / (total_mel_len * hparams.n_mel_channels)
    mel_std = torch.sqrt((total_mel_sq_sum / (total_mel_len * hparams.n_mel_channels)) - torch.pow(mel_mean, 2))

    motion_mean = total_motion_sum / (total_mel_len * hparams.n_motion_joints)
    motion_std = torch.sqrt(
        (total_motion_sq_sum / (total_mel_len * hparams.n_motion_joints)) - torch.pow(motion_mean, 2)
    )

    print("Single loop values")
    print("".join(["-"] * 50))
    print("Mel mean: ", mel_mean)
    print("Mel std: ", mel_std)
    print("Motion mean: ", motion_mean)
    print("Motion std: ", motion_std)

    N_mean = total_state_len / len(train_loader.dataset)
    if hparams.add_blank or (
        hparams.encoder_type == "conv" and hparams.encoder_params["conv"]["states_per_phone"] == 2
    ):
        N_mean *= 2

    average_mel_len = total_mel_len / len(train_loader.dataset)
    average_duration_each_state = average_mel_len / N_mean
    init_transition_prob = 1 / average_duration_each_state

    go_token_init_value = sum_first_observation / len(train_loader.dataset)
    go_token_init_value[: hparams.n_mel_channels].sub_(mel_mean).div_(mel_std)
    go_token_init_value[hparams.n_mel_channels :].sub_(motion_mean).div_(motion_std)

    print("Total Processing Time:", time.perf_counter() - start)

    print("Getting standard deviation")
    sum_of_squared_error_mel = torch.zeros(1).cuda().type(torch.double)
    sum_of_squared_error_motion = torch.zeros(1).cuda().type(torch.double)

    start = time.perf_counter()

    for i, batch in enumerate(tqdm(train_loader)):
        (text_inputs, text_lengths, mels, motions, mel_lengths), (
            _,
            gate_padded,
        ) = parse_batch(batch)
        x_minus_mean_square_mel = (mels - mel_mean).pow(2)
        x_minus_mean_square_motion = (motions - motion_mean).pow(2)

        T_max_batch = torch.max(mel_lengths)

        mask_tensor = mels.new_zeros(T_max_batch)
        mask = (
            torch.arange(float(T_max_batch), out=mask_tensor).expand(len(mel_lengths), T_max_batch)
            < (mel_lengths).unsqueeze(1)
        ).unsqueeze(1)

        x_minus_mean_square_mel *= mask.expand(len(mel_lengths), hparams.n_mel_channels, T_max_batch)
        x_minus_mean_square_motion *= mask.expand(len(mel_lengths), hparams.n_motion_joints, T_max_batch)

        sum_of_squared_error_mel += torch.sum(x_minus_mean_square_mel)
        sum_of_squared_error_motion += torch.sum(x_minus_mean_square_motion)

    std_mel = torch.sqrt(sum_of_squared_error_mel / (total_mel_len * hparams.n_mel_channels))
    std_motion = torch.sqrt(sum_of_squared_error_motion / (total_mel_len * hparams.n_motion_joints))

    print("Total Processing Time:", time.perf_counter() - start)

    mel_mean = mel_mean.type(torch.float).cpu()
    mel_std = std_mel.type(torch.float).cpu()
    motion_mean = motion_mean.type(torch.float).cpu()
    motion_std = std_motion.type(torch.float).cpu()
    go_token_init_value = go_token_init_value.type(torch.float).cpu()
    init_transition_prob = init_transition_prob.type(torch.float).cpu()

    output = {
        "mel_mean": mel_mean,
        "mel_std": mel_std,
        "motion_mean": motion_mean,
        "motion_std": motion_std,
        "go_token_init_value": go_token_init_value,
        "init_transition_prob": init_transition_prob,
    }

    return output


def main(args):
    hparams = create_hparams(generate_parameters=True)

    hparams.batch_size = args.batch_size

    trainset = TextMelLoader(hparams.training_files, hparams)
    collate_fn = TextMelMotionCollate(hparams.n_frames_per_step)

    train_loader = DataLoader(
        trainset,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    output = get_data_parameters_for_flat_start(train_loader, hparams)

    print({k: v.item() if v.numel() == 1 else v for k, v in output.items()})

    torch.save(
        output,
        args.output_file,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="data_parameters.pt",
        required=False,
        help="checkpoint path",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=256,
        required=False,
        help="batch size to fetch data properties",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        required=False,
        help="force overwrite the file",
    )
    args = parser.parse_args()

    if os.path.exists(args.output_file) and not args.force:
        print("File already exists. Use -f to force overwrite")
        sys.exit(1)

    main(args)
