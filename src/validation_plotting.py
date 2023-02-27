import subprocess
import warnings
from pathlib import Path

import soundfile as sf
import torch
from pytorch_lightning.utilities import rank_zero_only

from pymo.preprocessing import MocapParameterizer
from pymo.viz_tools import render_mp4
from pymo.writers import BVHWriter
from src.utilities.plotting import (
    plot_alpha_scaled_to_numpy,
    plot_go_tokens_to_numpy,
    plot_hidden_states_to_numpy,
    plot_mel_spectrogram_to_numpy,
    plot_transition_matrix,
    plot_transition_probabilities_to_numpy,
)


@rank_zero_only
@torch.inference_mode()
def log_validation(
    logger,
    model,
    mel_output,
    mel_output_normalised,
    state_travelled,
    mel_targets,
    input_parameters,
    output_parameters,
    iteration,
    stft_module,
    motion_input,
    motion_output,
    motion_visualizer_pipeline,
    resumed_from_checkpoint,
):
    """
    Args:
        logger (SummaryWriter): logger from pytorch lightning
        model: model to plot alpha scaled
        mel_output: mel generated
        mel_output_normalised: normalised version of mel output
        state_travelled: phones/states travelled
        mel_targets: target mel
        input_parameters: input parameters to decoder model while sampling
        output_parameters: output parameters from the decoder model while sampling
        iteration: iteration number
        Stft_fn: stft function to generate waveform using griffin lim

    Returns:
        None
    """
    # plot distribution of parameters
    for tag, value in model.named_parameters():
        tag = tag.replace(".", "/")
        logger.add_histogram(tag, value.data.cpu().numpy(), iteration)

    logger.add_image(
        "alignment/log_alpha_scaled",
        plot_alpha_scaled_to_numpy(model.hmm.log_alpha_scaled[0, :, :].T, plot_logrithmic=True),
        iteration,
        dataformats="HWC",
    )

    logger.add_image(
        "transition_probabilities",
        plot_transition_matrix(torch.sigmoid(model.hmm.transition_vector[0, :, :])),
        iteration,
        dataformats="HWC",
    )

    logger.add_image(
        "alignment/alpha_scaled",
        plot_alpha_scaled_to_numpy(torch.exp(model.hmm.log_alpha_scaled[0, :, :]).T),
        iteration,
        dataformats="HWC",
    )

    logger.add_image(
        "mel_target",
        plot_mel_spectrogram_to_numpy(mel_targets),
        iteration,
        dataformats="HWC",
    )

    logger.add_image(
        "synthesised/mel_synthesised",
        plot_mel_spectrogram_to_numpy(mel_output.squeeze(0).T),
        iteration,
        dataformats="HWC",
    )
    generated_audio, sr = stft_module.griffin_lim(mel_output.transpose(1, 2))
    logger.add_audio("synthesised/waveform_synthesised", generated_audio, iteration, sample_rate=sr)

    logger.add_image(
        "synthesised/mel_synthesised_normalised",
        plot_mel_spectrogram_to_numpy(mel_output_normalised.squeeze(0).T),
        iteration,
        dataformats="HWC",
    )

    logger.add_image(
        "synthesised/hidden_state_travelled",
        plot_hidden_states_to_numpy(state_travelled),
        iteration,
        dataformats="HWC",
    )

    logger.add_image(
        "parameters/go_tokens",
        plot_go_tokens_to_numpy(model.hmm.go_tokens.clone().detach()),
        iteration,
        dataformats="HWC",
    )

    states = [p[1] for p in input_parameters]
    transition_probability_synthesising = [p[2].cpu().numpy() for p in output_parameters]

    for i in range((len(transition_probability_synthesising) // 200) + 1):
        start = i * 200
        end = (i + 1) * 200
        logger.add_image(
            f"synthesised_transition_probabilities/{i}",
            plot_transition_probabilities_to_numpy(states[start:end], transition_probability_synthesising[start:end]),
            iteration,
            dataformats="HWC",
        )

    # Plotting means of most probable state
    max_state_numbers = torch.max(model.hmm.log_alpha_scaled[0, :, :], dim=1)[1]
    means = torch.stack(model.hmm.means, dim=1).squeeze(0)

    max_len = means.shape[0]
    n_mel_channels = means.shape[2]

    max_state_numbers = max_state_numbers.unsqueeze(1).unsqueeze(1).expand(max_len, 1, n_mel_channels)
    means = torch.gather(means, 1, max_state_numbers).squeeze(1)

    # Passing through the decoder
    mel_mean, _, _ = model.decoder_mel(
        means.T[: model.n_mel_channels].unsqueeze(0), means.new_tensor([means.shape[0]]).int(), reverse=True
    )

    logger.add_image(
        "mel_from_the_means_predicted_by_most_probable_state",
        plot_mel_spectrogram_to_numpy(mel_mean.squeeze(0)),
        iteration,
        dataformats="HWC",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if iteration == 0:
            # Add target audio
            target_audio, sr = stft_module.griffin_lim(mel_targets.unsqueeze(0))
            logger.add_audio("natural_speech/griffin_lim", target_audio, iteration, sample_rate=sr)
            sf.write(f"{logger.log_dir}/input_{iteration}.wav", target_audio.flatten(), 22500, "PCM_24")

            # Add motion target
            motion_input = motion_input.squeeze(0).cpu().numpy()[:45].T
            bvh_values = motion_visualizer_pipeline.inverse_transform([motion_input])
            bvh_filename = f"{logger.log_dir}/input_{iteration}.bvh"

            # Write input bvh file
            writer = BVHWriter()
            with open(bvh_filename, "w") as f:
                writer.write(bvh_values[0], f)

            # To stickfigure
            X_pos = MocapParameterizer("position").fit_transform(bvh_values)
            render_mp4(X_pos[0], f"{logger.log_dir}/input_temp_{iteration}.mp4", axis_scale=200)
            combine_video_audio(
                f"{logger.log_dir}/input_temp_{iteration}.mp4",
                f"{logger.log_dir}/input_{iteration}.wav",
                f"{logger.log_dir}/input_{iteration}.mp4",
            )

        # Generate motion output
        motion_output = motion_output.squeeze(0).cpu().numpy()[:, :45]
        bvh_values = motion_visualizer_pipeline.inverse_transform([motion_output])
        X_pos = MocapParameterizer("position").fit_transform(bvh_values)
        sf.write(f"{logger.log_dir}/output_{iteration}.wav", generated_audio.flatten(), 22500, "PCM_24")
        render_mp4(X_pos[0], f"{logger.log_dir}/output_temp_{iteration}.mp4", axis_scale=200)
        combine_video_audio(
            f"{logger.log_dir}/output_temp_{iteration}.mp4",
            f"{logger.log_dir}/output_{iteration}.wav",
            f"{logger.log_dir}/output_{iteration}.mp4",
        )


def combine_video_audio(video_filename, audio_filename, final_filename):
    command = f"ffmpeg -i {video_filename} -i {audio_filename} -c:v copy -c:a aac {final_filename}"
    subprocess.check_call(command, shell=True)
    Path(video_filename).unlink()
    Path(audio_filename).unlink()
