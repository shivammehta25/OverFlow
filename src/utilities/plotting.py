r"""
plotting.py

File contains utilities for plotting
"""
import subprocess
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import soundfile as sf
import torch
from matplotlib.colors import LogNorm

from pymo.preprocessing import MocapParameterizer
from pymo.viz_tools import render_mp4
from pymo.writers import BVHWriter

matplotlib.use("Agg")


def validate_numpy_array(value: Any):
    r"""
    Validates the input and makes sure it returns a numpy array (i.e on CPU)

    Args:
        value (Any): the input value

    Raises:
        TypeError: if the value is not a numpy array or torch tensor

    Returns:
        np.ndarray: numpy array of the value
    """
    if isinstance(value, np.ndarray):
        return value
    elif isinstance(value, list):
        return np.array(value)
    elif torch.is_tensor(value):
        return value.cpu().numpy()
    else:
        raise TypeError("Value must be a numpy array, a torch tensor or a list")


def save_figure_to_numpy(fig):
    r"""
    Saves the figure to an numpy array
    """
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_alpha_scaled_to_numpy(alpha_scaled, plot_logrithmic=False):
    r"""
    Plots the forward probabilities to a sns heapmap

    Args:
        alpha_scaled (torch.FloatTensor): forward probabilities
        plot_logrithmic (Optional[bool]): [description]. Defaults to False.
    """
    alpha_scaled = validate_numpy_array(alpha_scaled)

    fig, ax = plt.subplots(figsize=(20, 20))

    if plot_logrithmic:
        ax = sns.heatmap(np.exp(alpha_scaled), cmap="viridis", norm=LogNorm())
        ax.set_title("State probabilities in logarithmic scale")
    else:
        ax = sns.heatmap(np.exp(alpha_scaled), cmap="viridis")
        ax.set_title("State probabilities")

    ax.set_xlabel("Time step")
    ax.set_ylabel("Hidden states")
    ax.invert_yaxis()
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_transition_matrix(transition_matrix):
    r"""
    Plots the transition matrix obtained in the forward pass

    Args:
        transition_matrix (torch.FloatTensor): transition matrix
    """
    transition_matrix = validate_numpy_array(transition_matrix)

    fig = plt.figure(figsize=(20, 20))
    ax = sns.heatmap(data=transition_matrix.T, annot=False, cbar=True)
    ax.set_ylabel("Hidden states")
    ax.set_xlabel("Time step")
    ax.set_title("Transition probabilities")
    ax.invert_yaxis()
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_mel_spectrogram_to_numpy(mel_spectrogram):
    r"""
    Plots the spectrogram to numpy

    Args:
        spectrogram (torch.FloatTensor): the mel-spectrogram
    """
    mel_spectrogram = validate_numpy_array(mel_spectrogram)

    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(mel_spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_transition_probabilities_to_numpy(states, transition_probabilities):
    """Generates trainsition probabilities plot for the states and the probability of transition.

    Args:
        states (torch.IntTensor): the states
        transition_probabilities (torch.FloatTensor): the transition probabilities
    """
    states = validate_numpy_array(states)
    transition_probabilities = validate_numpy_array(transition_probabilities)

    fig, ax = plt.subplots(figsize=(30, 3))
    ax.plot(transition_probabilities, "o")
    ax.set_title("Transition probability of state")
    ax.set_xlabel("hidden state")
    ax.set_ylabel("probability")
    ax.set_xticks([i for i in range(len(transition_probabilities))])
    ax.set_xticklabels([int(x) for x in states], rotation=90)

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_go_tokens_to_numpy(go_tokens):
    """Plots the trainable go token to numpy.

    Args:
        go_tokens (torch.FloatTensor): go tokens
    """
    go_tokens = validate_numpy_array(go_tokens)

    fig = plt.figure(figsize=(20, 10))
    ax = sns.heatmap(data=go_tokens, annot=False, cbar=True)
    ax.set_title("Go token")
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_hidden_states_to_numpy(hidden_states):
    """Plots the hidden state to numpy.

    Args:
        hidden_states (torch.FloatTensor): hidden states
    """
    hidden_states = validate_numpy_array(hidden_states)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(hidden_states)
    plt.xlabel("Time")
    plt.ylabel("Hidden State Travelled")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def combine_video_audio(video_filename, audio_filename, final_filename):
    command = f"ffmpeg -i {video_filename} -i {audio_filename} -c:v copy -c:a aac {final_filename} -y"
    subprocess.check_call(command, shell=True)
    Path(video_filename).unlink()
    Path(audio_filename).unlink()


def generate_motion_visualization(
    audio, audio_filename, motion, motion_filename, motion_visualizer_pipeline, bvh_filename=None
):
    """

    Args:
        audio (_type_): (1, T_audio)
        audio_filename (_type_): str/path
        motion (_type_): (T_motion, 45)
        motion_filename (_type_): str/path
        motion_visualizer_pipeline (_type_): Pipeline
        bvh_filename (_type_, optional): str/path. Defaults to None.
    """
    audio_filename = Path(audio_filename)
    motion_filename = Path(motion_filename)

    sf.write(audio_filename, audio.flatten(), 22500, "PCM_24")

    # Add motion target
    bvh_values = motion_visualizer_pipeline.inverse_transform([motion])

    if bvh_filename is not None:
        # Write input bvh file
        writer = BVHWriter()
        with open(bvh_filename, "w") as f:
            writer.write(bvh_values[0], f)

    # To stickfigure
    X_pos = MocapParameterizer("position").fit_transform(bvh_values)

    render_mp4(X_pos[0], motion_filename.with_suffix(".temp.mp4"), axis_scale=200)
    combine_video_audio(
        motion_filename.with_suffix(".temp.mp4"),
        audio_filename,
        motion_filename,
    )
