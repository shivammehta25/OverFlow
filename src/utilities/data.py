r"""
data.py

Utilities for processing of Data
"""
import random
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io.wavfile import read
from torch.utils.data.dataset import Dataset
from tqdm.auto import tqdm

from src.model.layers import TacotronSTFT
from src.utilities.text import (
    _clean_text,
    cleaned_text_to_sequence,
    intersperse,
    text_to_sequence,
)


def load_wav_to_torch(full_path):
    r"""
    Uses scipy to convert the wav file into torch tensor
    Args:
        full_path: "Wave location"

    Returns:
        torch.FloatTensor of wav data and sampling rate
    """
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def cache_text(data_item, text_cleaners):
    loc, original_text = data_item
    cleaned_text = _clean_text(original_text, text_cleaners)
    output = f"{loc}|{cleaned_text}\n"
    return output


def cache_mel(data_item, mel_function, ext=".npy"):
    loc, _ = data_item
    if Path(loc).with_suffix(ext).exists():
        return 0  # Already cached
    mel = mel_function(loc).numpy()
    np.save(Path(loc).with_suffix(ext), mel)
    return 1


class TextMelMotionCollate:
    r"""
    Zero-pads model inputs and targets based on number of frames per setep
    """

    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        r"""
        Collate's training batch from normalized text and mel-spectrogram

        Args:
            batch (List): [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True
        )
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, : text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max(x[1].size(1) for x in batch)

        # include mel padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))

        # include motion padded
        num_motion = batch[0][2].size(0)
        motion_padded = torch.FloatTensor(len(batch), num_motion, max_target_len)
        motion_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, : mel.size(1)] = mel

            motion = batch[ids_sorted_decreasing[i]][2]
            motion_padded[i, :, : motion.size(1)] = motion

            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, motion_padded, output_lengths


class TextMelLoader(Dataset):
    r"""
    Taken from Nvidia-Tacotron-2 implementation

    1) loads audio,text pairs
    2) normalizes text and converts them to sequences of one-hot vectors
    3) computes mel-spectrograms from audio files.
    """

    def __init__(self, audiopaths_and_text, hparams, mel_transform=None, motion_transform=None):
        r"""
        Args:
            audiopaths_and_text:
            hparams:
            transform (list): list of transformation
        """
        self.file_loc = Path(audiopaths_and_text)
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.motion_fileloc = Path(hparams.motion_fileloc)
        self.mel_transform = mel_transform
        self.motion_transform = motion_transform
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.phonetise = hparams.phonetise
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.add_blank = hparams.add_blank
        self.n_motion_joints = hparams.n_motion_joints
        self.stft = TacotronSTFT(
            hparams.filter_length,
            hparams.hop_length,
            hparams.win_length,
            hparams.n_mel_channels,
            hparams.sampling_rate,
            hparams.mel_fmin,
            hparams.mel_fmax,
        )
        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)
        self.cleaned_text = False
        self.preprocess_text()
        self.preprocess_mels()

    def preprocess_text(self):
        out_filename = self.file_loc.with_suffix(f"{self.file_loc.suffix}.cleaned")
        if not out_filename.exists():
            print(f"Cache not found caching the dataset: {self.file_loc}")
            output = []

            pbar = tqdm(self.audiopaths_and_text)
            pbar.set_description("Caching data with cpu count: " + str(cpu_count()))
            with Pool(cpu_count()) as p:
                for _, data_item in enumerate(
                    p.imap(
                        partial(cache_text, text_cleaners=self.text_cleaners), self.audiopaths_and_text, chunksize=10
                    )
                ):
                    if data_item is not None:
                        output.append(data_item)
                    pbar.update(1)
                    p.close()

            with open(out_filename, "w", encoding="utf-8") as f:
                f.writelines(output)

            print("Done caching the dataset")
        else:
            print(f"Data cache found at : {out_filename}! Loading cache...")
        self.audiopaths_and_text = load_filepaths_and_text(out_filename)
        self.cleaned_text = True

    def preprocess_mels(self):
        pbar = tqdm(self.audiopaths_and_text, leave=False)
        total = 0

        for i, data_item in enumerate(pbar):
            cached = cache_mel(data_item, mel_function=self.get_mel)
            total += cached

        self.load_mel_from_disk = True
        print("Done caching mels! New mels cached: " + str(total))

    def get_mel_text_motion_tuple(self, audiopath_and_text):
        r"""
        Takes audiopath_text list input where list[0] is location for wav file
            and list[1] is the text
        Args:
            audiopath_and_text (list): list of size 2
        """
        # separate filename and text (string)
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        # This text is int tensor of the input representation
        text = self.get_text(text)
        mel = self.get_mel(audiopath)

        if self.mel_transform:
            for t in self.mel_transform:
                mel = t(mel)

        motion = self.get_motion(audiopath, mel.shape[1])
        if self.motion_transform:
            for t in self.motion_transform:
                motion = t(motion)
        mel, motion = self.resize_mel_motion_to_same_size(mel, motion)
        return (text, mel, motion)

    def get_motion(self, filename, mel_shape, ext=".expmap_86.1328125fps.pkl"):
        try:
            # raise FileNotFoundError
            file_loc = self.motion_fileloc / Path(Path(filename).name).with_suffix(ext)
            motion = torch.from_numpy(pd.read_pickle(file_loc).to_numpy())
            # motion = torch.concat([motion, torch.randn(motion.shape[0], 3)], dim=1)
            motion = torch.concat([motion, torch.zeros(motion.shape[0], 3)], dim=1)

            # motion += torch.randn(motion.shape) * 0.01
        except FileNotFoundError:
            motion = torch.randn(mel_shape, self.n_motion_joints)
        return motion.T

    def resize_mel_motion_to_same_size(self, mel, motion):
        splitter_idx = min(mel.shape[1], motion.shape[1])
        mel, motion = mel[:, :splitter_idx], motion[:, :splitter_idx]
        return mel, motion

    def get_mel(self, filename):
        r"""
        Takes filename as input and returns its mel spectrogram
        Args:
            filename (string): Example: 'LJSpeech-1.1/wavs/LJ039-0212.wav'
        """
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError(f"{sampling_rate} SR doesn't match target {self.stft.sampling_rate} SR")
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(Path(filename).with_suffix(".npy")))
            assert melspec.size(0) == self.stft.n_mel_channels, "Mel dimension mismatch: given {}, expected {}".format(
                melspec.size(0), self.stft.n_mel_channels
            )
        return melspec

    def get_text(self, text):
        if self.cleaned_text:
            text_norm = cleaned_text_to_sequence(text)
        else:
            text_norm = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    # def get_text(self, text):
    #     if self.phonetise:
    #         text = phonetise_text(self.cmu_phonetiser, text, word_tokenize)

    #     text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
    #     return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_motion_tuple(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class Normalise:
    r"""
    Z-Score normalisation class / Standardisation class
    normalises the data with mean and std, when the data object is called

    Args:
        mean (int/tensor): Mean of the data
        std (int/tensor): Standard deviation
    """

    def __init__(self, mean, std):
        super().__init__()

        if not torch.is_tensor(mean):
            mean = torch.tensor(mean)
        if not torch.is_tensor(std):
            std = torch.tensor(std)

        self.mean = mean
        self.std = std

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        r"""
        Takes an input and normalises it

        Args:
            x (Any): Input to the normaliser

        Returns:
            (torch.FloatTensor): Normalised value
        """
        if not torch.is_tensor(x):
            x = torch.tensor(x)

        x = x.sub(self.mean).div(self.std)
        return x

    def inverse_normalise(self, x):
        r"""
        Takes an input and de-normalises it

        Args:
            x (Any): Input to the normaliser

        Returns:
            (torch.FloatTensor): Normalised value
        """
        if not torch.is_tensor(x):
            x = torch.tensor([x])

        x = x.mul(self.std).add(self.mean)
        return x
