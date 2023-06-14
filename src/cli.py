import argparse
import json
from pathlib import Path

import gdown
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from hifigan.env import AttrDict
from hifigan.hifigandenoiser import Denoiser
from hifigan.models import Generator
from src.training_module import TrainingModule
from src.utilities.text import _clean_text, cleaned_text_to_sequence, intersperse

HIFIGAN_URL = "https://drive.google.com/file/d/1qpgI41wNXFcH-iKq1Y42JlBC9j0je8PW/view?usp=drive_link"
HIFIGAN_CONFIG = "https://drive.google.com/file/d/1pAB2kQunkDuv6W5fcJiQ0CY8xcJKB22e/view?usp=drive_link"
OVERFLOW_URL = "https://drive.google.com/file/d/1bkHVlM_NutczEBd9ZxzuF6cJBuIvnWLX/view?usp=sharing"


def plot_spectrogram_to_numpy(spectrogram, filename):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.title("Synthesised Mel-Spectrogram")
    fig.canvas.draw()
    plt.savefig(filename)


def get_text(i, text, hparams, device):
    print("".join(["="] * 100))
    print(f"{i} - Input text: {text}")
    text = _clean_text(text, hparams.text_cleaners)
    print(f"{i} - Phonetised text: {text}")
    sequence = cleaned_text_to_sequence(text)
    if hparams.add_blank:
        sequence = intersperse(sequence, 0)
    sequence = torch.LongTensor(sequence).to(device)
    return sequence


def get_texts(args):
    if args.text:
        texts = [args.text]
    else:
        with open(args.file) as f:
            texts = f.readlines()
    return texts


def assert_model_downloaded(checkpoint_path, url):
    if Path(checkpoint_path).exists():
        return
    print(f"[-] Model not found at {checkpoint_path}! Will download it")
    gdown.download(url=url, output=checkpoint_path, quiet=False, fuzzy=True)


def validate_args(args):
    assert args.text or args.file, "Either text or file must be provided I need something to synthesise from"
    assert args.speaking_rate > 0 and args.speaking_rate <= 1, "Speaking rate must be between 0 and 1"
    assert args.sampling_temp > 0, "Sampling temperature must be greater than 0"
    assert args.speaker_id >= 0 and args.speaker_id < 109, "Speaker ID must be between 0 and 108"
    assert_model_downloaded(args.hifigan_path, HIFIGAN_URL)
    assert_model_downloaded(args.checkpoint_path, OVERFLOW_URL)
    assert_model_downloaded("config.json", HIFIGAN_CONFIG)


def load_hifi_gan(hifi_gan_checkpoint_path, config_file, device):
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)

    h = AttrDict(json_config)
    torch.manual_seed(h.seed)
    generator = Generator(h).to(device)
    state_dict_g = torch.load(hifi_gan_checkpoint_path, map_location=device)
    generator.load_state_dict(state_dict_g["generator"])
    if device.type.startswith("cuda"):
        generator.eval().half()
    else:
        generator.eval()

    generator.remove_weight_norm()

    denoiser = Denoiser(generator, mode="zeros")
    return generator, denoiser


@torch.inference_mode()
def cli():
    parser = argparse.ArgumentParser(description="OverFlow: Putting flows on top of neural transducers for better TTS")
    parser.add_argument("-t", "--text", type=str, default=None, help="Text to synthesize")
    parser.add_argument("-f", "--file", type=str, default=None, help="Text file to synthesize")
    parser.add_argument(
        "-c",
        "--checkpoint_path",
        type=str,
        default="multispeaker_overflow.ckpt",
        help="OverFlow checkpoint path",
    )
    parser.add_argument(
        "-hc",
        "--hifigan_path",
        type=str,
        default="g_02500000",
        help="HiFiGAN checkpoint path",
    )
    parser.add_argument("--cpu", action="store_true", help="Use CPU for inference (default: use GPU if available)")
    parser.add_argument(
        "-sr", "--speaking_rate", type=float, default=0.4, help="Speaking rate of the synthesised speech"
    )
    parser.add_argument("--max_sampling_time", type=int, default=10000, help="Max sampling time in frames")
    parser.add_argument("--sampling_temp", type=float, default=0.667, help="Sampling temperature")
    parser.add_argument("-s", "--speaker_id", type=int, default=49, help="Speaker ID")
    args = parser.parse_args()
    validate_args(args)
    device = get_device(args)
    print_config(args)

    print("[!] Loading OverFlow!")
    model = TrainingModule.load_from_checkpoint(args.checkpoint_path, map_location=device)
    model.to(device)
    hparams = model.hparams
    if device.type.startswith("cuda"):
        model.eval().half()
    else:
        model.eval()

    model.model.hmm.hparams.max_sampling_time = args.max_sampling_time
    model.model.hmm.hparams.duration_quantile_threshold = args.speaking_rate
    model.model.hmm.hparams.deterministic_transition = True
    print("[+] OverFlow loaded!")

    print("[!] Loading HiFiGAN model!")
    generator, denoiser = load_hifi_gan(args.hifigan_path, "config.json", device)
    print("[+] HiFiGAN model loaded!")

    texts = get_texts(args)

    t = args.sampling_temp
    speaker_id = torch.LongTensor([args.speaker_id]).to(device)

    for i, text in enumerate(texts):
        i = i + 1
        base_name = f"utterance_{i:03d}_speaker_{args.speaker_id:03d}"
        text = text.strip()
        text = text + "." if not text.endswith(".") else text
        sequence = get_text(i, text, hparams, device)
        print(f"[+] Generating mel spectrogram for {i}")
        mel_output, hidden_state_travelled, _, _ = model.sample(
            sequence.squeeze(0), speaker_id=speaker_id, sampling_temp=t
        )
        plot_spectrogram_to_numpy(np.array(mel_output.float().cpu()).T, f"{base_name}.png")
        print(f"[+] Mel spectrogram saved at {base_name}.png")
        print(f"[+] Generating waveform for {i}")
        mel_output = mel_output.transpose(1, 2)
        audio = generator(mel_output)
        audio = denoiser(audio[:, 0], strength=0.001)[:, 0]
        sf.write(f"{base_name}.wav", audio.squeeze().cpu().numpy(), 22050, "PCM_24")
        print(f"[+] Waveform saved at {base_name}.wav")

    print("".join(["="] * 100))
    print("[+] All text synthesised!")


def print_config(args):
    print("[!] Configurations")
    print(f"\t- Speaking rate: {args.speaking_rate}")
    print(f"\t- Speaker ID: {args.speaker_id}")


def get_device(args):
    if torch.cuda.is_available() and not args.cpu:
        print("[+] GPU Available! Using GPU")
        device = torch.device("cuda")
    else:
        print("[-] GPU not available or forced CPU run! Using CPU")
        device = torch.device("cpu")
    return device
