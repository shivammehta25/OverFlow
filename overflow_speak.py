import json
import sys

sys.path.append("src/model")
sys.path.insert(0, "./hifigan")
import argparse  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from nltk import word_tokenize  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from hifigan.env import AttrDict  # noqa: E402
from hifigan.models import Generator  # noqa: E402
from hifigandenoiser import Denoiser  # noqa: E402
from src.hparams import create_hparams  # noqa: E402
from src.training_module import TrainingModule  # noqa: E402
from src.utilities.text import phonetise_text, text_to_sequence  # noqa: E402

device = None


def load_model(checkpoint_path):
    print("[*] Loading model")
    assert os.path.isfile(checkpoint_path), f"[-] Checkpoint file not found at {checkpoint_path} recheck the path"
    model = TrainingModule.load_from_checkpoint(checkpoint_path)
    _ = model.to(device).eval().half()
    print(f"[+] Model Loaded: {checkpoint_path}")
    return model


def configure_model(model, speaking_rate):
    model.model.hmm.hparams.max_sampling_time = 100000
    model.model.hmm.hparams.duration_quantile_threshold = speaking_rate
    model.model.hmm.hparams.deterministic_transition = True
    model.model.hmm.hparams.predict_means = False
    model.model.hmm.hparams.prenet_dropout_while_eval = True
    model.model.hmm.prenet.prenet_dropout = 0.5


def load_hifigan(hifigan_checkpoint_path, hifigan_config):
    print("[*] Loading HiFi-GAN")
    with open(hifigan_config) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    checkpoint_dict = torch.load(hifigan_checkpoint_path, map_location=device)
    generator = Generator(h).to(device)
    generator.load_state_dict(checkpoint_dict["generator"])
    generator.eval().half()
    generator.remove_weight_norm()
    print("[+] HiFi-GAN Loaded")
    return generator


def synthesise(model, sequence, vocoder, denoiser, sampling_temp):
    with torch.no_grad() and torch.inference_mode():
        mel_output, hidden_state_travelled, _, _ = model.sample(sequence.squeeze(0), sampling_temp=sampling_temp)

        mel_output = mel_output.transpose(1, 2)
        audio = vocoder(mel_output)
        audio = denoiser(audio[:, 0], strength=0.004)[:, 0]
    return audio


def main(args):
    hparams = create_hparams()
    model = load_model(args.checkpoint_path)
    configure_model(model, args.speaking_rate)
    if args.vocoder == "hifigan":
        vocoder = load_hifigan(args.hifigan_checkpoint_path, args.hifigan_config)
        denoiser = Denoiser(vocoder, mode="zeros")
    else:
        raise ValueError(f"[-] Vocoder {args.vocoder} not supported yet!")  # future failcheck

    if args.text:
        sentences = [args.text]
    elif args.file:
        with open(args.file) as f:
            sentences = f.readlines()
    sequences = [text_to_seq(sentence, hparams) for sentence in sentences]
    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    for i, sequence in tqdm(enumerate(sequences), leave=False, total=len(sequences)):
        audio = synthesise(model, sequence, vocoder, denoiser, args.sampling_temp).squeeze().cpu().numpy()
        sf.write(output_path / f"synth_{i + 1}.wav", audio, 22050, "PCM_24")

    print(f"[+] {i + 1} audio files saved to {args.output_folder}")


def text_to_seq(text, hparams):
    text += "."
    text = phonetise_text(hparams.cmu_phonetiser, text, word_tokenize)
    sequence = np.array(text_to_sequence(text, ["english_cleaners"]))[None, :]
    sequence = torch.from_numpy(sequence).to(device).long()
    return sequence


def validate_args(args):
    assert os.path.isfile(
        args.checkpoint_path
    ), f"[-] Checkpoint file not found at {args.checkpoint_path} recheck the path"
    assert (
        args.text or args.file
    ), f"[-] Either text or file must be provided provided -> text: {args.text} file: {args.file}"
    if args.vocoder == "hifigan":
        assert os.path.isfile(
            args.hifigan_checkpoint_path
        ), f"[-] Vocoder checkpoint path not found at {args.hifigan_checkpoint_path}"
        assert os.path.isfile(args.hifigan_config), f"[-] Vocoder config file not found at {args.hifigan_config}"


def restricted_speaking_rate(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x} not a floating-point literal")

    if x <= 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(f"Speaking rate: {x} must be in range [0.0, 1.0]")
    return x


def speak():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--text", type=str, default=None, help="text to speak")
    parser.add_argument(
        "-f", "--file", type=str, default=None, help="file containing sentences each on different line to speak"
    )
    parser.add_argument(
        "-o", "--output_folder", type=str, default="synthesised_wavs", help="output folder to save audio files"
    )
    parser.add_argument("-c", "--checkpoint_path", type=str, default="OverFlow-Female.ckpt", help="checkpoint path")
    parser.add_argument("-v", "--vocoder", type=str, default="hifigan", help="vocoder to use", choices=["hifigan"])
    parser.add_argument(
        "-hp", "--hifigan_checkpoint_path", type=str, default="g_02500000", help="hifigan checkpoint path"
    )
    parser.add_argument(
        "-hc", "--hifigan_config", type=str, default="hifigan/config_v1.json", help="hifigan config file"
    )
    parser.add_argument("-d", "--device", type=str, default="cuda", help="device to use", choices=["cuda", "cpu"]),
    parser.add_argument("-sr", "--speaking_rate", type=restricted_speaking_rate, default=0.55, help="speaking rate")
    parser.add_argument("-st", "--sampling_temp", type=restricted_speaking_rate, default=0.667, help="speaking rate")

    global device
    args = parser.parse_args()
    device = torch.device(args.device)
    print(f"[*] Using device: {device}")
    print(f"[*] With args: {args}")
    validate_args(args)
    main(args)


if __name__ == "__main__":
    speak()
