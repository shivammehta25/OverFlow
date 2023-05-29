r"""
train.py

PyTorch-Lightning Trainer file, main file to run your training
"""
import argparse
import os

import lightning as L
import torch
from lightning import seed_everything
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from src.data_module import LightningLoader
from src.hparams import create_hparams
from src.training_module import TrainingModule


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print(f"Warm starting model from checkpoint '{checkpoint_path}'")
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    model_dict = checkpoint_dict["state_dict"]
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items() if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--checkpoint_path",
        type=str,
        default=None,
        required=False,
        help="checkpoint path",
    )
    parser.add_argument("-r", "--run_name", type=str, default=None, required=False, help="run name")
    parser.add_argument("-g", "--gpus", nargs="*", default=None, required=False, help="gpu")
    parser.add_argument("-w", "--warm_start", action="store_true", default=False, help="warm start")

    args = parser.parse_args()

    if args.checkpoint_path and not os.path.exists(args.checkpoint_path):
        raise FileExistsError("Check point not present recheck the name")

    hparams = create_hparams()
    if args.run_name:
        hparams.run_name = args.run_name

    if args.gpus:
        hparams.gpus = args.gpus

    hparams.warm_start = args.warm_start
    hparams.checkpoint_path = args.checkpoint_path

    seed_everything(hparams.seed)

    data_module = LightningLoader(hparams)
    model = TrainingModule(hparams)

    def count_parameters(model, element_name):
        return f"{element_name}  has {sum(p.numel() for p in model.parameters() if p.requires_grad): ,} trainable \
            parameters"

    elements = {"Encoder": model.model.encoder, "HMM": model.model.hmm, "Decoder": model.model.decoder}

    for element_name, element in elements.items():
        print(count_parameters(element, element_name))

    if hparams.warm_start:
        model = warm_start_model(args.checkpoint_path, model, hparams.ignore_layers)
        args.checkpoint_path = None
        # We have already loaded the model weights, so we don't want to load optimizer states from checkpoint

    logger = TensorBoardLogger(hparams.tensorboard_log_dir, name=hparams.run_name)

    trainer = L.Trainer(
        devices=hparams.gpus,
        logger=logger,
        log_every_n_steps=1,
        val_check_interval=hparams.val_check_interval,
        gradient_clip_val=hparams.grad_clip_thresh,
        max_epochs=hparams.max_epochs,
        precision=hparams.precision,
        callbacks=[DeviceStatsMonitor()],
    )

    trainer.fit(model=model, datamodule=data_module, ckpt_path=args.checkpoint_path)
