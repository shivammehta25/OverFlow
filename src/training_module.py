r"""
training_model.py

This file contains PyTorch Lightning's main module where code
of the main model is implemented
"""
import os
from argparse import Namespace

import pytorch_lightning as pl
import torch

from src.model.OverFlow import OverFlow
from src.validation_plotting import log_validation


class TrainingModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        if type(hparams) != Namespace:
            hparams = Namespace(**hparams)
        self.motion_visualizer = hparams.motion_visualizer
        del hparams.motion_visualizer
        self.save_hyperparameters(hparams)
        hparams.logger = self.logger
        self.max_gpu_usage = 0
        self.model = OverFlow(hparams)

    def forward(self, x):
        r"""
        Forward pass of the model

        Args:
            x (Any): input to the forward function

        Returns:
            output (Any): output of the forward function
        """
        log_probs = self.model(x)
        return log_probs

    def configure_optimizers(self):
        r"""
        Configure optimizer

        Returns:
            (torch.optim.Optimizer)
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.optimizer_params["scheduler"] == "noam":
            self.warmup = self.hparams.optimizer_params["warmup"]

            def warm_decay(step):
                if step < self.warmup:
                    return step / self.warmup
                return self.warmup**0.5 * step**-0.5

            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, warm_decay),
                "interval": "step",  # runs per batch rather than per epoch
                "frequency": 1,
                "name": "trainer_stats/NoamLR",  # uncomment if using LearningRateMonitor
            }
            return [optimizer], [scheduler]

        else:
            return optimizer

    def training_step(self, train_batch, batch_idx):
        r"""
        Main training loop of your model

        Args:
            train_batch (List): batch of input data
            batch_idx (Int): index of the current batch

        Returns:
            loss (torch.FloatTensor): loss of the forward run of your model
        """
        x, y = self.model.parse_batch(train_batch)
        log_probs = self(x)
        loss = -log_probs.mean()
        self.log(
            "loss/train",
            loss.item(),
            prog_bar=True,
            on_step=True,
            sync_dist=True,
            logger=True,
        )
        self.log(
            "Global_Step",
            int(self.global_step),
            prog_bar=True,
            on_step=True,
            sync_dist=True,
            logger=False,
        )
        self.set_gpu_stats(loss)
        return loss

    def set_gpu_stats(self, some_tensor):
        free, total = (x / (1024 * 1024) for x in torch.cuda.mem_get_info(some_tensor.device.index))
        current = total - free
        self.max_gpu_usage = max(self.max_gpu_usage, current)
        self.log(
            "trainer_stats/MaxGPUMemory", self.max_gpu_usage, logger=True, prog_bar=True, on_step=True, sync_dist=True
        )
        self.log("trainer_stats/CurrentGPUMemory", current, logger=True, prog_bar=True, on_step=True, sync_dist=True)

    def validation_step(self, val_batch, batch_idx):
        r"""
        Validation step

        Args:
            val_batch (Any): output depends what you are returning from the train loop
            batch_idx (): batch index
        """
        x, y = self.model.parse_batch(val_batch)
        log_probs = self(x)
        loss = -log_probs.mean()
        self.log("loss/val", loss.item(), prog_bar=True, sync_dist=True, logger=True)
        return loss

    def on_before_zero_grad(self, optimizer):
        r"""
        Takes actions before zeroing the gradients.
        We use it to plot the output of the model at
        the save_model_checkpoint iteration from hparams.

        Args:
            optimizer ([type]): [description]
        """

        if self.trainer.is_global_zero and (self.global_step % self.hparams.save_model_checkpoint == 0):
            (
                text_inputs,
                text_lengths,
                mels,
                motions,
                mel_lengths,
            ) = self.get_an_element_of_validation_dataset()
            (
                mel_output,
                motion_output,
                state_travelled,
                input_parameters,
                output_parameters,
            ) = self.model.sample(text_inputs[0], text_lengths[0])
            mel_output_normalised = self.model.mel_normaliser(mel_output)
            # motion_output_normalised = self.model.motion_normaliser(motion_output)

            with torch.inference_mode():
                _ = self.model((text_inputs, text_lengths, mels, motions, mel_lengths))

            log_validation(
                self.logger.experiment,
                self.model,
                mel_output,
                mel_output_normalised,
                state_travelled,
                self.hparams.mel_normaliser.inverse_normalise(mels[0]),
                input_parameters,
                output_parameters,
                self.global_step,
                self.train_dataloader().dataset.stft,
                self.hparams.motion_normaliser.inverse_normalise(motions),
                motion_output,
                self.motion_visualizer,
                self.trainer.resume_from_checkpoint,
            )

            self.trainer.save_checkpoint(
                os.path.join(
                    self.hparams.checkpoint_dir,
                    self.hparams.run_name,
                    f"checkpoint_{self.global_step}.ckpt",
                )
            )

    def get_an_element_of_validation_dataset(self):
        r"""
        Gets an element of the validation dataset.

        Returns:
            text_inputs (torch.FloatTensor): The text inputs.
            text_lengths (torch.LongTensor): The text lengths.
            mels (torch.FloatTensor): The mels spectrogram.
            max_len (int): The maximum length of the mels spectrogram.
            mel_lengths (torch.LongTensor): The lengths of the mel spectrogram.
        """
        x, y = self.model.parse_batch(next(iter(self.val_dataloader())))
        (text_inputs, text_lengths, mels, motions, mel_lengths) = x
        text_inputs = text_inputs[0].unsqueeze(0).to(self.device)
        text_lengths = text_lengths[0].unsqueeze(0).to(self.device)
        mels = mels[0].unsqueeze(0).to(self.device)
        motions = motions[0].unsqueeze(0).to(self.device)
        mel_lengths = mel_lengths[0].unsqueeze(0).to(self.device)
        # Sometimes in a batch the element which has the maximum mel len
        # is not the same as the element which has the maximum text len.
        # This prevent the model to break down when plotting validation.
        mels = mels[:, :, : mel_lengths.item()]
        motions = motions[:, :, : mel_lengths.item()]

        return text_inputs, text_lengths, mels, motions, mel_lengths

    @torch.inference_mode()
    def inference(self, text_inputs, sampling_temp=1.0):
        """
        Similar to sampling but returns only mel outputs and states travelled.

        Args:
            text_inputs (torch.IntTensor): phonetised text inputs

        Returns:
            torch.FloatTensor: mel outputs
            torch.IntTensor: states travelled
        """
        mel_output, states_travelled, _, _ = self.sample(text_inputs, sampling_temp=sampling_temp)
        return mel_output, states_travelled

    @torch.inference_mode()
    def sample(self, text_inputs, text_lengths=None, sampling_temps={"audio": 1.0, "motion": 1.0}):
        """
        Samples from the model

        Args:
            text_inputs (torch.IntTensor): phonetised text inputs
            text_lengths (torch.IntTensor, Optional): text lengths

        Returns:
            torch.FloatTensor: mel outputs
            torch.InteTensor: states travelled
            List[Tuple[torch.FloatTensor]]: input parameters
            List[Tuple[torch.FloatTensor]]: output parameters
        """
        return self.model.sample(text_inputs, text_lengths, sampling_temps=sampling_temps)

    def log_grad_norm(self, grad_norm_dict):
        r"""
        Lightning method to log the grad norm of the model.
        change prog_bar to True to track on progress bar

        Args:
            grad_norm_dict: Dictionary containing current grad norm metrics

        """
        norm_dict = {"grad_norm/" + k: v for k, v in grad_norm_dict.items()}
        self.log_dict(norm_dict, on_step=True, on_epoch=True, prog_bar=False, logger=True)
