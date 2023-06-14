r"""
training_model.py

This file contains PyTorch Lightning's main module where code
of the main model is implemented
"""
import os
from argparse import Namespace

import lightning as L
import torch
from lightning.pytorch.utilities import grad_norm

from src.model.OverFlow import OverFlow
from src.validation_plotting import log_validation


class TrainingModule(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        if type(hparams) != Namespace:
            hparams = Namespace(**hparams)

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
        if self.hparams.scheduler is None:
            return optimizer
        elif self.hparams.scheduler == "warmup":
            warmup_steps = self.hparams.scheduler_params["warmup_steps"]

            def warm_decay(step):
                if step < warmup_steps:
                    return step / warmup_steps
                return 1.0

            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, warm_decay),
                "interval": "step",  # runs per batch rather than per epoch
                "frequency": 1,
                "name": "warmup",
            }
        else:
            raise NotImplementedError(f"Scheduler {self.hparams.scheduler} not implemented")

        return [optimizer], [scheduler]

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
            float(loss.item()),
            prog_bar=True,
            on_step=True,
            sync_dist=True,
            logger=True,
        )
        self.log(
            "Global_Step",
            float(self.global_step),
            prog_bar=True,
            on_step=True,
            sync_dist=True,
            logger=False,
        )
        self._get_gpu_stats(loss)
        self.log(
            "trainer_stats/MaxGPUMemory", self.max_gpu_usage, logger=True, prog_bar=True, on_step=True, sync_dist=True
        )
        return loss

    def _get_gpu_stats(self, some_tensor):
        free, total = (x / (1024 * 1024) for x in torch.cuda.mem_get_info(some_tensor.device.index))
        self.max_gpu_usage = max(self.max_gpu_usage, total - free)

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

    def on_train_start(self):
        self.on_train_batch_end(None, None, None)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.trainer.is_global_zero and (self.global_step % self.hparams.save_model_checkpoint == 0):
            (
                text_inputs,
                text_lengths,
                mels,
                max_len,
                mel_lengths,
                speaker_id,
            ) = self.get_an_element_of_validation_dataset()
            (
                mel_output,
                state_travelled,
                input_parameters,
                output_parameters,
            ) = self.model.sample(text_inputs[0], text_lengths[0], speaker_id=speaker_id[0])
            mel_output_normalised = self.model.normaliser(mel_output)

            with torch.inference_mode():
                _ = self.model((text_inputs, text_lengths, mels, max_len, mel_lengths, speaker_id))

            log_validation(
                self.logger.experiment,
                self.model,
                mel_output,
                mel_output_normalised,
                state_travelled,
                mels[0],
                input_parameters,
                output_parameters,
                self.global_step,
                self.trainer.datamodule.train_dataloader().dataset.stft,
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
        x, y = self.model.parse_batch(next(iter(self.trainer.datamodule.val_dataloader())))
        (text_inputs, text_lengths, mels, max_len, mel_lengths, speaker_id) = x
        text_inputs = text_inputs[0].unsqueeze(0).to(self.device)
        speaker_id = speaker_id[0].unsqueeze(0).to(self.device)
        text_lengths = text_lengths[0].unsqueeze(0).to(self.device)
        mels = mels[0].unsqueeze(0).to(self.device)
        max_len = torch.max(text_lengths).data
        mel_lengths = mel_lengths[0].unsqueeze(0).to(self.device)
        # Sometimes in a batch the element which has the maximum mel len
        # is not the same as the element which has the maximum text len.
        # This prevent the model to break down when plotting validation.
        mels = mels[:, :, : mel_lengths.item()]

        return text_inputs, text_lengths, mels, max_len, mel_lengths, speaker_id

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
    def sample(self, text_inputs, text_lengths=None, speaker_id=None, sampling_temp=1.0):
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
        return self.model.sample(text_inputs, text_lengths, speaker_id=speaker_id, sampling_temp=sampling_temp)

    def on_before_optimizer_step(self, optimizer):
        r"""
        Lightning method to log the grad norm of the model.
        change prog_bar to True to track on progress bar

        Args:
            grad_norm_dict: Dictionary containing current grad norm metrics

        """
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms, on_step=True, on_epoch=True, prog_bar=False, logger=True)
