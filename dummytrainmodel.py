import argparse
import os
from argparse import Namespace

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import rearrange
from pytorch_lightning.loggers import TensorBoardLogger

from src.data_module import LightningLoader
from src.hparams import create_hparams
from src.model.Decoder import MotionDecoder
from src.model.OverFlow import OverFlow
from src.model.transformer import FFTransformer
from src.utilities.functions import fix_len_compatibility
from src.utilities.plotting import generate_motion_visualization


class TestTrainingModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        if type(hparams) != Namespace:
            hparams = Namespace(**hparams)
        self.motion_visualizer = hparams.motion_visualizer
        del hparams.motion_visualizer
        self.save_hyperparameters(hparams)
        hparams.logger = self.logger
        self.max_gpu_usage = 0
        self.mel_proj = torch.nn.Conv1d(80, 384, 1)
        self.mel_encoder = FFTransformer(**hparams.motion_decoder_param["transformer"])
        self.mel_outproj = torch.nn.Conv1d(384, 80, 1)
        self.decoder_motion = MotionDecoder(hparams, hparams.motion_decoder_type)

    def run_forward(self, mels, mel_lengths, motion=None, reverse=False):
        mels = self.mel_proj(mels)
        mels = rearrange(self.mel_encoder(mels, mel_lengths)[0], "b t c -> b c t")
        mels = self.mel_outproj(mels)
        motion_decoder_output = self.decoder_motion(mels, mel_lengths, motion, reverse=reverse)
        return motion_decoder_output

    def forward(self, batch):
        r"""
        Forward pass of the model

        Args:
            x (Any): input to the forward function

        Returns:
            output (Any): output of the forward function
        """
        (_, _, mels, motion, mel_lengths), _ = OverFlow.parse_batch(batch)
        motion_decoder_output = self.run_forward(mels, mel_lengths, motion, reverse=False)
        return motion_decoder_output["loss"]

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
        motion_loss = self(train_batch)
        self.log(
            "loss/train_motion",
            motion_loss.item(),
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
        self.set_gpu_stats(motion_loss)
        return motion_loss

    def set_gpu_stats(self, some_tensor):
        free, total = (x / (1024 * 1024) for x in torch.cuda.mem_get_info(some_tensor.device.index))
        current = total - free
        self.max_gpu_usage = max(self.max_gpu_usage, current)
        self.log(
            "trainer_stats/MaxGPUMemory", self.max_gpu_usage, prog_bar=True, logger=True, on_step=True, sync_dist=True
        )
        self.log("trainer_stats/CurrentGPUMemory", current, logger=True, on_step=True, sync_dist=True)

    def validation_step(self, val_batch, batch_idx):
        r"""
        Validation step

        Args:
            val_batch (Any): output depends what you are returning from the train loop
            batch_idx (): batch index
        """
        motion_loss = self(val_batch)
        self.log(
            "loss/val_motion",
            motion_loss.item(),
            sync_dist=True,
            logger=True,
        )
        return motion_loss

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

            max_z_len = fix_len_compatibility(mels.shape[2])
            mels = F.pad(mels, (0, max_z_len - mels.shape[-1]))
            motion_decoder_output = self.run_forward(mels, mel_lengths, reverse=True)
            motion_output = motion_decoder_output["motions"]
            motion_output = rearrange(motion_output, "b t c -> b c t")
            motion_output = self.hparams.motion_normaliser.inverse_normalise(motion_output)
            stft_module = self.train_dataloader().dataset.stft
            target_audio, sr = stft_module.griffin_lim(mels[0].unsqueeze(0))
            logger = self.logger.experiment
            if self.global_step > 0:
                generate_motion_visualization(
                    target_audio,
                    f"{logger.log_dir}/input_{self.global_step}.wav",
                    motion_output.squeeze(0).cpu().numpy().T,
                    f"{logger.log_dir}/input_{self.global_step}.mp4",
                    self.motion_visualizer,
                    f"{logger.log_dir}/input_{self.global_step}.bvh",
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
        x, y = OverFlow.parse_batch(next(iter(self.val_dataloader())))
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
    def sample(self, text_inputs, text_lengths=None, sampling_temp=0.0):
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
        return self.model.sample(text_inputs, text_lengths, sampling_temp=sampling_temp)

    def log_grad_norm(self, grad_norm_dict):
        r"""
        Lightning method to log the grad norm of the model.
        change prog_bar to True to track on progress bar

        Args:
            grad_norm_dict: Dictionary containing current grad norm metrics

        """
        norm_dict = {"grad_norm/" + k: v for k, v in grad_norm_dict.items()}
        self.log_dict(norm_dict, on_step=True, on_epoch=True, prog_bar=False, logger=True)


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
    args = parser.parse_args()

    hparams = create_hparams()
    hparams.run_name = "test"
    hparams.tensorboard_log_dir = "dummy_tb_logs"
    hparams.checkpoint_dir = "dummy_checkpoint"
    hparams.motion_decoder_type = "mydiffusion"
    hparams.save_model_checkpoint = 1000
    hparams.learning_rate = 3e-4
    hparams.gpus = [0]  # Run with CUDA VISIBLE DEVICES
    # hparams.batch_size=32

    data_module = LightningLoader(hparams)
    model = TestTrainingModule(hparams)
    logger = TensorBoardLogger(hparams.tensorboard_log_dir, name=hparams.run_name)

    trainer = pl.Trainer(
        resume_from_checkpoint=args.checkpoint_path,
        gpus=hparams.gpus,
        logger=logger,
        log_every_n_steps=10,
        flush_logs_every_n_steps=1,
        val_check_interval=hparams.val_check_interval,
        gradient_clip_val=hparams.grad_clip_thresh,
        max_epochs=hparams.max_epochs,
        stochastic_weight_avg=hparams.stochastic_weight_avg,
        precision=hparams.precision,
        track_grad_norm=2,
        limit_val_batches=10,
    )

    trainer.fit(model, data_module)
