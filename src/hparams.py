r"""
hparams.py

Hyper Parameters for the experiment
"""
import os
from argparse import Namespace

import joblib as jl
import torch

from src.utilities.data import Normalise
from src.utilities.text import symbols


def create_hparams(generate_parameters=False):
    """
    Model hyperparemters
    Args:
        generate_paramters: default False
            Only used when you run data_properties.py

    Returns:
        hparams (Namespace)
    """

    # root_folder = pathlib.Path(__file__).parent.parent.resolve()
    # data_parameters_filename =  / "data_parameters.pt"
    data_parameters_filename = "data_parameters.pt"

    if not generate_parameters:
        if not os.path.exists(data_parameters_filename):
            raise FileNotFoundError(
                "Data Normalizing file not found! " + 'Run "python generate_data_properties.py" first'
            )

        data_properties = torch.load(data_parameters_filename)
        mel_mean = data_properties["mel_mean"].item()
        mel_std = data_properties["mel_std"].item()

        motion_mean = data_properties["motion_mean"].item()
        motion_std = data_properties["motion_std"].item()

        init_transition_prob = data_properties["init_transition_prob"]
        go_token_init_value = data_properties["go_token_init_value"]
        mel_normaliser = Normalise(mel_mean, mel_std)
        motion_normaliser = Normalise(motion_mean, motion_std)
    else:
        # Must be while generating data properties
        mel_normaliser = None
        motion_normaliser = None
        init_transition_prob = None
        go_token_init_value = None

    hparams = Namespace(
        ################################
        # Experiment Parameters        #
        ################################
        run_name="ConformerEvery4NoConv",
        gpus=[3],
        max_epochs=50000,
        val_check_interval=100,
        save_model_checkpoint=500,
        gradient_checkpoint=True,
        seed=1234,
        checkpoint_dir="checkpoints",
        tensorboard_log_dir="tb_logs",
        gradient_accumulation_steps=1,
        precision=16,
        # Placeholder to use it later while loading model
        logger=None,
        run_tests=False,
        warm_start=False,
        ignore_layers=["model.embedding.weight"],
        ################################
        # Data Parameters             #
        ################################
        batch_size=12,
        load_mel_from_disk=False,
        training_files="data/filelists/cormac_train.txt",
        validation_files="data/filelists/cormac_val.txt",
        text_cleaners=["english_cleaners"],
        # training_files="data/filelists/ljs_audio_text_train_filelist.txt",
        # validation_files="data/filelists/ljs_audio_text_val_filelist.txt",
        phonetise=True,
        add_blank=True,
        num_workers=40,
        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,
        ################################
        # Motion Parameters            #
        ################################
        n_motion_joints=45,
        frame_rate_reduction_factor=4,
        motion_fileloc="data/cormac/processed_sm0_0_86fps",
        motion_filename_ext="expmap_86.1328125fps.pkl",
        motion_visualizer=jl.load("data/cormac/processed_sm0_0_86fps/data_pipe.expmap_86.1328125fps.sav"),
        ################################
        # Data Properties              #
        ################################
        mel_normaliser=mel_normaliser,
        motion_normaliser=motion_normaliser,
        go_token_init_value=go_token_init_value,
        init_transition_probability=init_transition_prob,
        init_mean=0.0,
        init_std=1.0,
        data_mean=0,
        data_std=0,
        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),
        ################################
        # Encoder parameters           #
        ################################
        encoder_type="transformer",
        encoder_params={
            "conv": {"kernel_size": 5, "n_convolutions": 3, "hidden_channels": 512, "state_per_phone": 2},
            "transformer": {
                "hidden_channels": 384,
                "n_layer": 6,
                "n_head": 1,
                "d_head": 64,
                "d_inner": 1024,
                "kernel_size": 3,
                "dropout": 0.1,
                "dropatt": 0.1,
                "dropemb": 0.0,
                "embed_input": False,
                "pre_lnorm": True,
                "rel_attention": False,
                "rel_window_size": 10,
            },
        },
        ################################
        # HMM Parameters               #
        ################################
        n_frames_per_step=1,  # AR Order
        train_go=False,
        variance_floor=0.001,
        data_dropout=0,
        data_dropout_while_eval=True,
        data_dropout_while_sampling=True,
        predict_means=False,
        base_sampling_temperature=0.0,
        max_sampling_time=1000,
        deterministic_transition=True,
        duration_quantile_threshold=0.3,
        ################################
        # Prenet parameters            #
        ################################
        prenet_n_layers=2,
        prenet_dim=256,
        prenet_dropout=0.5,
        prenet_dropout_while_eval=True,
        ################################
        # Decoder RNN parameters       #
        ################################
        post_prenet_rnn_dim=1024,
        ################################
        # Decoder Parameters           #
        ################################
        parameternetwork=[1024],
        ################################
        # Decoder Flow Parameters      #
        ################################
        flow_hidden_channels=150,
        kernel_size_dec=5,
        dilation_rate=1,
        n_blocks_dec=12,
        n_block_layers=4,
        p_dropout_dec_mel=0.05,
        p_dropout_dec_motion=None,
        n_split=4,
        n_sqz=2,
        sigmoid_scale=False,
        gin_channels=0,
        ################################
        # Decoder Transformer Parameters#
        ################################
        motion_decoder_type="flow",
        motion_decoder_param={
            "transformer": {
                "hidden_channels": 384,
                "n_layer": 6,
                "n_head": 1,
                "d_head": 64,
                "d_inner": 1024,
                "kernel_size": 3,
                "dropout": 0.1,
                "dropatt": 0.1,
                "dropemb": 0.0,
                "embed_input": False,
                "pre_lnorm": True,
                "rel_attention": False,
                "rel_window_size": 10,
            },
            "conformer": {
                "hidden_channels": 384,
                "d_head": 64,
                "n_layer": 6,
                "n_head": 1,
                "ff_mult": 4,
                "conv_expansion_factor": 2,
                "dropout": 0.1,
                "dropconv": 0.1,
                "dropatt": 0.1,
                "conv_kernel_size": 21,
            },
            "flow": {
                "hidden_channels": 384,
                "flow_hidden_channels": 150,
                "kernel_size_dec": 5,
                "dilation_rate": 1,
                "n_blocks_dec": 12,
                "n_block_layers": 4,
                "p_dropout_dec": 0.05,
                "n_split": 4,
                "n_sqz": 2,
                "sigmoid_scale": False,
                "gin_channels": 0,
            },
        },
        ################################
        # Optimization Hyperparameters #
        ################################
        learning_rate=1e-3,
        weight_decay=1e-6,
        grad_clip_thresh=5.0,
        stochastic_weight_avg=False,
        optimizer_params={
            "scheduler": None,
            "noam_params": {"warmup": 1000},
        },
    )

    return hparams
