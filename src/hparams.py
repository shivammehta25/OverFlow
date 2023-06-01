r"""
hparams.py

Hyper Parameters for the experiment
"""
import os
from argparse import Namespace

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
        mean = data_properties["data_mean"].item()
        std = data_properties["data_std"].item()
        init_transition_prob = data_properties["init_transition_prob"]
        go_token_init_value = data_properties["go_token_init_value"]
        normaliser = Normalise(mean, std)
    else:
        # Must be while generating data properties
        normaliser = None
        init_transition_prob = None
        go_token_init_value = None
        mean = None
        std = None

    hparams = Namespace(
        ################################
        # Experiment Parameters        #
        ################################
        run_name="VCTK",
        gpus=[2],
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
        num_speakers=109,  # 1 for LJ-Speech
        ################################
        # Data Parameters             #
        ################################
        batch_size=18,
        load_mel_from_disk=False,
        training_files="data/filelists/vctk_train_filelist.txt",
        validation_files="data/filelists/vctk_val_filelist.txt",
        text_cleaners=["english_cleaners"],
        # phonetise=False,
        # training_files="data/filelists/ljs_audio_text_train_filelist.txt",
        # validation_files="data/filelists/ljs_audio_text_val_filelist.txt",
        # text_cleaners=["english_cleaners"],
        phonetise=True,
        add_blank=True,
        num_workers=32,
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
        # Data Properties              #
        ################################
        normaliser=normaliser,
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
        encoder_type="xformer",
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
            "hfT5": {
                "hidden_channels": 1024,
                "n_layer": 6,
                "n_head": 1,
                "d_head": 64,
                "d_inner": 1024,
                "dropout": 0.1,
                "feed_forward_proj": "gated-gelu",
            },
            "xformer": {
                "block_type": "encoder",
                "hidden_channels": 384,  # will be changed to dim model later
                "num_layers": 6,
                "residual_norm_style": "pre",  # Optional, pre/post
                # "position_encoding_config": {
                #     "name": "rotary",  # whatever position encodinhg makes sens
                # },
                "multi_head_config": {
                    "num_heads": 1,
                    # "dim_model": 64,
                    "use_rotary_embeddings": True,
                    "attention": {
                        "name": "scaled_dot_product",
                        "dropout": 0.1,
                        "causal": False,
                    },
                },
                "feedforward_config": {
                    "name": "MLP",
                    "dropout": 0.1,
                    "activation": "gelu",
                    "hidden_layer_multiplier": 4,
                    # "dim_model": 1024,
                },
            },
        },
        ################################
        # HMM Parameters               #
        ################################
        n_frames_per_step=1,  # AR Order
        train_go=True,
        variance_floor=0.001,
        data_dropout=0,
        data_dropout_while_eval=True,
        data_dropout_while_sampling=False,
        predict_means=True,
        max_sampling_time=1000,
        deterministic_transition=True,
        duration_quantile_threshold=0.5,
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
        p_dropout_dec=0.05,
        n_split=4,
        n_sqz=2,
        sigmoid_scale=False,
        gin_channels=384,
        ################################
        # Optimization Hyperparameters #
        ################################
        learning_rate=1e-3,
        weight_decay=1e-6,
        grad_clip_thresh=5.0,
        stochastic_weight_avg=False,
        scheduler=None,
        scheduler_params={
            "warmup_steps": 4000,
        },
    )

    return hparams
