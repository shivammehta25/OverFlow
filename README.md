# OverFlow: Putting flows on top of neural transducers for better TTS
##### [Shivam Mehta][shivam_profile], [Éva Székely][eva_profile], [Jonas Beskow][jonas_profile], and [Gustav Eje Henter][gustav_profile]
---

[paper_link]: https://shivammehta25.github.io/OverFlow/

[github_link]: https://github.com/shivammehta25/OverFlow
[shivam_profile]: https://www.kth.se/profile/smehta
[ambika_profile]: https://www.kth.se/profile/kirkland
[harm_profile]: https://www.kth.se/profile/lameris
[eva_profile]: https://www.kth.se/profile/szekely
[jonas_profile]: https://www.kth.se/profile/beskow
[gustav_profile]: https://people.kth.se/~ghe/
[demo_page]: https://shivammehta25.github.io/OverFlow/
[ljspeech_link]: https://keithito.com/LJ-Speech-Dataset/
[github_new_issue_link]: https://github.com/shivammehta25/OverFlow/issues/new
[docker_install_link]: https://docs.docker.com/get-docker/
[tacotron2_link]: https://github.com/NVIDIA/tacotron2
[glow_tts_link]: https://github.com/jaywalnut310/glow-tts
[pretrained_model_link_female]: https://github.com/shivammehta25/OverFlow/releases/download/Neural-HMM/OverFlow-Female.ckpt
[pretrained_model_link_male]: https://github.com/shivammehta25/OverFlow/releases/download/Neural-HMM/OverFlow-Male.ckpt
[hifigan_all]: https://drive.google.com/drive/folders/1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y
[hifigan_t2]: https://drive.google.com/drive/folders/1dqpUYEYF_hH7T0rII9_VQbps45FvNBqf
[pytorch_lightning_link]: https://github.com/PyTorchLightning/pytorch-lightning


This is the official code repository for the paper "[OverFlow: Putting flows on top of neural transducers for better TTS][paper_link]". For audio examples, visit our [demo page][demo_page]. [pre-trained model (female)][pretrained_model_link_female] and [pre-trained model (male)][pretrained_model_link_male] are also available.

<!-- ![Synthesising from Neural-HMM](docs/images/model_video.gif) -->

## Setup and training using LJ Speech
1. Download and extract the [LJ Speech dataset][ljspeech_link]. Place it in the `data` folder such that the directory becomes `data/LJSpeech-1.1`. Otherwise update the filelists in `data/filelists` accordingly.
2. Clone this repository ```git clone https://github.com/shivammehta25/OverFlow.git```
   * If using single GPU checkout the branch ```gradient_checkpointing``` it will help to fit bigger batch size during training.
   * Use `git clone --single-branch -b gradient_checkpointing https://github.com/shivammehta25/OverFlow.git` for that.
3. Initalise the submodules ```git submodule init; git submodule update```
4. Make sure you have [docker installed][docker_install_link] and running.
    * It is recommended to use Docker (it manages the CUDA runtime libraries and Python dependencies itself specified in Dockerfile)
    * Alternatively, If you do not intend to use Docker, you can use pip to install the dependencies using ```pip install -r requirements.txt```
5. Run ``bash start.sh`` and it will install all the dependencies and run the container.
6. Check `src/hparams.py` for hyperparameters and set GPUs.
    1. For multi-GPU training, set GPUs to ```[0, 1 ..]```
    2. For CPU training (not recommended), set GPUs to an empty list ```[]```
    3. Check the location of transcriptions
7. Once your filelists and hparams are updated run `python generate_data_properties.py` to generate `data_parameters.pt` for your dataset (the default `data_parameters.pt` is available for LJSpeech in the repository).
8. Run ```python train.py``` to train the model.
    1. Checkpoints will be saved in the `hparams.checkpoint_dir`.
    2. Tensorboard logs will be saved in the `hparams.tensorboard_log_dir`.
9. To resume training, run ```python train.py -c <CHECKPOINT_PATH>```

## Synthesis
1. Download our [pre-trained LJ Speech model][pretrained_model_link_female].
    - Alternatively, you can also use a [pre-trained RyanSpeech model][pretrained_model_link_male].
2. Download HiFi gan pretrained [HiFiGAN model][hifigan_all].
    - We recommend using [fine tuned][hifigan_t2] on Tacotron2 if you cannot finetune on OverFlow.
3. Run jupyter notebook and open ```synthesis.ipynb```.


## Miscellaneous
### Mixed-precision training or full-precision training
* In ```src.hparams.py``` change ```hparams.precision``` to ```16``` for mixed precision and ```32``` for full precision.
### Multi-GPU training or single-GPU training
* Since the code uses PyTorch Lightning, providing more than one element in the list of GPUs will enable multi-GPU training. So change ```hparams.gpus``` to ```[0, 1, 2]``` for multi-GPU training and single element ```[0]``` for single-GPU training.


### Known issues/warnings
#### Torchmetric error on RTX 3090
* If you encoder this error message ```ImportError: cannot import name 'get_num_classes' from 'torchmetrics.utilities.data' (/opt/conda/lib/python3.8/site-packages/torchmetrics/utilities/data.py)```
* Update the requirement.txt file with these requirements:
```python
torch==1.11.0a0+b6df043
--extra-index-url https://download.pytorch.org/whl/cu113
torchmetrics==0.6.0
```

## Support
If you have any questions or comments, please open an [issue][github_new_issue_link] on our GitHub repository.

## Citation information
If you use or build on our method or code for your research, please cite our paper:
```
Coming Soon
```
## Acknowledgements
The code implementation is based on [Nvidia's implementation of Tacotron 2][tacotron2_link], [Glow TTS][glow_tts_link] and uses [PyTorch Lightning][pytorch_lightning_link] for boilerplate-free code.
