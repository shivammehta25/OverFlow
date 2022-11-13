import pytest
import pytorch_lightning as pl

from src.training_module import TrainingModule


# TODO: remove the skip
@pytest.mark.skip(reason="Will save a checkpoint first then run it")
@pytest.mark.parametrize("checkpoint_path", ["checkpoint_326000.ckpt"])
def test_loading_checkpoint(checkpoint_path):
    model = TrainingModule.load_from_checkpoint(checkpoint_path)
    assert isinstance(model, pl.LightningModule)
