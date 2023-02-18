from pytorch_lightning import Callback


class GradNormCallback(Callback):
    """
    Logs the gradient norm.
    """

    def on_after_backward(self, trainer, model):  # pylint: disable=unused-argument
        model.log("training_stats/grad_norm", self.gradient_norm(model))

    def gradient_norm(self, model):  # pylint: disable=no-self-use
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        return total_norm
