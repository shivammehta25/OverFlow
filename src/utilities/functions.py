"""functions.py.

File for custom utility functions to improve numerical precision
"""
import torch


def log_clamped(x, eps=1e-04):
    clamped_x = torch.clamp(x, min=eps)
    return torch.log(clamped_x)


def inverse_sigmod(x):
    r"""
    Inverse of the sigmoid function
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    return log_clamped(x / (1.0 - x))


def inverse_softplus(x):
    r"""
    Inverse of the softplus function
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    return log_clamped(torch.exp(x) - 1.0)


def logsumexp(x, dim):
    r"""
    Differentiable LogSumExp: Does not creates nan gradients
        when all the inputs are -inf
    Args:
        x : torch.Tensor -  The input tensor
        dim: int - The dimension on which the log sum exp has to be applied
    """

    m, _ = x.max(dim=dim)
    mask = m == -float("inf")

    s = (x - m.masked_fill_(mask, 0).unsqueeze(dim=dim)).exp().sum(dim=dim)
    return s.masked_fill_(mask, 1).log() + m.masked_fill_(mask, -float("inf"))


def log_domain_matmul(log_a, log_b):
    r"""
    Multiply two matrices in log domain
    Args:
        log_a : m x n
        lob_b : n x p
        out : m x p
    Returns:
        Computes output_{i, j} = logsumexp_k [ log_A_{i, k} + log_B{k, j} ]
    """

    m, n, p = log_a.shape[0], log_a.shape[1], log_b.shape[1]

    # Dimensions must be same to add

    # Expand A to the p size
    log_A_expanded = log_a.unsqueeze(2).expand((m, n, p))
    # Expand B to m size
    log_B_expanded = log_b.unsqueeze(0).expand((m, n, p))
    # These expansion will result in addition

    elementwise_sum = log_A_expanded + log_B_expanded

    out = logsumexp(elementwise_sum, 1)

    return out


def masked_softmax(vec, dim=0):
    r"""Outputs masked softmax"""
    mask = ~torch.eq(vec, 0)
    exps = torch.exp(vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True)
    softmax_values = masked_exps / masked_sums
    return softmax_values


def masked_log_softmax(vec, dim=0):
    r"""Outputs masked log_softmax"""
    mask = ~torch.eq(vec, 0)
    exps = torch.exp(vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True)
    softmax_values = masked_exps / masked_sums
    idx = softmax_values != 0
    softmax_values[idx] = torch.log(softmax_values[idx])
    return softmax_values


def get_mask_from_len(lengths, device="cpu", out_tensor=None):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=device) if out_tensor is None else torch.arange(0, max_len, out=out_tensor)
    mask = ids < lengths.unsqueeze(1)
    return mask


def get_mask_for_last_item(lengths, device="cpu", out_tensor=None):
    """Returns n-1 mask for the last item in the sequence.

    Args:
        lengths (torch.IntTensor): lengths in a batch
        device (str, optional): Defaults to "cpu".
        out_tensor (torch.Tensor, optional): uses the memory of a specific tensor.
            Defaults to None.
    """
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=device) if out_tensor is None else torch.arange(0, max_len, out=out_tensor)
    mask = ids == lengths.unsqueeze(1) - 1
    return mask


######################################################
# Begin Glow-TTS methods
# https://github.com/jaywalnut310/glow-tts
######################################################


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def squeeze(x, x_mask=None, n_sqz=2):
    b, c, t = x.size()

    t = (t // n_sqz) * n_sqz
    x = x[:, :, :t]
    x_sqz = x.view(b, c, t // n_sqz, n_sqz)
    x_sqz = x_sqz.permute(0, 3, 1, 2).contiguous().view(b, c * n_sqz, t // n_sqz)

    if x_mask is not None:
        x_mask = x_mask[:, :, n_sqz - 1 :: n_sqz]
    else:
        x_mask = torch.ones(b, 1, t // n_sqz).to(device=x.device, dtype=x.dtype)
    return x_sqz * x_mask, x_mask


def unsqueeze(x, x_mask=None, n_sqz=2):
    b, c, t = x.size()

    x_unsqz = x.view(b, n_sqz, c // n_sqz, t)
    x_unsqz = x_unsqz.permute(0, 2, 3, 1).contiguous().view(b, c // n_sqz, t * n_sqz)

    if x_mask is not None:
        x_mask = x_mask.unsqueeze(-1).repeat(1, 1, 1, n_sqz).view(b, 1, t * n_sqz)
    else:
        x_mask = torch.ones(b, 1, t * n_sqz).to(device=x.device, dtype=x.dtype)
    return x_unsqz * x_mask, x_mask


######################################################
# End Glow TTS Methods
######################################################
