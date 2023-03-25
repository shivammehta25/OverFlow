import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler
from einops import rearrange

from src.model.wavegrad import WaveGrad
from src.utilities.functions import fix_len_compatibility


class BaseModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def nparams(self):
        """
        Returns number of trainable parameters of the module.
        """
        num_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                num_params += np.prod(param.detach().cpu().numpy().shape)
        return num_params

    def relocate_input(self, x: list):
        """
        Relocates provided tensors to the same device set for the module.
        """
        device = next(self.parameters()).device
        for i in range(len(x)):
            if isinstance(x[i], torch.Tensor) and x[i].device != device:
                x[i] = x[i].to(device)
        return x


class Mish(BaseModule):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class Upsample(BaseModule):
    def __init__(self, dim):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Downsample(BaseModule):
    def __init__(self, dim):
        super().__init__()
        self.conv = torch.nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Rezero(BaseModule):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


class Block(BaseModule):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(dim, dim_out, 3, padding=1), torch.nn.GroupNorm(groups, dim_out), Mish()
        )

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


class ResnetBlock(BaseModule):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super().__init__()
        self.mlp = torch.nn.Sequential(Mish(), torch.nn.Linear(time_emb_dim, dim_out))

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        if dim != dim_out:
            self.res_conv = torch.nn.Conv2d(dim, dim_out, 1)
        else:
            self.res_conv = torch.nn.Identity()

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class LinearAttention(BaseModule):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = torch.nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w)
        return self.to_out(out)


class Residual(BaseModule):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        output = self.fn(x, *args, **kwargs) + x
        return output


class SinusoidalPosEmb(BaseModule):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class GradLogPEstimator2d(BaseModule):
    def __init__(self, dim, dim_mults=(1, 2, 4), groups=8, n_spks=None, spk_emb_dim=64, n_feats=45, pe_scale=1000):
        super().__init__()
        self.dim = dim
        self.dim_mults = dim_mults
        self.groups = groups
        self.n_spks = n_spks if not isinstance(n_spks, type(None)) else 1
        self.spk_emb_dim = spk_emb_dim
        self.pe_scale = pe_scale

        if n_spks > 1:
            self.spk_mlp = torch.nn.Sequential(
                torch.nn.Linear(spk_emb_dim, spk_emb_dim * 4), Mish(), torch.nn.Linear(spk_emb_dim * 4, n_feats)
            )

        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim, dim * 4), Mish(), torch.nn.Linear(dim * 4, dim))

        self.in_proj = torch.nn.Sequential(torch.nn.Linear(n_feats, dim), torch.nn.Mish())
        dims = [2 + (1 if n_spks > 1 else 0), *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                torch.nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, time_emb_dim=dim),
                        ResnetBlock(dim_out, dim_out, time_emb_dim=dim),
                        Residual(Rezero(LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else torch.nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(
                torch.nn.ModuleList(
                    [
                        ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim),
                        ResnetBlock(dim_in, dim_in, time_emb_dim=dim),
                        Residual(Rezero(LinearAttention(dim_in))),
                        Upsample(dim_in),
                    ]
                )
            )
        self.final_block = Block(dim, dim)
        self.final_conv = torch.nn.Conv2d(dim, 1, 1)
        self.out_proj = torch.nn.Conv1d(dim, n_feats, 1)

    def forward(self, x, mask, mu, t, spk=None):
        max_length_orig = mask.sum(-1).max().item()
        mask_orig = mask
        max_length = fix_len_compatibility(max_length_orig)
        mask = F.pad(mask, (0, max_length - max_length_orig), value=False)
        x = F.interpolate(x, size=max_length)
        mu = F.interpolate(mu, size=max_length)

        if not isinstance(spk, type(None)):
            s = self.spk_mlp(spk)

        t = self.time_pos_emb(t, scale=self.pe_scale)
        t = self.mlp(t)

        if self.n_spks < 2:
            x = torch.stack([mu, x], 1)
        else:
            s = s.unsqueeze(-1).repeat(1, 1, x.shape[-1])
            x = torch.stack([mu, x, s], 1)
        mask = mask.unsqueeze(1)

        x = self.in_proj(rearrange(x * mask, "b x c t -> b x t c"))
        x = rearrange(x, "b x t c -> b x c t")

        hiddens = []
        masks = [mask]
        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)
            x = resnet2(x, mask_down, t)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])
        masks = masks[:-1]
        mask_mid = masks[-1]
        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)

        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            x = attn(x)
            x = upsample(x * mask_up)

        x = self.final_block(x, mask)
        output = self.final_conv(x * mask).squeeze(1)
        mask = mask.squeeze(1)
        output = self.out_proj(output * mask)
        return F.interpolate(output, size=max_length_orig) * mask_orig


def get_noise(t, beta_init, beta_term, cumulative=False):
    if cumulative:
        noise = beta_init * t + 0.5 * (beta_term - beta_init) * (t**2)
    else:
        noise = beta_init + (beta_term - beta_init) * t
    return noise


class Diffusion(BaseModule):
    def __init__(
        self,
        n_feats,
        hidden_channels,
        n_spks=1,
        spk_emb_dim=64,
        beta_min=0.05,
        beta_max=20,
        pe_scale=1000,
        n_timesteps=50,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.dim = hidden_channels
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale
        self.n_timesteps = n_timesteps

        self.estimator = GradLogPEstimator2d(
            hidden_channels, n_spks=n_spks, spk_emb_dim=spk_emb_dim, pe_scale=pe_scale, n_feats=n_feats
        )

    def forward_diffusion(self, x0, mask, mu, t):
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)  # \int_0^t \beta_t
        mean = x0 * torch.exp(-0.5 * cum_noise) + mu * (1.0 - torch.exp(-0.5 * cum_noise))  # eq 23. from eq 3.
        variance = 1.0 - torch.exp(-cum_noise)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, requires_grad=False)
        xt = mean + z * torch.sqrt(variance)
        return xt * mask, z * mask

    @torch.no_grad()
    def reverse_diffusion(self, z, mask, mu, n_timesteps, stoc=False, spk=None):
        h = 1.0 / n_timesteps
        xt = z * mask
        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5) * h) * torch.ones(z.shape[0], dtype=z.dtype, device=z.device)
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = get_noise(time, self.beta_min, self.beta_max, cumulative=False)
            if stoc:  # adds stochastic term
                dxt_det = 0.5 * (mu - xt) - self.estimator(xt, mask, mu, t, spk)  # eq 8 in paper.
                dxt_det = dxt_det * noise_t * h
                dxt_stoc = torch.randn(z.shape, dtype=z.dtype, device=z.device, requires_grad=False)
                dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)
                dxt = dxt_det + dxt_stoc
            else:
                dxt = 0.5 * (mu - xt - self.estimator(xt, mask, mu, t, spk))
                dxt = dxt * noise_t * h
            xt = (xt - dxt) * mask
        return xt

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps=None, stoc=False, spk=None):
        if n_timesteps is None:
            n_timesteps = self.n_timesteps

        z = mu + torch.randn_like(mu, device=mu.device)  # this is xT
        return self.reverse_diffusion(z, mask, mu, n_timesteps, stoc, spk)

    def loss_t(self, x0, mask, mu, t, spk=None):
        xt, z = self.forward_diffusion(x0, mask, mu, t)
        time = t.unsqueeze(-1).unsqueeze(-1)  # t =[0.6215, 0.0191, 0.0391]
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        noise_estimation = self.estimator(
            xt, mask, mu, t, spk
        )  # xt = [3, 80, 172], mask=[3, 1, 172], mu=[3, 80, 172], t=[3]
        noise_estimation *= torch.sqrt(1.0 - torch.exp(-cum_noise))
        loss = torch.sum((noise_estimation + z) ** 2) / (torch.sum(mask) * self.n_feats)
        return loss, xt

    def compute_loss(self, x0, mask, mu, spk=None, offset=1e-5):
        """

        Args:
            x0 : [B, C, T]
            mask: [B, 1, T]
            mu (_type_): [B, C, T]
            spk (_type_, optional): _description_. Defaults to None.
            offset (_type_, optional): _description_. Defaults to 1e-5.

        Returns:
            _type_: _description_
        """
        t = torch.rand(x0.shape[0], dtype=x0.dtype, device=x0.device, requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)
        return self.loss_t(x0, mask, mu, t, spk)


class MyDiffusion(BaseModule):
    __AVAILABLE_SCHEDULERS__ = ["ddim", "ddpm"]
    __AVAILABLE_LOSS__ = ["l1", "l2"]

    def __init__(
        self,
        motion_channels,
        latent_channels,
        scheduler="ddpm",
        beta_schedule="squaredcos_cap_v2",
        loss="l1",
        n_timesteps_train=1000,
        n_timesteps_inference=50,
    ):
        super().__init__()
        self.motion_channels = motion_channels
        self.latent_channels = latent_channels
        self.n_timesteps_train = n_timesteps_train
        self.n_timesteps_inference = n_timesteps_inference

        assert scheduler in self.__AVAILABLE_SCHEDULERS__, f"Scheduler must be one of {self.__AVAILABLE_SCHEDULERS__}"
        assert loss in self.__AVAILABLE_LOSS__, f"Loss must be one of {self.__AVAILABLE_LOSS__}"

        if scheduler == "ddim":
            self.scheduler = DDIMScheduler(num_train_timesteps=n_timesteps_train, beta_schedule=beta_schedule)
        elif scheduler == "ddpm":
            self.scheduler = DDPMScheduler(num_train_timesteps=n_timesteps_train, beta_schedule=beta_schedule)

        self.estimator = WaveGrad(motion_channels, latent_channels)
        if loss == "l1":
            self.loss = nn.L1Loss()
        elif loss == "l2":
            self.loss = nn.MSELoss()
        else:
            raise ValueError("loss must be l1 or l2")

    def update_scheduler_inference_timesteps(self, n_timesteps):
        if n_timesteps is None:
            n_timesteps = self.n_timesteps_inference

        if isinstance(self.scheduler, DDIMScheduler):
            self.scheduler.set_timesteps(num_inference_steps=n_timesteps)

    @torch.inference_mode()
    def forward(self, z, mask, n_timesteps=None, stoc=False, spk=None):
        self.update_scheduler_inference_timesteps(n_timesteps)

        x = torch.randn(z.shape[0], self.motion_channels, z.shape[-1], device=z.device, dtype=z.dtype)

        for i, t in enumerate(self.scheduler.timesteps):
            t = t.unsqueeze(0).to(z.device)
            residual = self.estimator(x, mask, z, t)

            x = self.scheduler.step(residual, t, x).prev_sample

        return x * mask

    def compute_loss(self, x0, mask, z, spk=None):
        noise = torch.randn_like(x0)
        timesteps = torch.randint(0, self.n_timesteps_train, (x0.shape[0],), dtype=torch.long, device=x0.device)
        noisy_x0 = self.scheduler.add_noise(x0, noise, timesteps)
        pred = self.estimator(noisy_x0, mask, z, timesteps)
        loss = self.loss(pred * mask, x0 * mask)
        return loss, pred
