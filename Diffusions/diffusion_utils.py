import numpy as np
import torch


def normal_kl(mean1, logvar1, mean2, logvar2):

    kl = -0.5 * (
        logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + (mean1 - mean2) ** 2 * torch.exp(-logvar2)
        - 1.0
    )

    return kl


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    """
    betas as HPO
    """
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float32)

    warmup_time = int(num_diffusion_timesteps * warmup_frac)

    betas[:warmup_time] = np.linspace(
        beta_start, beta_end, warmup_time, dtype=np.float64
    )

    return betas


class GaussianDiffusion:

    def __init__(self, *, betas, loss_type, dt_type=torch.float32):

        self.loss_type = loss_type #??

        self.np_betas = betas = betas.astype(np.float64)

        assert (betas > 0).all() and (betas <= 1).all()

        (timesteps, ) = betas.shape

        self.num_timesteps = int(timesteps)

        alphas = 1 - betas 

        alphas_cumprod = np.cumprod(alphas, axis=0)

        alphas_cumprod_prev = np.append(1, alphas_cumprod[:-1]) # Beta = 0

        self.betas = torch.tensor(betas, dtype=dt_type)

        self.alphas_cumprod = torch.tensor(alphas_cumprod, dtype=dt_type)

        self.alphas_cumprod_pred = torch.tensor(alphas_cumprod_prev, dtype=dt_type)

        self.sqrt_one_minus_alphas_cumprod = torch.tensor(
            np.sqrt(1.0 - alphas_cumprod), dtype=dt_type
        )
        self.log_one_minus_alphas_cumprod = torch.tensor(
            np.log(1.0 - alphas_cumprod), dtype=dt_type
        )
        self.sqrt_recip_alphas_cumprod = torch.tensor(
            np.sqrt(1.0 / alphas_cumprod), dtype=dt_type
        )
        self.sqrt_recipm1_alphas_cumprod = torch.tensor(
            np.sqrt(1.0 / alphas_cumprod - 1), dtype=dt_type
        )

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        self.posterior_variance = torch.tensor(posterior_variance, dtype=dt_type)

        self.posterior_log_variance_clipped = torch.tensor(
            np.log(np.maximum(posterior_variance, 1e-20)), dtype=dt_type
        )

        self.posterior_mean_coef1 = torch.tensor(
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
            dtype=dt_type,
        )
        
        self.posterior_mean_coef2 = torch.tensor(
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
            dtype=dt_type,
        )



