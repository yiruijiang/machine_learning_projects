import torch
import torch.nn as nn


# Loss function
def elbo_loss(recon_x, x, mu, log_var, weight=1):

    # BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="mean")

    BCE = nn.functional.cross_entropy(recon_x, x.view(-1, 784) * 5, reduction="sum")

    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    # print(BCE, KLD)

    return BCE * weight + KLD
