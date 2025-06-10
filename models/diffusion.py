# models/diffusion.py
import torch
import torch.nn as nn

class DiffusionModel(nn.Module):
    def __init__(self, input_length, **kwargs):
        super().__init__()
        # TODO: integrate an existing diffusion backbone (e.g., UNet1D)
        raise NotImplementedError("Add diffusion model backbone here.")

    def forward(self, x, t):
        pass