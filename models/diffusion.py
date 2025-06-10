# models/diffusion.py
import math
import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = nn.functional.pad(emb, (0,1))
        return emb

class ResBlock(nn.Module):
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, channels)

    def forward(self, x, emb):
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.emb_proj(emb)[:, :, None]
        h = self.conv2(self.act2(self.norm2(h)))
        return x + h

class DiffusionModel(nn.Module):
    def __init__(self, input_length, base_channels=64, num_res_blocks=2):
        super().__init__()
        emb_dim = base_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        self.init_conv = nn.Conv1d(1, base_channels, 3, padding=1)
        self.down1 = nn.ModuleList([ResBlock(base_channels, emb_dim) for _ in range(num_res_blocks)])
        self.downsample1 = nn.Conv1d(base_channels, base_channels * 2, 4, 2, 1)
        self.down2 = nn.ModuleList([ResBlock(base_channels * 2, emb_dim) for _ in range(num_res_blocks)])
        self.downsample2 = nn.Conv1d(base_channels * 2, base_channels * 4, 4, 2, 1)
        self.mid = nn.ModuleList([ResBlock(base_channels * 4, emb_dim) for _ in range(num_res_blocks)])
        self.up2_conv = nn.ConvTranspose1d(base_channels * 4, base_channels * 2, 4, 2, 1)
        self.up2 = nn.ModuleList([ResBlock(base_channels * 2, emb_dim) for _ in range(num_res_blocks)])
        self.up1_conv = nn.ConvTranspose1d(base_channels * 2, base_channels, 4, 2, 1)
        self.up1 = nn.ModuleList([ResBlock(base_channels, emb_dim) for _ in range(num_res_blocks)])
        self.final_conv = nn.Conv1d(base_channels, 1, 3, padding=1)

    def forward(self, x, t):
        emb = self.time_embed(t)
        h = self.init_conv(x)
        hs = []
        for block in self.down1:
            h = block(h, emb)
        hs.append(h)
        h = self.downsample1(h)
        for block in self.down2:
            h = block(h, emb)
        hs.append(h)
        h = self.downsample2(h)
        for block in self.mid:
            h = block(h, emb)
        h = self.up2_conv(h)
        h = h + hs.pop()
        for block in self.up2:
            h = block(h, emb)
        h = self.up1_conv(h)
        h = h + hs.pop()
        for block in self.up1:
            h = block(h, emb)
        return self.final_conv(h)

