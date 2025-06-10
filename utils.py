# utils.py
# =========
import os
import torchaudio
import torch

def load_dataset(data_dir, sample_rate):
    """
    Load all WAV files in data_dir, resample to sample_rate, return tensor [N, 1, time_steps]
    """
    files = [os.path.join(data_dir, f)
             for f in os.listdir(data_dir) if f.lower().endswith('.wav')]
    waves = []
    for path in files:
        wav, sr = torchaudio.load(path)
        if wav.ndim > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != sample_rate:
            wav = torchaudio.transforms.Resample(sr, sample_rate)(wav)
        waves.append(wav)
    return torch.stack(waves)  # shape: [N, 1, time_steps]