import yaml
import torch
import torchaudio
import importlib


def infer():
    cfg = yaml.safe_load(open('config.yaml'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Inference on device: {device}")

    mod_cfg = cfg['model']
    mod_name = mod_cfg['type']
    assert mod_name == 'wavegan', "This script only supports waveGAN models"

    ModelClass = getattr(importlib.import_module(f"models.{mod_name}"), mod_cfg['class'])
    params = mod_cfg.get('params', {})
    length = cfg['sample_rate'] * cfg.get('duration', 1)
    params.setdefault('output_length', length)

    model = ModelClass(**params).to(device)
    model.load_state_dict(torch.load(f"{mod_name}_final.pt", map_location=device))
    model.eval()

    latent_dim = params.get('latent_dim', 100)
    num_samples = cfg['inference']['num_samples']

    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim, 1, device=device)
        out = model(z)
        for i, wav in enumerate(out):
            torchaudio.save(f"sample_{i}.wav", wav.cpu(), cfg['sample_rate'])


if __name__ == '__main__':
    infer()
