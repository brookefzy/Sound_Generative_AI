import yaml
import torch
import torchaudio
import importlib


def infer():
    cfg = yaml.safe_load(open('config_diffusion.yaml'))

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Inference on device: {device}")

    mod_cfg = cfg['model']
    mod_name = mod_cfg['type']
    assert mod_name == 'diffusion', "This script only supports diffusion models"

    ModelClass = getattr(importlib.import_module(f"models.{mod_name}"), mod_cfg['class'])
    params = mod_cfg.get('params', {})
    length = cfg['sample_rate'] * cfg.get('duration', 1)
    params.setdefault('input_length', length)

    model = ModelClass(**params).to(device)
    model.load_state_dict(torch.load(f"{mod_name}_final.pt", map_location=device))
    model.eval()

    num_samples = cfg['inference']['num_samples']
    steps = cfg['inference'].get('steps', 1)
    with torch.no_grad():
        x = torch.randn(num_samples, 1, params['input_length'], device=device)
        for _ in range(steps):
            t = torch.zeros(num_samples, dtype=torch.long, device=device)
            noise = model(x, t)
            x = x - noise

        for i, wav in enumerate(x):
            torchaudio.save(f"sample_{i}.wav", wav.cpu(), cfg['sample_rate'])


if __name__ == '__main__':
    infer()
