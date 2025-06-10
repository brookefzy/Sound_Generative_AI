# inference.py
import yaml, torch, torchaudio, importlib

def infer():
    cfg = yaml.safe_load(open('config_diffusion.yaml'))

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Inference on device: {device}")

    mod_cfg = cfg['model']; mod_name = mod_cfg['type']
    ModelClass = getattr(importlib.import_module(f"models.{mod_name}"), mod_cfg['class'])
    params = mod_cfg.get('params', {})
    length = cfg['sample_rate'] * cfg.get('duration', 1)
    params.setdefault('input_length', length)
    model = ModelClass(**params).to(device)
    # Load checkpoint and verify configuration matches training
    checkpoint = torch.load(f"{mod_name}_final.pt", map_location=device)
    if mod_name == 'vae':
        # encoder reduces length by factor 4, with 64 channels
        ck_enc_size = checkpoint['fc_mu.weight'].shape[1]
        expected_enc_size = (params['input_length'] // 4) * 64
        if ck_enc_size != expected_enc_size:
            raise ValueError(
                f"Configuration mismatch: checkpoint was trained with enc_size={ck_enc_size}, "
                f"but current config enc_size={expected_enc_size}. "
                f"Please set sample_rate and duration in config.yaml to match your training settings."
            )
    model.load_state_dict(checkpoint)
    model.eval()

    with torch.no_grad():
        latent_dim = params.get('latent_dim', 100)
        batch = cfg['inference']['num_samples']
        z = torch.randn(batch, latent_dim, 1, device=device)
        out = model(z) if mod_name != 'vae' else model.decoder(model.fc_decode(torch.randn(batch, latent_dim, device=device)))
        for i, wav in enumerate(out):
            torchaudio.save(f"sample_{i}.wav", wav.cpu(), cfg['sample_rate'])

if __name__ == '__main__': infer()