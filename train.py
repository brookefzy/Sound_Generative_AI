# train.py
import yaml
import torch
from torch.utils.data import DataLoader, TensorDataset
from utils import load_dataset
import importlib

def train():
    cfg = yaml.safe_load(open('config_diffusion.yaml'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data = load_dataset(cfg['data_dir'], cfg['sample_rate']).to(device)
    _, _, time_steps = data.shape
    loader = DataLoader(TensorDataset(data), batch_size=cfg['batch_size'], shuffle=True)

    mod_cfg = cfg['model']
    mod_name = mod_cfg['type']
    ModelClass = getattr(importlib.import_module(f"models.{mod_name}"), mod_cfg['class'])
    params = dict(mod_cfg.get('params', {}))
    # Handle models without input_length
    if mod_name not in ['vae', 'autoregressive', 'diffusion']:
        params.pop('input_length', None)
    else:
        params.setdefault('input_length', time_steps)
    model = ModelClass(**params).to(device)

    lr = cfg['training']['lr']
    if isinstance(lr, str):
        lr = float(lr)

    if mod_name == 'wavegan':
        from models.wavegan import WaveGANDiscriminator
        discriminator = WaveGANDiscriminator().to(device)
        opt_G = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        adversarial_loss = torch.nn.BCEWithLogitsLoss()
        fm_weight = cfg['training'].get('feature_matching_weight', 0.0)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(cfg['training']['epochs']):
        for batch, in loader:
            batch = batch.to(device)

            if mod_name == 'vae':
                optimizer.zero_grad()
                recon, mu, logvar = model(batch)
                recons_loss = torch.nn.functional.mse_loss(recon, batch)
                kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recons_loss + cfg['training']['kld_weight'] * kld
                loss.backward()
                optimizer.step()

            elif mod_name == 'wavegan':
                # Train discriminator
                opt_D.zero_grad()
                z = torch.randn(batch.size(0), params.get('latent_dim', 100), 1, device=device)
                fake = model(z).detach()
                pred_real = discriminator(batch)
                pred_fake = discriminator(fake)
                loss_D = adversarial_loss(pred_real, torch.ones_like(pred_real)) + \
                         adversarial_loss(pred_fake, torch.zeros_like(pred_fake))
                loss_D.backward()
                opt_D.step()

                # Train generator
                opt_G.zero_grad()
                z = torch.randn(batch.size(0), params.get('latent_dim', 100), 1, device=device)
                fake = model(z)
                if fm_weight > 0:
                    adv_pred, fake_feats = discriminator(fake, return_features=True)
                    _, real_feats = discriminator(batch, return_features=True)
                    fm_loss = sum(torch.nn.functional.l1_loss(f.mean(dim=2), r.mean(dim=2).detach())
                                  for f, r in zip(fake_feats, real_feats))
                else:
                    adv_pred = discriminator(fake)
                    fm_loss = 0.0
                adv_loss = adversarial_loss(adv_pred, torch.ones_like(adv_pred))
                loss = adv_loss + fm_weight * fm_loss
                loss.backward()
                opt_G.step()

            elif mod_name == 'diffusion':
                optimizer.zero_grad()
                noise = torch.randn_like(batch)
                # assuming forward predicts noise given noisy audio and timestep
                t = torch.zeros(batch.size(0), dtype=torch.long, device=device)
                pred = model(batch + noise, t)
                loss = torch.nn.functional.mse_loss(pred, noise)
                loss.backward()
                optimizer.step()

            else:
                optimizer.zero_grad()
                pred = model(batch)
                loss = torch.nn.functional.mse_loss(pred, batch)
                loss.backward()
                optimizer.step()
        print(f"Epoch {epoch+1}/{cfg['training']['epochs']} - Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), f"{mod_name}_final.pt")

if __name__ == '__main__':
    train()
