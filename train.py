# train.py
import yaml
import torch
from torch.utils.data import DataLoader, TensorDataset
from utils import load_dataset
import importlib

def train():
    cfg = yaml.safe_load(open('config.yaml'))
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
    if isinstance(lr, str): lr = float(lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(cfg['training']['epochs']):
        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            if mod_name == 'vae':
                recon, mu, logvar = model(batch)
                recons_loss = torch.nn.functional.mse_loss(recon, batch)
                kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recons_loss + cfg['training']['kld_weight'] * kld
            else:
                # TODO: implement specific losses
                loss = torch.tensor(0., device=device)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{cfg['training']['epochs']} - Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), f"{mod_name}_final.pt")

if __name__ == '__main__':
    train()