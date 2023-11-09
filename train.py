import torch
from torch.utils.data import DataLoader
import numpy as np
import warnings
import argparse
from tqdm.auto import tqdm
import wandb

from src.model import HiFiVC
from src.utils import read_json
from src.dataset.dataset import VCDataset
from src.loss.HiFiLoss import GeneratorLoss, DescriminatorLoss


warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def train(config_path):
    config = read_json(config_path)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    model = HiFiVC(device, **config)
    model.to(device)
    dataset = VCDataset()
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=2)
    progress_bar = tqdm(dataloader)

    D_optimizer = torch.optim.Adam(lr=0.0002)
    G_optimizer = torch.optim.Adam(lr=0.0002)
    D_scheduler = torch.optim.lr_scheduler.ExponentialLR(D_optimizer, gamma=0.995)
    G_scheduler = torch.optim.lr_scheduler.ExponentialLR(G_optimizer, gamma=0.995)

    log_step = 50

    for i, (real_audio, f0) in enumerate(progress_bar):
        batch = {
            "real_audio": real_audio.to(device),
            "f0": f0.to(device)
        }
        g_outputs = model(**batch)
        batch.update(g_outputs)
        d_outputs = model.descriminate(**batch)
        batch.update(d_outputs)

        D_optimizer.zero_grad()
        D_loss = DescriminatorLoss(**batch)
        D_loss.backward()

        if i % log_step == 0:
            wandb.log({"D_loss": D_loss.item()})
            print(f"D_loss: {D_loss.item()}")

        D_optimizer.step()
        D_scheduler.step()

        G_optimizer.zero_grad()
        d_outputs = model.descriminate(batch)
        batch.update(d_outputs)
        G_loss = GeneratorLoss(**batch)

        if i % log_step == 0:
            wandb.log({"G_loss": G_loss.item()})
            print(f"G_loss: {G_loss.item()}")

        G_loss.backward()
        G_optimizer.step()
        G_scheduler.step()
        

if __name__ == '__main__':
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default="config.json",
        type=str,
        help="config file path (default: None)",
    )

    args = args.parse_args()

    with wandb.init(
        project="HiFiVC",
        name="test"):
        train(args.config)