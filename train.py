import torch
from torch.utils.data import DataLoader
import numpy as np
import warnings
import argparse
from tqdm.auto import tqdm
import wandb
from pathlib import Path

from src.model import HiFiVC
from src.utils import read_json
from src.dataset.dataset import VCDataset, collate_fn
from src.loss.HiFiLoss import GeneratorLoss, DescriminatorLoss


warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def train(args):
    config = read_json(args.config)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    model = HiFiVC(device, **config)
    model.to(device)
    dataset = VCDataset(data_path=args.data_path, part='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, collate_fn=collate_fn)

    D_optimizer = torch.optim.Adam(model.descriminator.parameters(), lr=0.0002)

    G_params = list(model.generator.parameters()) + list(model.FModel.parameters())

    G_optimizer = torch.optim.Adam(G_params, lr=0.0002)
    D_scheduler = torch.optim.lr_scheduler.ExponentialLR(D_optimizer, gamma=0.995)
    G_scheduler = torch.optim.lr_scheduler.ExponentialLR(G_optimizer, gamma=0.995)

    descriminator_criterion = DescriminatorLoss()
    generator_criterion = GeneratorLoss()

    descriminator_criterion.to(device)
    generator_criterion.to(device)

    log_step = 50

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    step = 0

    for epoch in range(args.n_epochs):
        print(f'Epoch: {epoch}')
        progress_bar = tqdm(dataloader)
        for i, batch in enumerate(progress_bar):
            batch['real_audio'] = batch['real_audio'].to(device)
            batch['f0'] = batch['f0'].to(device)
            batch['audio_length'] = batch['audio_length'].to(device)

            g_outputs = model(**batch)
            batch.update(g_outputs)
            d_outputs = model.descriminate(generated_audio=batch["generated_audio"].detach(),
                                        real_audio=batch["real_audio"])
            batch.update(d_outputs)

            D_optimizer.zero_grad()
            D_loss = descriminator_criterion(**batch)
            D_loss.backward()

            if i % log_step == 0:
                wandb.log({"D_loss": D_loss.item()}, step=step)
                print(f"D_loss: {D_loss.item()}")

            D_optimizer.step()
            D_scheduler.step()

            G_optimizer.zero_grad()
            d_outputs = model.descriminate(**batch)
            batch.update(d_outputs)
            G_loss = generator_criterion(**batch)

            if i % log_step == 0:
                wandb.log({"G_loss": G_loss.item()}, step=step)
                print(f"G_loss: {G_loss.item()}")

            G_loss.backward()
            G_optimizer.step()
            G_scheduler.step()
            step += 1
        torch.save(model.state_dict(), str(save_path / f'model.pth'),
                   _use_new_zipfile_serialization=False)
        generated_audio = batch['generated_audio'][0].detach().cpu().numpy().T
        real_audio = batch['real_audio'][0].detach().cpu().numpy().T
        wandb.log({
            'generated_audio': wandb.Audio(generated_audio, sample_rate=16000),
            'real_audio': wandb.Audio(real_audio, sample_rate=16000)
        }, step=step)
        

if __name__ == '__main__':
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default="config.json",
        type=str,
        help="config file path (default: config.json)",
    )
    args.add_argument(
        "-d",
        "--data_path",
        default=None,
        type=str,
        help="data path (default: None)",
    )
    args.add_argument(
        "-s",
        "--save_path",
        default='saved',
        type=str,
        help="save path (default: saved)",
    )
    args.add_argument(
        "-n",
        "--n_epochs",
        default=120,
        type=int,
        help="number of epochs (default: 120)",
    )
    args.add_argument(
        "-b",
        "--batch_size",
        default=20,
        type=int,
        help="batch_size (default: 20)",
    )
    args.add_argument(
        "--num_workers",
        default=2,
        type=int,
        help="number of workers (default: 2)",
    )

    args = args.parse_args()

    with wandb.init(
        project="HiFiVC",
        name="test"):
        train(args)