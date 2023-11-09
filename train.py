import torch
from torch.utils.data import DataLoader
import numpy as np
import warnings
import argparse
from tqdm.auto import tqdm
import wandb

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
    progress_bar = tqdm(dataloader)

    D_optimizer = torch.optim.Adam(model.descriminator.parameters(), lr=0.0002)

    G_params = list(model.generator.parameters()) + list(model.FModel.parameters())

    G_optimizer = torch.optim.Adam(G_params, lr=0.0002)
    D_scheduler = torch.optim.lr_scheduler.ExponentialLR(D_optimizer, gamma=0.995)
    G_scheduler = torch.optim.lr_scheduler.ExponentialLR(G_optimizer, gamma=0.995)

    descriminator_criterion = DescriminatorLoss()
    generator_criterion = GeneratorLoss()

    log_step = 50

    for epoch in range(args.n_epochs):
        print(f'Epoch: {epoch}')
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
                wandb.log({"D_loss": D_loss.item()})
                print(f"D_loss: {D_loss.item()}")

            D_optimizer.step()
            D_scheduler.step()

            G_optimizer.zero_grad()
            d_outputs = model.descriminate(**batch)
            batch.update(d_outputs)
            G_loss = generator_criterion(**batch)

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
        "-n",
        "--n_epochs",
        default=100,
        type=int,
        help="number of epochs (default: 100)",
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