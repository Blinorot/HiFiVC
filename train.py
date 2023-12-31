import torch
from torch.utils.data import DataLoader
import numpy as np
import warnings
import wandb
from pathlib import Path
import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import shutil

from src.utils import inf_loop
from src.dataset.dataset import collate_fn
from src.trainer.train_utils import train_epoch, load_checkpoint


warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
DEFAULT_CONFIG_NAME = "hifivc"
ROOT_PATH = Path(__file__).absolute().resolve().parent
SAVE_DIR = ROOT_PATH / "saved"


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def train(cfg: DictConfig, save_path: Path, resume: str):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    model = instantiate(cfg.model, device=device)
    model.to(device)

    names = ["generator", "discriminator", "AsrModel", "speaker_encoder"]
    for name in names:
        print(f"{name} params: {sum([p.numel() for p in getattr(model, name).parameters()])}")

    dataset = instantiate(cfg.dataset, part="train")
    dataloader = DataLoader(dataset, batch_size=cfg.trainer.batch_size, shuffle=True,
                            num_workers=cfg.trainer.num_workers, collate_fn=collate_fn)
    dataloader = inf_loop(dataloader)

    if cfg.trainer.generator_only:
        G_params = model.generator.parameters()
    else:
        G_params = list(model.generator.parameters()) + list(model.speaker_encoder.parameters())
    G_optimizer = instantiate(cfg.G_optimizer, params=G_params)
    G_scheduler = instantiate(cfg.G_scheduler, optimizer=G_optimizer)
    D_optimizer = instantiate(cfg.D_optimizer, params=model.discriminator.parameters())
    D_scheduler = instantiate(cfg.D_scheduler, optimizer=D_optimizer)

    discriminator_criterion = instantiate(cfg.D_loss_function)
    generator_criterion = instantiate(cfg.G_loss_function)

    discriminator_criterion.to(device)
    generator_criterion.to(device)

    log_step = cfg.trainer.log_step
    epoch_len = cfg.trainer.epoch_len

    model.train()
    model.AsrModel.eval()
    if cfg.trainer.generator_only:
        model.speaker_encoder.eval()

    start_epoch = 0

    if resume == "must":
        checkpoint_path = save_path / "save_dict.pth"
        start_epoch = load_checkpoint(model=model, G_optimizer=G_optimizer, D_optimizer=D_optimizer,
                        G_scheduler=G_scheduler, D_scheduler=D_scheduler,
                        checkpoint_path=checkpoint_path)

    for epoch in range(start_epoch, cfg.trainer.n_epochs):
        print(f'Epoch: {epoch}')
        train_epoch(model=model, dataloader=dataloader, G_optimizer=G_optimizer, D_optimizer=D_optimizer,
                    D_scheduler=D_scheduler,G_scheduler=G_scheduler,
                    discriminator_criterion=discriminator_criterion,
                    generator_criterion=generator_criterion,
                    epoch=epoch, epoch_len=epoch_len, log_step=log_step,
                    device=device, cfg=cfg, save_path=save_path)


@hydra.main(
    version_base=None,
    config_path=str(ROOT_PATH / "src" / "configs"),
    config_name=DEFAULT_CONFIG_NAME,
)
def main(cfg: DictConfig):
    SAVE_DIR.mkdir(exist_ok=True, parents=True)
    fix_seed(SEED)


    save_path = SAVE_DIR / cfg.trainer.run_name
    if save_path.exists() and not cfg.trainer.override:
        run_id_path = save_path / "run_id.log"
        run_id = torch.load(run_id_path)["run_id"]
        resume = "must"
    else:
        if save_path.exists():
            shutil.rmtree(str(save_path))
        save_path.mkdir(parents=True, exist_ok=True)
        run_id = wandb.util.generate_id()
        torch.save({"run_id": run_id}, str(save_path / "run_id.log"))
        resume = None

    with wandb.init(
        project="HiFiVC",
        name=cfg.trainer.run_name,
        id=run_id,
        resume=resume):
        train(cfg, save_path, resume)
        

if __name__ == '__main__':
    main()