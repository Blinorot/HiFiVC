import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import numpy as np
import warnings
import argparse
from tqdm.auto import tqdm
import wandb
from pathlib import Path
import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.utils import inf_loop
from src.dataset.dataset import collate_fn


warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
DEFAULT_CONFIG_NAME = "hifivc"
ROOT_PATH = Path(__file__).absolute().resolve().parent
SAVE_DIR = ROOT_PATH / "saved"
MAX_NORM = 50000


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


@torch.no_grad()
def get_grad_norm(model, norm_type=2):
    parameters = model.parameters()
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
        ),
        norm_type,
    )
    return total_norm.item()


def train(cfg: DictConfig):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    model = instantiate(cfg.model, device=device)
    model.to(device)
    dataset = instantiate(cfg.dataset, part="train")
    dataloader = DataLoader(dataset, batch_size=cfg.trainer.batch_size, shuffle=True,
                            num_workers=cfg.trainer.num_workers, collate_fn=collate_fn)
    dataloader = inf_loop(dataloader)


    if cfg.trainer.get("generator_only", False):
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

    save_path = SAVE_DIR
    epoch_len = cfg.trainer.epoch_len

    step = 0

    model.train()
    model.AsrModel.eval()

    for epoch in range(cfg.trainer.n_epochs):
        print(f'Epoch: {epoch}')
        progress_bar = tqdm(dataloader, total=epoch_len)
        for i, batch in enumerate(progress_bar):
            if i == epoch_len:
                break

            batch['mel_spec'] = batch['mel_spec'].to(device)
            batch['real_audio'] = batch['real_audio'].to(device)
            batch['source_audio'] = batch['source_audio'].to(device)
            #batch['f0'] = batch['f0'].to(device)
            batch['audio_length'] = batch['audio_length'].to(device)

            g_outputs = model(**batch)
            batch.update(g_outputs)

            D_optimizer.zero_grad()

            d_outputs = model.discriminate(generated_audio=batch["generated_audio"].detach(),
                                        real_audio=batch["real_audio"])
            batch.update(d_outputs)

            D_loss = discriminator_criterion(**batch)
            D_loss.backward()
            clip_grad_norm_(model.discriminator.parameters(), cfg.trainer.get("max_grad_norm", MAX_NORM))

            if i % log_step == 0:
                wandb.log({"D_loss": D_loss.item(),
                           "D_grad": get_grad_norm(model.discriminator)}, step=step)
                print(f"D_loss: {D_loss.item()}")

            D_optimizer.step()

            G_optimizer.zero_grad()
            d_outputs = model.discriminate(**batch)
            batch.update(d_outputs)
            G_loss, adv_loss, fm_loss, mel_loss, kl_loss = generator_criterion(**batch)

            G_loss.backward()
            clip_grad_norm_(model.generator.parameters(),  cfg.trainer.get("max_grad_norm", MAX_NORM))
            clip_grad_norm_(model.speaker_encoder.parameters(),  cfg.trainer.get("max_grad_norm", MAX_NORM))
            if i % log_step == 0:
                wandb.log({
                    "G_loss": G_loss.item(),
                    "adv_loss": adv_loss.item(),
                    "fm_loss": fm_loss.item(),
                    "mel_loss": mel_loss.item(),
                    "kl_loss": kl_loss.item(),
                    "G_grad": get_grad_norm(model.generator),
                    "VAE_grad": get_grad_norm(model.speaker_encoder)
                },step=step)
                generated_audio = batch['generated_audio'][0].detach().cpu().numpy().T
                real_audio = batch['real_audio'][0].detach().cpu().numpy().T
                wandb.log({
                    'step_generated_audio': wandb.Audio(generated_audio, sample_rate=24000),
                    'step_real_audio': wandb.Audio(real_audio, sample_rate=24000)
                }, step=step)

                print(f"G_loss: {G_loss.item()}")

            G_optimizer.step()

            D_scheduler.step()
            G_scheduler.step()
            step += 1
        torch.save(model.state_dict(), str(save_path / f'model.pth'),
                   _use_new_zipfile_serialization=False)
        generated_audio = batch['generated_audio'][0].detach().cpu().numpy().T
        real_audio = batch['real_audio'][0].detach().cpu().numpy().T
        wandb.log({
            'generated_audio': wandb.Audio(generated_audio, sample_rate=24000),
            'real_audio': wandb.Audio(real_audio, sample_rate=24000)
        }, step=step)


@hydra.main(
    version_base=None,
    config_path=str(ROOT_PATH / "src" / "configs"),
    config_name=DEFAULT_CONFIG_NAME,
)
def main(cfg: DictConfig):
    SAVE_DIR.mkdir(exist_ok=True, parents=True)
    fix_seed(SEED)

    with wandb.init(
        project="HiFiVC",
        name="seminar_test"):
        train(cfg)
        

if __name__ == '__main__':
    main()