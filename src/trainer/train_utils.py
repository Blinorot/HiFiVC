import torch
import wandb
from tqdm.auto import tqdm
from torch.nn.utils import clip_grad_norm_


MAX_NORM = 500000


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


def save_checkpoint(model, G_optimizer, D_optimizer, G_scheduler, D_scheduler, epoch, save_path):
    save_dict = {
        "model": model.state_dict(),
        "G_optimizer": G_optimizer.state_dict(),
        "D_optimizer": D_optimizer.state_dict(),
        "G_scheduler": G_scheduler.state_dict(),
        "D_scheduler": D_scheduler.state_dict(),
        "epoch": epoch
    }
    torch.save(save_dict, str(save_path / f'save_dict.pth'),
               _use_new_zipfile_serialization=False)
    

def load_checkpoint(checkpoint_path, model, G_optimizer, D_optimizer, G_scheduler, D_scheduler):
    checkpoint = torch.load(checkpoint_path)
    epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["model"])
    G_optimizer.load_state_dict(checkpoint["G_optimizer"])
    D_optimizer.load_state_dict(checkpoint["D_optimizer"])
    G_scheduler.load_state_dict(checkpoint["G_scheduler"])
    D_scheduler.load_state_dict(checkpoint["D_scheduler"])
    return epoch + 1


def move_batch_to_device(batch, device):
    names = ["mel_spec", "real_audio", "source_audio", "audio_length"]
    for name in names:
        batch[name] = batch[name].to(device)
    return batch


def train_epoch(model, dataloader, D_optimizer, G_optimizer, D_scheduler, G_scheduler,
                discriminator_criterion, generator_criterion, epoch_len, log_step, cfg, epoch, device, save_path):
    step = epoch * epoch_len
    progress_bar = tqdm(dataloader, total=epoch_len)
    for i, batch in enumerate(progress_bar):
        batch = move_batch_to_device(batch, device)

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
        if not cfg.trainer.generator_only:
            clip_grad_norm_(model.speaker_encoder.parameters(),  cfg.trainer.get("max_grad_norm", MAX_NORM))
        
        if i % log_step == 0:
            wandb.log({
                "G_loss": G_loss.item(),
                "adv_loss": adv_loss.item(),
                "fm_loss": fm_loss.item(),
                "mel_loss": mel_loss.item(),
                "kl_loss": kl_loss.item(),
                "G_grad": get_grad_norm(model.generator),
                "VAE_grad": 0 if cfg.trainer.generator_only else get_grad_norm(model.speaker_encoder)
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
        if i == epoch_len - 1:
            break

    generated_audio = batch['generated_audio'][0].detach().cpu().numpy().T
    real_audio = batch['real_audio'][0].detach().cpu().numpy().T
    wandb.log({
        'generated_audio': wandb.Audio(generated_audio, sample_rate=24000),
        'real_audio': wandb.Audio(real_audio, sample_rate=24000),
        'epoch': epoch
    }, step=step)

    save_checkpoint(model=model, G_optimizer=G_optimizer, D_optimizer=D_optimizer,
                    G_scheduler=G_scheduler, D_scheduler=D_scheduler, epoch=epoch,
                    save_path=save_path)