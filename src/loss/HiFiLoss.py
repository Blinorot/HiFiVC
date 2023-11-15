import torch
from src.preprocessing.melspec import MelSpectrogram
from src.dataset.audio_utils import mel_spectrogram, AudioFeaturesParams
from src.model.blocks.gan import feature_loss, generator_loss, discriminator_loss
from torch import nn


class GeneratorLoss(nn.Module):
    def __init__(self, fm_coef=1, mel_coef=45, kl_coef=0.01):
        super().__init__()

        self.fm_coef = fm_coef
        self.mel_coef = mel_coef
        self.kl_coef = kl_coef

        self.l1_loss = nn.L1Loss()
        self.mel_transform = MelSpectrogram()

    def forward(self, 
        real_audio, 
        generated_audio,
        p_gen_outs,
        p_real_feat,
        p_gen_feat,
        s_gen_outs,
        s_real_feat,
        s_gen_feat,
        mean_info,
        std_info,
        **kwargs):

        mean_info = mean_info[..., 0]
        std_info = std_info[..., 0]

        KL_loss = torch.mean(-0.5 * torch.sum(1 + std_info - mean_info ** 2 - std_info.exp(), dim = 1), dim = 0)
        #KL_loss = torch.tensor([0]).to(real_audio.device)

        spectrogram = self.mel_transform(real_audio)
        generated_audio = generated_audio.squeeze(1) # remove channel
        generated_spectrogram = self.mel_transform(generated_audio) 

        fm_loss_p = feature_loss(p_real_feat, p_gen_feat)
        fm_loss_s = feature_loss(s_real_feat, s_gen_feat)

        fm_loss = fm_loss_s + fm_loss_p

        adv_loss_p, _ = generator_loss(p_gen_outs)
        adv_loss_s, _ = generator_loss(s_gen_outs)

        adv_loss = adv_loss_p + adv_loss_s

        # mel_loss
        mel_loss = self.l1_loss(generated_spectrogram, spectrogram)

        G_loss = adv_loss + self.fm_coef * fm_loss + self.mel_coef * mel_loss\
              + self.kl_coef * KL_loss

        return G_loss, adv_loss, fm_loss, mel_loss, KL_loss
    

class DescriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, 
        p_real_outs,
        p_gen_outs,
        s_real_outs,
        s_gen_outs,
        **kwargs):

        # D_loss
        d_loss_p, _, _ = discriminator_loss(p_real_outs, p_gen_outs)
        d_loss_s, _, _ = discriminator_loss(s_real_outs, s_gen_outs)

        D_loss = d_loss_p + d_loss_s

        return D_loss
