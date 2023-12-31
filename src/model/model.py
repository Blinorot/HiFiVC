import torch
from src.model.blocks import Discriminator, Generator, ECAPA_TDNN, VAEPretrained, VAE, ASRModel, FModel
from torch import nn
from speechbrain.utils.data_utils import download_file
from pathlib import Path


class HiFiVC(nn.Module):
    def __init__(self, encoder, device, generator_only=False, use_f=False, **kwargs):
        super().__init__()
        self.device = device

        self.generator = Generator(**kwargs)

        self.encoder = encoder
        self.generator_only = generator_only

        if encoder == "ECAPA":
            self.speaker_encoder = ECAPA_TDNN(**kwargs)
            self._load_encoder()
        elif encoder == "VAE":
            self.speaker_encoder = VAE()
        elif encoder == "VAEPretrained":
            self.speaker_encoder = VAEPretrained()
        else:
            raise NotImplementedError()
        
        self.use_f = use_f
        if use_f:
            self.FModel = FModel(**kwargs)

        self.AsrModel = ASRModel(**kwargs)

        self.discriminator = Discriminator(**kwargs)


    def _load_encoder(self):
        data_path = Path(__file__).absolute().resolve().parent.parent.parent / 'data'
        model_path = data_path / 'encoder.pth'
        data_path.mkdir(parents=True, exist_ok=True)
        if not model_path.exists():
            download_file("https://github.com/TaoRuijie/ECAPA-TDNN/raw/main/exps/pretrain.model",
                          model_path)
        state_dict = torch.load(model_path, map_location=self.device)

        new_state_dict = {}
        for k, v in state_dict.items():
            k_split = k.split('.')
            if k_split[0] == 'speaker_encoder':
                new_k = '.'.join(k_split[1:])
                new_state_dict[new_k] = v

        self.speaker_encoder.load_state_dict(new_state_dict)

    def forward(self, source_audio, mel_spec, real_audio, f0=None, audio_length=None, **batch):
        encoder_input = mel_spec
        if self.encoder == "ECAPA":
            encoder_input = source_audio[:,0,:]
        
        with torch.no_grad():
            text_info = self.AsrModel(source_audio[:,0,:], audio_length)
            if self.generator_only:
                speaker_res = self.speaker_encoder(encoder_input)

        if not self.generator_only:
            speaker_res = self.speaker_encoder(encoder_input)

        
        if self.use_f:
            f_info = self.FModel(f0)
            spectrogram = f_info + text_info
        else:
            spectrogram = text_info

        speaker_info = speaker_res['result']
        mean_info = speaker_res.get('mean', None)
        std_info = speaker_res.get('std', None)

        result = self.generator(spectrogram, speaker_info)
        #print(result['generated_audio'].shape, real_audio.shape)
        #result['generated_audio'] = result['generated_audio'][..., :real_audio[0].shape[-1]]
        #result['generated_audio'] = result['generated_audio'][:, :, :audio_length[0]]    

        result['mean_info'] = mean_info
        result['std_info'] = std_info

        return result

    def generate(self, **batch):
        return self.forward(**batch)

    def discriminate(self, generated_audio, real_audio, **batch):
        real_audio_pad = torch.zeros_like(generated_audio, device=real_audio.device)
        real_audio_pad[..., :real_audio.shape[-1]] = real_audio
        return self.discriminator(generated_audio, real_audio_pad)
