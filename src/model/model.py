import torch
from src.model.blocks import Descriminator, Generator, ECAPA_TDNN, ASRModel, FModel
from torch import nn
from speechbrain.utils.data_utils import download_file
from pathlib import Path


class HiFiVC(nn.Module):
    def __init__(self, device, **kwargs):
        super().__init__()
        self.device = device

        self.generator = Generator(**kwargs)
        self.speaker_encoder = ECAPA_TDNN(**kwargs)
        self.load_encoder()

        self.FModel = FModel(**kwargs)
        self.AsrModel = ASRModel(**kwargs)
        #self.AsrModel = nn.Identity()

        self.descriminator = Descriminator(**kwargs)

        self.speaker_proj = []
        for i in range(len(kwargs['speaker_proj'])):
            self.speaker_proj.append(nn.Linear(192, kwargs['speaker_proj'][i]))
        self.speaker_proj_length = len(self.speaker_proj)
        self.speaker_proj = nn.ModuleList(self.speaker_proj)   
    

    def load_encoder(self):
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

    def forward(self, source_audio, real_audio, f0, audio_length, **batch):
        #print("source_shape", source_audio.shape)
        with torch.no_grad():
            speaker_info = self.speaker_encoder(source_audio[:,0,:])
            text_info = self.AsrModel(source_audio[:,0,:], audio_length)

        f_info = self.FModel(f0)

        spectrogram = f_info + text_info
        #print('asr', text_info.shape)

        # speaker_info_list = []
        # for i in range(self.speaker_proj_length):
        #     speaker_info_list.append(self.speaker_proj[i](speaker_info).unsqueeze(1))

        #print('speaker_info.shape', speaker_info.shape)
        result = self.generator(spectrogram, speaker_info.unsqueeze(1))
        #print(result['generated_audio'].shape, real_audio.shape)
        #result['generated_audio'] = result['generated_audio'][:, :, :audio_length[0]]    

        return result

    def generate(self, **batch):
        return self.forward(**batch)

    def descriminate(self, generated_audio, real_audio, **batch):
        return self.descriminator(generated_audio, real_audio)
