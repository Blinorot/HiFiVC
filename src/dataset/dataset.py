from torch.utils.data import Dataset
from pathlib import Path
import json
import torchaudio
import torch
from tqdm.auto import tqdm
import numpy as np
import random

from src.dataset.audio_utils import load_and_preprocess_audio
from src.dataset.f0_utils import get_lf0_from_wav

class VCDataset(Dataset):
    def __init__(self, data_path = None, part='train', max_audio_length=16000, limit=None):
        super().__init__()
        self.part = part
        if data_path == None:
            data_path = Path(__file__).absolute().resolve().parent.parent.parent / 'data' / 'VCTK-Corpus/VCTK-Corpus/wav48'
        else:
            data_path = Path(data_path)
        self.index_path = Path(__file__).absolute().resolve().parent.parent.parent / 'index'
        self.data_path = data_path

        self.data_path.mkdir(parents=True, exist_ok=True)
        self.index_path.mkdir(parents=True, exist_ok=True)

        self.index = self.load_index()
        if limit is not None:
            random.shuffle(self.index)
            self.index = self.index[:limit]
        self.limit = limit
        self.max_audio_length = max_audio_length

    def load_index(self):
        index_path = self.index_path / f'{self.part}.json'
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self.create_index()
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index
    
    def create_index(self):
        index = []
        if self.part == 'train':
            speakers = [f'p{i}' for i in range(225, 227)]
        else:
            speakers = [f'p{i}' for i in range(361, 377)]

        for speaker_id in tqdm(
                speakers, desc=f"Preparing VCTK folders: {self.part}"
        ):
            wav_dir = self.data_path / speaker_id
            if not wav_dir.exists():
                continue
            wav_paths = list(wav_dir.glob("*.wav"))
            for wav_path in wav_paths:
                t_info = torchaudio.info(str(wav_path))
                length = t_info.num_frames / t_info.sample_rate
                index.append(
                    {
                        "path": str(wav_path.absolute().resolve()),
                        "audio_len": length,
                    }
                )
        return index
            

    def __len__(self):
        return len(self.index)

    def __getitem__(self, ind):
        data_dict = self.index[ind]
        audio_path = data_dict["path"]
        audio_wave = load_and_preprocess_audio(audio_path, trim=True)

        if audio_wave.shape[-1] > self.max_audio_length:
            random_start = np.random.randint(0, audio_wave.shape[-1] - self.max_audio_length + 1)
            if self.limit:
                random_start = 0
            audio_wave = audio_wave[..., random_start:random_start+self.max_audio_length]
        elif audio_wave.shape[-1] < self.max_audio_length:
            pad_length = self.max_audio_length - audio_wave.shape[-1]
            audio_wave = torch.cat([audio_wave, torch.zeros(pad_length).unsqueeze(0)], dim=1)

        f0 = get_lf0_from_wav(audio_wave.numpy()[0])

        audio_length = audio_wave.shape[-1]
        #print(f0.shape, audio_wave.shape, audio_length)
        return {
            "real_audio": audio_wave,
            "f0": f0,
            "audio_length": audio_length
        }


def collate_fn(data_list):
    batch = {}
    batch['real_audio'] = torch.cat([elem['real_audio'] for elem in data_list], dim=0).unsqueeze(1)
    batch['f0'] = torch.cat([elem['f0'] for elem in data_list], dim=0)
    batch['audio_length'] = torch.tensor([elem['audio_length'] for elem in data_list])
    return batch