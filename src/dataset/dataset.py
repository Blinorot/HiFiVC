from torch.utils.data import Dataset
from pathlib import Path
import json
import torchaudio
from tqdm.auto import tqdm

from src.dataset.audio_utils import load_and_preprocess_audio
from src.dataset.f0_utils import get_lf0_from_wav

class VCDataset(Dataset):
    def __init__(self, data_path = None, part='train'):
        super().__init__()
        self.part = part
        if data_path == None:
            data_path = Path(__file__).absolute().resolve().parent.parent.parent / 'data' / 'VCTK-Corpus/VCTK-Corpus/wav48'
        self.data_path = data_path
        self.index = self.load_index()

    def load_index(self):
        index_path = self.data_path.parent / f'{self.part}.json'
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index()
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index
    
    def create_index(self):
        index = []
        if self.part == 'train':
            speakers = [f'p{i}' for i in range(225, 241)]
        else:
            speakers = [f'p{i}' for i in range(241, 247)]

        for speaker_id in tqdm(
                speakers, desc=f"Preparing VCTK folders: {self.part}"
        ):
            wav_dir = self.data_path / speaker_id
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
        audio = load_and_preprocess_audio(audio_path, trim=True)
        f0 = get_lf0_from_wav(audio_path)
        return audio, f0
