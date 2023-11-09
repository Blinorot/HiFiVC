from torch import nn
import nemo.collections.asr as nemo_asr

class ASRModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_conformer_ctc_large_ls")
    
    def forward(self, audio, audio_length, **batch):
        preproc_audio, preproc_audio_length = self.asr_model.preprocessor.forward(
            input_signal=audio,
            length=audio_length
        )
        encoded, encoded_length = self.asr_model.encoder.forward(
            audio_signal=preproc_audio,
            length=preproc_audio_length)
        return encoded