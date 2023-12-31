from torch import nn

LRELU_SLOPE = 0.1

class FModel(nn.Module):
    def __init__(self, pitch_embed_dim, **kwargs):
        super().__init__()
        self.pitch_convs = nn.Sequential(
            nn.Conv1d(
                    2, pitch_embed_dim // 2, kernel_size=1, stride=1, 
                    padding=0, bias=False),
            nn.LeakyReLU(LRELU_SLOPE),
            nn.InstanceNorm1d(pitch_embed_dim // 2, affine=False),
            nn.Conv1d(
                    pitch_embed_dim // 2, pitch_embed_dim // 2, kernel_size=5, stride=2, 
                    padding=1, bias=False),
            nn.LeakyReLU(LRELU_SLOPE),
            nn.InstanceNorm1d(pitch_embed_dim // 2, affine=False),
            nn.Conv1d(
                    pitch_embed_dim // 2, pitch_embed_dim // 2, kernel_size=5, stride=1, 
                    padding=1, bias=False),
            nn.LeakyReLU(LRELU_SLOPE),
            nn.InstanceNorm1d(pitch_embed_dim // 2, affine=False),
            nn.Conv1d(pitch_embed_dim // 2, pitch_embed_dim, kernel_size=7, 
                      stride=1, padding=0, bias=False)
        )

    def forward(self, logf0_uv, **batch):
        #print('pitch_was', logf0_uv.shape)
        logf0_uv = self.pitch_convs(logf0_uv)
        #print('pitch_now', logf0_uv.shape)
        return logf0_uv