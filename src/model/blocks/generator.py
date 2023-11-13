from src.model.utils import get_conv_padding_size
from torch import nn
from torch.nn.utils import weight_norm

LRELU_SLOPE = 0.1

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

class MRFBlock(nn.Module):
    def __init__(self, channels, kernel,  dilations):
        super().__init__()

        self.kernel = kernel
        self.dilations = dilations

        layers = []
        speakers = []
        speakers_linear = []

        for m in range(len(dilations)):
            layer = nn.Sequential(
                nn.LeakyReLU(LRELU_SLOPE),
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel, 
                          dilation=dilations[m], 
                          padding=get_conv_padding_size(kernel, 1, dilations[m]))),
                nn.LeakyReLU(LRELU_SLOPE),
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel, 
                          dilation=1, 
                          padding=get_conv_padding_size(kernel, 1, 1))),
            )
            layers.append(layer)

            speaker = nn.Sequential(
                nn.LeakyReLU(LRELU_SLOPE),
                weight_norm(
                    nn.Conv1d(1, 1, kernel_size=kernel, 
                          dilation=1, 
                          padding=get_conv_padding_size(kernel, 1, 1))
                )
            )
            speakers.append(speaker)
        
        self.block = nn.ModuleList(layers)
        self.speakers = nn.ModuleList(speakers)

        self.block.apply(init_weights)
        self.speakers.apply(init_weights)

    def forward(self, x, speaker):
        result = 0
        for i in range(len(self.block)):
            speaker_result = self.speakers[i](speaker)
            result = result + self.block[i](x + speaker_result.transpose(1, 2))
        return result


class MRF(nn.Module):
    def __init__(self, channels, resblock_kernels, resblock_dilations):
        super().__init__()

        resblocks = []
        for i in range(len(resblock_kernels)):
            resblocks.append(MRFBlock(channels, resblock_kernels[i], resblock_dilations[i]))

        self.resblocks = nn.ModuleList(resblocks)

    def forward(self, x, speaker):
        result = 0
        for block in self.resblocks:
            result = result + block(x, speaker)
        return result


class Generator(nn.Module):
    def __init__(self, input_channels, hidden_channels, upsample_kernels,
                 upsample_stride, resblock_kernels,
                 resblock_dilations, **kwargs):
        super().__init__()

        self.in_conv = weight_norm(nn.Conv1d(input_channels, hidden_channels, 7, 
                                  padding=get_conv_padding_size(7, 1, 1)))

        blocks = []
        current_channels = hidden_channels
        for i in range(len(upsample_kernels)):
            upsample = weight_norm(nn.ConvTranspose1d(current_channels, current_channels // 2,
                                          upsample_kernels[i], upsample_stride[i],
                                          padding=(upsample_kernels[i] - upsample_stride[i]) // 2))
            upsample.apply(init_weights)
            mrf = MRF(current_channels // 2, resblock_kernels, resblock_dilations)
            block = nn.ModuleList([upsample, mrf])
            blocks.append(block)

            current_channels = current_channels // 2

        self.blocks_length = len(blocks)
        self.blocks = nn.ModuleList(blocks)

        self.out_conv = nn.Sequential(
            nn.LeakyReLU(LRELU_SLOPE),
            weight_norm(nn.Conv1d(current_channels, 1, 7, 
                      padding=get_conv_padding_size(7, 1, 1), bias=False)),
            nn.Tanh()
        )
        self.out_conv.apply(init_weights)

    def forward(self, spectrogram, speaker_info_list):
        spectrogram = self.in_conv(spectrogram)
        for i in range(self.blocks_length):
            spectrogram = self.blocks[i][0](spectrogram)
            spectrogram = self.blocks[i][1](spectrogram, speaker_info_list[i])
        generated_audio = self.out_conv(spectrogram)
        return {"generated_audio": generated_audio}
