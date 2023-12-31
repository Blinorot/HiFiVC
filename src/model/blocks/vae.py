import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.nn.utils import weight_norm
    

class VAEPretrained(nn.Module):
    def __init__(self, device):
        super().__init__()
        model_path = Path(__file__).absolute().resolve().parent.parent.parent.parent / 'model.pt'
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()

        self.mean_conv = weight_norm(nn.Conv1d(512, 128, 1)).to(device)
        self.std_conv = weight_norm(nn.Conv1d(512, 128, 1)).to(device)

        mean_conv_parameters = self.model._voice_conversion.speaker_encoder.mean_conv._parameters
        self.mean_conv.bias = nn.Parameter(mean_conv_parameters['bias']).to(device)
        self.mean_conv.weight_v = nn.Parameter(mean_conv_parameters['weight_v']).to(device)
        self.mean_conv.weight_g = nn.Parameter(mean_conv_parameters['weight_g']).to(device)

        std_conv_parameters = self.model._voice_conversion.speaker_encoder.std_conv._parameters
        self.std_conv.bias = nn.Parameter(std_conv_parameters['bias']).to(device)
        self.std_conv.weight_v = nn.Parameter(std_conv_parameters['weight_v']).to(device)
        self.std_conv.weight_g = nn.Parameter(std_conv_parameters['weight_g']).to(device)

        self.mean_conv.eval()
        self.std_conv.eval()

    def forward(self, x):
        x = self.model._voice_conversion.speaker_encoder.pre_conv(x)
        for i in range(5):
            x = getattr(self.model._voice_conversion.speaker_encoder.convs, str(i))(x)
        mean_x = self.mean_conv(x)
        std_x = self.std_conv(x)
        if self.training:
            exp_std = torch.exp(0.5 * std_x)
            eps = torch.randn_like(exp_std).to(x.device)
            res = mean_x + exp_std * eps
        else:
            res = mean_x
        return {"result": res, "mean": mean_x, "std": std_x}
    

class DownSample(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.skip_conv = weight_norm(nn.Conv1d(channel, channel * 2, 1))
        self.first_conv = weight_norm(nn.Conv1d(channel, channel, 3, 1, 1))
        self.second_conv = weight_norm(nn.Conv1d(channel, channel * 2, 1, 1))

    def forward(self, x):
        skip_x = self.skip_conv(x)
        skip_x = F.avg_pool1d(skip_x, 2, 2, 0)
        #avg skip
        xt = F.leaky_relu(x, 0.1)
        xt = self.first_conv(xt)
        #avg xt
        xt = F.avg_pool1d(xt, 2, 2, 0)
        xt = F.leaky_relu(xt, 0.1)
        xt = self.second_conv(xt)
        return xt + skip_x
    

class SameSample(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.first_conv = weight_norm(nn.Conv1d(channel, channel, 3, 1, 1))
        self.second_conv = weight_norm(nn.Conv1d(channel, channel, 1, 1))

    def forward(self, x):
        skip_x = F.avg_pool1d(x, 2, 2, 0)
        #avg skip
        xt = F.leaky_relu(x, 0.1)
        xt = self.first_conv(xt)
        #avg xt
        xt = F.avg_pool1d(xt, 2, 2, 0)
        xt = F.leaky_relu(xt, 0.1)
        xt = self.second_conv(xt)
        return xt + skip_x
    

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_conv = nn.Conv1d(80, 32, 3)
        
        self.convs = nn.Sequential(
            DownSample(32),
            DownSample(64),
            DownSample(128),
            DownSample(256),
            SameSample(512)
        )

        self.mean_conv = weight_norm(nn.Conv1d(512, 128, 1))
        self.std_conv = weight_norm(nn.Conv1d(512, 128, 1))
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.convs(x)
        mean_x = self.avg_pool(self.mean_conv(x))
        std_x = self.avg_pool(self.std_conv(x))
        if self.training:
            exp_std = torch.exp(0.5 * std_x)
            eps = torch.randn_like(exp_std).to(x.device)
            res = mean_x + exp_std * eps
        else:
            res = mean_x
        return {"result": res, "mean": mean_x, "std": std_x}