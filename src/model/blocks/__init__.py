#from src.model.blocks.descriminator import Descriminator
#from src.model.blocks.generator import Generator
from src.model.blocks.gan import Generator, Discriminator
from src.model.blocks.encoder import ECAPA_TDNN
from src.model.blocks.encoder import VAE, VAE2
from src.model.blocks.asr import ASRModel
from src.model.blocks.f import FModel

__all__ = [
    "Generator",
    "Discriminator",
    "ECAPA_TDNN",
    "ASRModel",
    "FModel",
    "VAE",
    "VAE2"
]
