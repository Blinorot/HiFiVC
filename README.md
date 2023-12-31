# HiFiVC

This repository contains implementation of [HiFi-VC](https://arxiv.org/abs/2203.16937) paper. Model structure is based on analysis of `graph` and `code` methods of TorchScript checkpoint provided by the authors of the paper. Most of the missing details were recovered. In addition, repository containt pre-trained versions of Speaker Encoder: VAE-part of the original solution and [ECAPA-TDNN](https://arxiv.org/pdf/2005.07143.pdf) taken from [available implementation](https://github.com/TaoRuijie/ECAPA-TDNN).

## Differences from the article

Currently, this implementation does not support F0 training. However, authors [reported results](https://openreview.net/pdf?id=1m4vHlcCQL) are not that different with or without F0.

To stabilize training, [Extra-Adam](https://arxiv.org/pdf/1802.10551.pdf) implementation was added based on [this repo](https://github.com/GauthierGidel/Variational-Inequality-GAN/blob/master/optim/extragradient.py).

## Installation

Install all packages using `pip install -r requirements.txt`.

If you want to use pre-trained VAE, run the following script:

```bash
pip install gdown
gdown 1oFwMeuQtwaBEyOFkyG7c7LfBQiRe3RdW -O "model.pt"
```

## Training

To run the experiment, run the following command:

```bash
python3 train.py -cn CONFIG_NAME +dataset.data_path=PATH_TO_WAV48_DIR
```

Where `CONFIG_NAME` is the name of the file (without `.yaml`) from `src/configs` folder, and `PATH_TO_WAV48_DIR` is the path to the VCTK dataset. For example, in Kaggle the path may look like this: `/kaggle/input/vctk-corpus/VCTK-Corpus/VCTK-Corpus/wav48`.

**Note**: add `HYDRA_FULL_ERROR=1` before `python3` to see errors.

## Credits

[Official repository]() (only inference). Extra-Adam implementation was taken from [this repository](https://github.com/GauthierGidel/Variational-Inequality-GAN/blob/master/optim/extragradient.py) and ECAPA-TDNN from [this one](https://github.com/TaoRuijie/ECAPA-TDNN)
