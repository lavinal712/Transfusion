# 💉🩸 Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2408.11039-b31b1b.svg)](https://arxiv.org/abs/2408.11039)&nbsp;
</div>

## Introduction

We introduce Transfusion, a recipe for training a multi-modal model over discrete and continuous data. Transfusion combines the language modeling loss function (next token prediction) with diffusion to train a single transformer over mixed-modality sequences. We pretrain multiple Transfusion models up to 7B parameters from scratch on a mixture of text and image data, establishing scaling laws with respect to a variety of uni- and cross-modal benchmarks. Our experiments show that Transfusion scales significantly better than quantizing images and training a language model over discrete image tokens. By introducing modality-specific encoding and decoding layers, we can further improve the performance of Transfusion models, and even compress each image to just 16 patches. We further demonstrate that scaling our Transfusion recipe to 7B parameters and 2T multi-modal tokens produces a model that can generate images and text on a par with similar scale diffusion models and language models, reaping the benefits of both worlds.

## Contents
- [Install](#install)
- [Train](#train)

## Install

```Shell
conda create -n transfusion python=3.10 -y
conda activate transfusion
pip install -e .
```

## Train

## Acknowledgements

- [Transfusion](https://arxiv.org/abs/2408.11039)
- [transfusion-pytorch](https://github.com/lucidrains/transfusion-pytorch)
- [Transfusion.torch](https://github.com/VachanVY/Transfusion.torch)
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [BLIP3o](https://github.com/JiuhaiChen/BLIP3o/tree/main)
- [MetaMorph](https://github.com/facebookresearch/metamorph)
- [OmniGen](https://github.com/VectorSpaceLab/OmniGen)
- [Show-o](https://github.com/showlab/Show-o)
- [Janus](https://github.com/deepseek-ai/Janus)
