# 💉🩸 Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2408.11039-b31b1b.svg)](https://arxiv.org/abs/2408.11039)&nbsp;

</div>

## Claims

⚠️ This is an **unofficial implementation** of Transfusion.

There are many great repositories about Transfusion on Github, including [transfusion-pytorch](https://github.com/lucidrains/transfusion-pytorch) and [Transfusion.torch](https://github.com/VachanVY/Transfusion.torch). This repository follows the overall architecture of [LLaVA](https://github.com/haotian-liu/LLaVA) and adopts LLaMA2 / Vicuna as the backbone Transformer, enabling Transfusion to inherit strong text modeling capabilities. Since the original paper does not specify the implementation details of the linear layers, we adopt the timestep embedding strategy from [OmniGen](https://github.com/VectorSpaceLab/OmniGen) to incorporate diffusion timesteps into the model.  

## Introduction

### Abstract

> We introduce Transfusion, a recipe for training a multi-modal model over discrete and continuous data. Transfusion combines the language modeling loss function (next token prediction) with diffusion to train a single transformer over mixed-modality sequences. We pretrain multiple Transfusion models up to 7B parameters from scratch on a mixture of text and image data, establishing scaling laws with respect to a variety of uni- and cross-modal benchmarks. Our experiments show that Transfusion scales significantly better than quantizing images and training a language model over discrete image tokens. By introducing modality-specific encoding and decoding layers, we can further improve the performance of Transfusion models, and even compress each image to just 16 patches. We further demonstrate that scaling our Transfusion recipe to 7B parameters and 2T multi-modal tokens produces a model that can generate images and text on a par with similar scale diffusion models and language models, reaping the benefits of both worlds.

### Overview

A high-level illustration of Transfusion. A single transformer perceives, processes, and produces data of every modality. Discrete (text) tokens are processed autoregressively and trained on the next token prediction objective. Continuous (image) vectors are processed together in parallel and trained on the diffusion objective. Marker BOI and EOI tokens separate the modalities.

![](images/model_diagram-crop.jpg)

We convert images to and from latent representations using a pretrained VAE, and then into patch representations with either a simple linear layer or U-Net down blocks.

Expanding on the causal mask, Transfusion allows patches of the same image to condition on each other.

<div align="center">
  <img src="images/model_vision-crop.jpg" width="44%" />
  <img src="images/attention_mask-crop.jpg" width="50%" />
</div>

## TODO

### Training

* [x] Train a continuous VAE
* [ ] Train an understanding model
* [ ] Train a generation model

### Model Architecture

* [x] Support LLaMA2 / Vicuna
* [x] Transfusion Attention
* [ ] U-Net blocks
* [ ] Noise limit
* [ ] Support Qwen2
* [ ] Support flow matching

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
- [DiT](https://github.com/facebookresearch/DiT)
- [BLIP3o](https://github.com/JiuhaiChen/BLIP3o/tree/main)
- [MetaMorph](https://github.com/facebookresearch/metamorph)
- [OmniGen](https://github.com/VectorSpaceLab/OmniGen)
- [Show-o](https://github.com/showlab/Show-o)
- [Janus](https://github.com/deepseek-ai/Janus)
