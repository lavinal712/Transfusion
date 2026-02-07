# 💉🩸 Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model

<p align="center">
  <a href="https://arxiv.org/abs/2408.11039">
    <img alt="Build" src="https://img.shields.io/badge/arXiv%20paper-2408.11039-b31b1b.svg">
  </a>
  <a href="https://huggingface.co/lavinal712/transfusion-7b">
    <img alt="Build" src="https://img.shields.io/badge/HF%20Model-🤗-yellow">
  </a>
</p>

## Claims

⚠️ This is an **unofficial implementation** of Transfusion.

There are many great repositories about Transfusion on Github, including [transfusion-pytorch](https://github.com/lucidrains/transfusion-pytorch) and [Transfusion.torch](https://github.com/VachanVY/Transfusion.torch). This repository follows the overall architecture of [LLaVA](https://github.com/haotian-liu/LLaVA) and adopts LLaMA2 / Vicuna as the backbone Transformer, enabling Transfusion to inherit strong text modeling capabilities. Since the original paper does not specify the implementation details of the linear layers, we adopt the patch embedding and final layer from [DiT](https://github.com/facebookresearch/DiT) to incorporate diffusion timesteps into the model.

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
* [x] Train an understanding model
* [ ] Train a generation model

### Model Architecture

* [x] Support LLaMA2 / Vicuna
* [x] Bidirectional attention
* [x] U-Net blocks
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

We use [liuhaotian/LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) as the dataset.

```bash
bash scripts/train.sh
```

<!--
[LLaVA/issues/1231](https://github.com/haotian-liu/LLaVA/issues/1231)
-->

## Evaluation

| Model | LLM | Vision Encoder | Image Size | VQAv2 | GQA | VisWiz | SciQA-IMG | TextVQA | POPE | MME | MMBench | MMBench-CN | SEED-Bench | LLaVA-Wild | MMVet |
|-------|-----|----------------|------------|-------|-----|--------|-----------|---------|------|-----|---------|------------|------------|------------|-------|
| LLaVA-1.5-7B | Vicuna-7B-v1.5 | CLIP-ViT-L-336px | 336² | 78.5 | 62.0 | 50.0 | 66.8 | 58.2 | 85.9 | 1510.7 | 64.3 | 58.3 | 58.6 | 65.4 | 31.1 |
| Transfusion-7B | Vicuna-7B-v1.5 | Transfusion-VAE | 256² | - | 42.7 | - | 60.6 | - | 59.7 | - | 25.5 | 17.6 | - | - | - |

We use [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) as the evaluation tool.

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

## Citations

```
@article{zhou2024transfusion,
  title={Transfusion: Predict the next token and diffuse images with one multi-modal model},
  author={Zhou, Chunting and Yu, Lili and Babu, Arun and Tirumala, Kushal and Yasunaga, Michihiro and Shamis, Leonid and Kahn, Jacob and Ma, Xuezhe and Zettlemoyer, Luke and Levy, Omer},
  journal={arXiv preprint arXiv:2408.11039},
  year={2024}
}
```
