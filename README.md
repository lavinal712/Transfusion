## Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model

### Unofficial PyTorch Implementation

### [Paper](https://arxiv.org/abs/2312.04557)

> [**Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model**](https://arxiv.org/html/2408.11039)</br>
> Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, Omer Levy
> <br>Meta, Waymo, University of Southern California</br>

## Setup

`requirements.txt` will be released later.

```bash
conda create -n transfusion python=3.10
conda activate transfusion
pip install -r requirements.txt
```

## Training

```bash
accelerate launch --mixed_precision fp16 train.py --data_path /path/to/ImageNet/train
```

## Acknowledgments

The code is highly inspired by the following repositories:

- [DiT](https://github.com/facebookresearch/DiT)
- [fast-DiT](https://github.com/chuanyangjin/fast-DiT)
- [MonoFormer](https://github.com/MonoFormer/MonoFormer)
- [OmniGen](https://github.com/VectorSpaceLab/OmniGen)

There are some repositories do the same work:

- [transfusion-pytorch](https://github.com/lucidrains/transfusion-pytorch)
- [Transfusion.torch](https://github.com/VachanVY/Transfusion.torch)

## Citation

```bibtex
@misc{zhou2024transfusionpredicttokendiffuse,
      title={Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model}, 
      author={Chunting Zhou and Lili Yu and Arun Babu and Kushal Tirumala and Michihiro Yasunaga and Leonid Shamis and Jacob Kahn and Xuezhe Ma and Luke Zettlemoyer and Omer Levy},
      year={2024},
      eprint={2408.11039},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2408.11039}, 
}
```
