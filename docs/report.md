# Report

Transfusion is a recipe for training a multi-modal model over discrete and continuous data. This repository provides a simple implementation of Transfusion.

We use TinyLlama-1.1B-Chat-v1.0 and ImageNet to train a Transfusion model. Timestep embedding `t` is added in the final layer block after transformer blocks.

Difference between the original Transfusion:

- no text-only input
- no up and down blocks of a U-Net
- no noise limit
- bad code