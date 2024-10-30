import argparse

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image
from transformers import AutoTokenizer

from data.ImageNet.imagenet import ImageNetDataset, ImageNetProcessor, ImageNetCollator
from models.modeling_transfusion import Transfusion


def parse_args(input_args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True, default="Transfusion")
    parser.add_argument("--vae", type=str, default="stabilityai/sd-vae-ft-ema")
    parser.add_argument("--image_size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--num_sampling_steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    latent_size = args.image_size // 8
    model = Transfusion.from_pretrained(args.model_name).to(device)
    model.eval()
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(args.vae).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    processor = ImageNetProcessor(tokenizer, resolution=args.image_size)
    collate_fn = ImageNetCollator()

    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    with open("./data/ImageNet/imagenet1000_clsidx_to_labels.txt", "r") as f:
        id2label = eval(f.read())

    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    input_z = torch.randn(n, 3, args.image_size, args.image_size, device=device)
    y = torch.tensor(class_labels, device=device)
    y = list(map(lambda id: id2label[int(id)], y))
    mllm_inputs = [processor.process_multimodel_input(y[i], input_z[i]) for i in range(n)]
    input_ids, attention_mask, position_ids, labels, input_images, images_position = collate_fn.process_mllm_input(mllm_inputs)
    input_ids_null = torch.zeros_like(input_ids)

    z = torch.cat((z, z), 0).to(device)
    input_ids = torch.cat([input_ids, input_ids_null], 0).to(device)
    input_images = torch.cat([input_images, input_images], 0).to(device)
    attention_mask = torch.cat([attention_mask, attention_mask], 0).to(device)
    position_ids = torch.cat([position_ids, position_ids], 0).to(device)
    labels = torch.cat([labels, labels], 0).to(device)
    images_position.extend(images_position)
    model_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        labels=labels,
        images_position=images_position,
        cfg_scale=args.cfg_scale,
    )

    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)
    samples = vae.decode(samples / 0.18215).sample

    save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    args = parse_args()
    main(args)
