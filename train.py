import argparse
import logging
import os
from glob import glob
from time import time

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from data.ImageNet.imagenet import ImageNetDataset, ImageNetProcessor, ImageNetCollator
from diffusion import create_diffusion
from models.modeling_transfusion import Transfusion
from models.utils import create_logger, requires_grad


def parse_args(input_args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--image_size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vae", type=str, default="stabilityai/sd-vae-ft-ema")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=50_000)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main(args=None):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    accelerator = Accelerator()
    device = accelerator.device
    set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = "Transfusion"
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

    model = Transfusion.from_pretrained(args.model_name)
    model.initialize_weights()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.add_tokens(["<|soi|>", "<|eoi|>"], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    if accelerator.is_main_process:
        logger.info(f"Transfusion Parameters: {sum(p.numel() for p in model.parameters()):,}")

    diffusion = create_diffusion(timestep_respacing="")
    vae = AutoencoderKL.from_pretrained(args.vae).to(device)
    requires_grad(vae, False)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-08, weight_decay=1.5e-5)

    processor = ImageNetProcessor(tokenizer)
    collate_fn = ImageNetCollator()
    dataset = ImageNetDataset(args.data_path, processor)
    loader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    model.train()
    model, opt, loader = accelerator.prepare(model, opt, loader)

    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        for step, batch in enumerate(loader):
            x = batch["input_images"].to(device)
            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                position_ids=batch["position_ids"],
                labels=batch["labels"],
                images_position=batch["images_position"],
            )
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss1 = loss_dict["loss"].mean()
            loss2 = loss_dict["lm_loss"]
            loss = loss1 * 5 + loss2
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item()
                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                running_loss = 0
                log_steps = 0
                start_time = time()

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/model_{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                model.save_pretrained(checkpoint_dir)
                processor.tokenizer.save_pretrained(checkpoint_dir)
                model.config.save_pretrained(checkpoint_dir)
                logger.info(f"Saved checkpoint to {checkpoint_dir}")

    model.eval()

    if accelerator.is_main_process:
        logger.info("Done!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
