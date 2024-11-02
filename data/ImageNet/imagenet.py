import os
import yaml

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer


def synset2idx(path_to_yaml="./index_synset.yaml"):
    with open(path_to_yaml) as f:
        di2s = yaml.safe_load(f)
    return dict((v, k) for k, v in di2s.items())


def idx2label(path_to_txt="./imagenet1000_clsidx_to_labels.txt"):
    with open(path_to_txt, "r") as f:
        id2l = eval(f.read())
    return id2l


def synset2label(path_to_txt="./synset_words.txt"):
    s2l = {}
    with open(path_to_txt, "r") as f:
        for line in f.readlines():
            line = line.strip()
            synset = line.split()[0]
            label = line[len(synset) + 1:]
            s2l[synset] = label
    return s2l


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class ImageNetDataset(Dataset):
    def __init__(self, data_root, processor):
        super().__init__()
        self.data_root = data_root
        self.processor = processor

        self.synset2label = synset2label()

        self.data = []
        for dirname in os.listdir(data_root):
            for filename in os.listdir(os.path.join(data_root, dirname)):
                self.data.append({
                    "image_path": os.path.join(data_root, dirname, filename),
                    "caption": self.synset2label[dirname]
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.processor(self.data[index])


class ImageNetProcessor:
    def __init__(
            self,
            tokenizer,
            resolution: int = 256,
    ):
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, resolution)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])

    @classmethod
    def from_pretrained(cls, model_name):
        if not os.path.exists(model_name):
            cache_folder = os.getenv('HF_HUB_CACHE')
            model_name = snapshot_download(repo_id=model_name, cache_dir=cache_folder, allow_patterns="*.json")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return cls(tokenizer)

    def process_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return self.transform(image)

    def process_text(self, text):
        messages1 = [
            {"role": "user", "content": f"Generate an image containing\n{text}"},
            {"role": "assistant", "content": "<|soi|><image><|eoi|>"}
        ]
        messages2 = [
            {"role": "user", "content": "<|soi|><image><|eoi|>\nWhat is in the image?"},
            {"role": "assistant", "content": f"{text}"}
        ]
        messages = messages1 if np.random.rand() < 0.8 else messages2
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return text

    def process_multimodel_input(self, text, input_image):
        assert input_image is not None

        text = self.process_text(text)
        image_token = "<image>"
        input_ids_before_image, input_ids_after_image = [self.tokenizer(t).input_ids for t in text.split(image_token)]

        input_ids = input_ids_before_image
        image_start_index = len(input_ids_before_image)
        image_seq_length = input_image.size(-2) * input_image.size(-1) // 16 // 16
        image_position = [image_start_index, image_start_index + image_seq_length]
        input_ids.extend([0] * image_seq_length)
        input_ids.extend(input_ids_after_image)

        return {"input_ids": input_ids, "input_image": input_image, "image_position": image_position}

    def __call__(self, inputs):
        text = inputs["caption"]
        image = self.process_image(inputs["image_path"])
        return self.process_multimodel_input(text, image)


class ImageNetCollator:
    def __init__(self, pad_token_id=0, hidden_size=2048):
        self.pad_token_id = pad_token_id
        self.hidden_size = hidden_size

    def pad_input_ids(self, input_ids):
        max_seq_length = max([len(x) for x in input_ids])
        padded_input_ids = []
        attention_mask = []

        for i, _ in enumerate(input_ids):
            curr_input_ids = input_ids[i]
            curr_seq_length = len(curr_input_ids)
            padded_seq_length = max_seq_length - curr_seq_length
            if padded_seq_length > 0:
                padded_input_ids.append(curr_input_ids + [self.pad_token_id] * padded_seq_length)
                attention_mask.append([1] * curr_seq_length + [0] * padded_seq_length)
            else:
                padded_input_ids.append(curr_input_ids)
                attention_mask.append([1] * curr_seq_length)

        return torch.LongTensor(padded_input_ids), torch.LongTensor(attention_mask)

    def create_position(self, attention_mask):
        position_ids = []
        padded_seq_length = attention_mask.size(-1)
        for i, mask in enumerate(attention_mask):
            seq_length = torch.sum(mask)
            position_ids.append([i for i in range(seq_length)] + [0] * (padded_seq_length - seq_length))

        return torch.LongTensor(position_ids)

    def create_attention_mask(self, attention_mask, images_position):
        seq_length = attention_mask.size(-1)
        attn_mask = []
        for i, mask in enumerate(attention_mask):
            image_start_index, image_end_index = images_position[i]
            curr_mask = torch.triu(torch.ones(seq_length, seq_length))
            curr_mask[image_start_index: image_end_index, image_start_index: image_end_index] = 1
            attn_mask.append(curr_mask)
        attn_mask = torch.stack(attn_mask, dim=0).unsqueeze(1)

        return attn_mask

    def process_mllm_input(self, mllm_inputs):
        input_ids = [x["input_ids"] for x in mllm_inputs]
        input_images = torch.stack([x["input_image"] for x in mllm_inputs], dim=0)
        images_position = [x["image_position"] for x in mllm_inputs]

        padded_input_ids, attention_mask = self.pad_input_ids(input_ids)
        position_ids = self.create_position(attention_mask)
        attention_mask = self.create_attention_mask(attention_mask, images_position)
        labels = torch.where(padded_input_ids == 0, -100, padded_input_ids)

        return padded_input_ids, attention_mask, position_ids, labels, input_images, images_position

    def __call__(self, mllm_inputs):
        input_ids, attention_mask, position_ids, labels, input_images, images_position = self.process_mllm_input(mllm_inputs)
        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "labels": labels,
            "input_images": input_images,
            "images_position": images_position,
        }

        return data
