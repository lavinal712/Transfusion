from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..transfusion_arch import TransfusionMetaModel, TransfusionMetaForCausalLM


class TransfusionConfig(LlamaConfig):
    model_type = "transfusion_llama"


class TransfusionLlamaModel(TransfusionMetaModel, LlamaModel):
    config_class = TransfusionConfig

    def __init__(self, config: LlamaConfig):
        super(TransfusionLlamaModel, self).__init__(config)


class TransfusionLlamaForCausalLM(LlamaForCausalLM, TransfusionMetaForCausalLM):
    config_class = TransfusionConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = TransfusionLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        packed_image_inputs = {}
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                packed_image_inputs
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        image_positions = packed_image_inputs.pop("image_positions", None)
        latents = packed_image_inputs.pop("latents", None)
        timesteps = packed_image_inputs.pop("timesteps", None)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        total_loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            if image_positions is not None:
                image_hidden_states = hidden_states[image_positions.bool()]
                image_hidden_states = image_hidden_states.view(hidden_states.shape[0], -1, hidden_states.shape[-1])
                output_latents = self.encode_image_embeds(image_hidden_states, **packed_image_inputs)

                model_fn = self.forward_images
                loss_dict = self.get_model().diffusion.training_losses(model_fn, latents, timesteps, model_output=output_latents)
                image_loss = loss_dict["loss"].mean()

                image_weight = 0.5
                total_loss = loss + image_weight * image_loss
            else:
                total_loss = loss

        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def forward_images(
        self,
        images: Optional[torch.FloatTensor] = None,
        timesteps: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        packed_image_inputs = {}
        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
            packed_image_inputs
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            labels,
            images,
            image_sizes,
            timesteps=timesteps,
            is_latent=True,
            add_noise=False,
        )
        image_positions = packed_image_inputs.pop("image_positions", None)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        if image_positions is not None:
            image_hidden_states = hidden_states[image_positions.bool()]
            image_hidden_states = image_hidden_states.view(hidden_states.shape[0], -1, hidden_states.shape[-1])
            output_latents = self.encode_image_embeds(image_hidden_states, **packed_image_inputs)

        return output_latents

    def forward_images_with_cfg(
        self,
        cfg_scale: float = 1.0,
        **kwargs,
    ):
        x = kwargs.pop("images", None)
        t = kwargs.pop("timesteps", None)
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward_images(combined, t, **kwargs)
        eps, rest = model_out[:, :8], model_out[:, 8:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    @torch.no_grad()
    def generate_images(
        self,
        inputs: Optional[torch.Tensor] = None,
        input_size: int = 32,
        cfg_scale: float = 1.0,
        **kwargs,
    ):
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        using_cfg = cfg_scale > 1.0
        num_images = inputs.shape[0]
        noise = torch.randn(num_images, 8, input_size, input_size, device=inputs.device)

        if using_cfg:
            noise = torch.cat([noise, noise], 0)
            model_kwargs = dict(
                input_ids=inputs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cfg_scale=cfg_scale,
            )
            sample_fn = self.forward_images_with_cfg
        else:
            model_kwargs = dict(
                input_ids=inputs,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            sample_fn = self.forward_images

        samples = self.get_model().diffusion.p_sample_loop(
            sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=self.get_model().device
        )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)

        samples = self.get_vision_tower.decode(samples)
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        return dict(images=[Image.fromarray(image) for image in samples])

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("transfusion_llama", TransfusionConfig)
AutoModelForCausalLM.register(TransfusionConfig, TransfusionLlamaForCausalLM)
