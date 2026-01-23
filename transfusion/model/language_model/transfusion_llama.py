from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

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

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                image_positions,
                latents,
                noised_latents,
                timesteps,
                noise
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
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
                image_hidden_states = (hidden_states * image_positions.unsqueeze(-1).to(hidden_states.dtype)).contiguous()
                output_image_features = self.encode_image_embeds(image_hidden_states)
                output_latents = self.unpatchify(output_image_features)

                terms = {}

                B, C = output_latents.shape[:2]
                assert output_latents.shape == (B, C * 2, *latents.shape[2:])
                output_latents, output_var_values = torch.split(output_latents, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = torch.cat([output_latents.detach(), output_var_values], dim=1)
                terms["vb"] = self.diffusion._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=latents,
                    x_t=noised_latents,
                    t=timesteps,
                    clip_denoised=False,
                )["output"]
                assert output_latents.shape == noise.shape == latents.shape
                terms["mse"] = mean_flat((noise - output_latents) ** 2)
                if "vb" in terms:
                    terms["loss"] = terms["mse"] + terms["vb"]
                else:
                    terms["loss"] = terms["mse"]
                
                image_loss = terms["loss"].mean()

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
                _,
                _,
                _,
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

    def generate_images(
        self,
        text
    ):
        pass

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
