#!/usr/bin/env python

from collections import deque

import numpy as np
import torch
from scipy.fft import idct
from torch import Tensor, nn
from torch.profiler import record_function
from torchvision.transforms import CenterCrop, RandomCrop
from transformers import AutoModelForImageTextToText, AutoProcessor, LogitsProcessorList
from transformers.models.smolvlm.image_processing_smolvlm_fast import SmolVLMImageProcessorFast

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolandfast.configuration_smolandfast import SMOLANDFASTConfig
from lerobot.policies.smolandfast.monkey_patch import patch_SmolVLMProcessor
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE

PRECISION = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


class SMOLANDFASTPolicy(PreTrainedPolicy):
    """Wrapper class around PI0FAST tokenizer and SMOLANDFAST model to train and run inference within LeRobot."""

    config_class = SMOLANDFASTConfig
    name = "smolandfast"

    def __init__(
        self,
        config: SMOLANDFASTConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = SMOLANDFAST(config)

        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        vision_model_params = self.model.vlm.model.vision_model.parameters()
        connector_params = self.model.vlm.model.connector.parameters()
        text_model_params = self.model.vlm.model.text_model.parameters()
        optim_groups = [
            {"params": vision_model_params, "lr": self.config.vision_model_optimizer_lr},
            {"params": connector_params, "lr": self.config.connector_optimizer_lr},
            {"params": text_model_params, "lr": self.config.text_model_optimizer_lr},
        ]
        return optim_groups

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        raise NotImplementedError("Currently not implemented for SMOLANDFAST")

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            actions = self.model.generate_actions(batch)

            actions = actions[:, : self.config.n_action_steps]

            original_action_dim = self.config.action_feature.shape[
                0
            ]  # self.config.max_action_dim  # self.config.action_feature.shape[0]
            actions = actions[:, :, :original_action_dim]
            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        loss_dict = self.model.forward(batch)
        loss = loss_dict.pop("loss")
        return loss, loss_dict


class SMOLANDFAST(nn.Module):
    def __init__(self, config: SMOLANDFASTConfig):
        super().__init__()
        self.config = config

        self.torch_precision = PRECISION.get(config.precision, torch.float32)
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            self.config.vlm_checkpoint, torch_dtype=self.torch_precision
        )

        # Patch SmolVLMProcessor to enable using SmolVLMImageProcessorFast
        patch_SmolVLMProcessor()

        image_processor = SmolVLMImageProcessorFast.from_pretrained(
            self.config.vlm_checkpoint,
        )

        self.processor = AutoProcessor.from_pretrained(
            self.config.vlm_checkpoint,
            image_processor=image_processor,
            use_fast=True,
        )

        if config.scale_factor != 4:
            # if config factor is not 4 we need to recreate a linear layer in connector
            siglip_seq_len = 1024  # for 512x512 image with patch size 16 sequence len is 1024
            self.processor.image_seq_len = int(siglip_seq_len / (config.scale_factor**2))
            self.vlm.scale_factor = config.scale_factor
            self.vlm.model.connector.scale_factor = config.scale_factor
            input_size = self.vlm.config.vision_config.hidden_size * (config.scale_factor**2)
            output_size = self.vlm.config.text_config.hidden_size
            self.vlm.model.connector.modality_projection = nn.Linear(
                input_size, output_size, bias=False, dtype=self.torch_precision
            )

        self.fast_tokenizer = AutoProcessor.from_pretrained(self.config.fast_tokenizer_path, trust_remote_code=True)
        self.fast_skip_tokens = self.config.fast_skip_tokens
        self.max_input_seq_len = self.config.max_input_seq_len
        self.action_horizon = self.config.chunk_size
        self.action_dim = self.config.action_feature.shape[0]

        if config.freeze_vision_encoder:
            for param in self.vlm.model.vision_model.parameters():
                param.requires_grad = False

        if config.freeze_connector and config.scale_factor != 4:
            raise ValueError("If scale factor is equal 4(default value) connector should be unfeezed")

        if config.freeze_connector:
            for param in self.vlm.model.connector.parameters():
                param.requires_grad = False

        self.pad_token_id = self.processor.tokenizer.pad_token_id
        self.eos_token_id = self.processor.tokenizer.eos_token_id
        self.action_start_token = 49279
        print(f"pad token: {self.pad_token_id}, eos token: {self.eos_token_id}")

        self.do_crop = config.crop_shape is not None
        if self.do_crop:
            self.random_crop_fn = RandomCrop(config.crop_shape)
            self.center_crop_fn = CenterCrop(config.crop_shape)

    def create_prefix_tokens(self, states, images, lang_text, actions):
        device = states.device
        batch_size = states.shape[0]

        # Precompute bin edges on GPU
        bins = torch.linspace(-1, 1, self.config.n_state_bins + 1, device=device)[:-1]

        # Discretize directly on GPU
        discretized_states = torch.bucketize(states, bins) - 1  # shape: [B, state_dim]

        # Move the batched results to CPU only once for string formatting
        disc_states_cpu = discretized_states.detach().cpu().numpy()

        if actions is None:
            disc_actions_cpu = [""]*batch_size
        else:
            discretized_actions = torch.bucketize(actions, bins) - 1  # shape: [B, state_dim]
            disc_actions_cpu = discretized_actions.detach().cpu().numpy()
        
        # Build strings in batch
        prefix_texts = []
        for txt, disc_st, act in zip(lang_text, disc_states_cpu, disc_actions_cpu, strict=False):
            if actions is None:
                action_str = ""
            else:
                act = act.reshape(1,-1)[0, ...]
                action_str = " ".join(map(str, act.tolist()))

            cleaned = txt.lower().strip().replace("_", " ")
            state_str = " ".join(map(str, disc_st.tolist()))
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": f"Task: Push the T-shaped block onto the T-shaped target, State: {state_str}, Actions: ",
                        },
                    ],
                }
            ]
            if actions is not None:
                message.append(
                    {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{action_str}",
                        },
                    ],
                }
                )
            prefix_texts.append(message)

        prompts = [self.processor.apply_chat_template(m, add_generation_prompt=actions is None) for m in prefix_texts]
        # print(prompts)
        images = list(torch.unbind(images, dim=0))
        if self.do_crop:
            # Always use center crop for eval
            crop_fn = self.random_crop_fn if self.training else self.center_crop_fn
            images = [[crop_fn(img)] for img in images]
        else:
            images = [[img] for img in images]

        prefix_out = self.processor(
            images=images,
            text=prompts,
            do_resize=self.config.do_image_splitting,
            do_rescale=False,
            return_tensors="pt",
            padding=True,
            padding_side = "right" if actions is not None else "left"
        )

        return prefix_out

    def _act_tokens_to_llm_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        llm_tokens = self.processor.tokenizer.vocab_size - 1 - self.fast_skip_tokens - tokens
        return llm_tokens

    def _llm_tokens_to_act_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        fast_tokens = self.processor.tokenizer.vocab_size - 1 - self.fast_skip_tokens - tokens
        return fast_tokens

    def create_input_tokens(self, states, images, lang_text, actions=None):

        device = states.device

        prefix_out = self.create_prefix_tokens(states=states, images=images, lang_text=lang_text, actions=actions)
        prefix_out = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in prefix_out.items()}        
        if actions is None:
            loss_mask = None
        else:
            split_mask = torch.where(prefix_out["input_ids"] == self.action_start_token, 1, 0)
            loss_mask = torch.cumsum(split_mask, dim=-1)
            split_mask = torch.where(loss_mask == 2, 0, split_mask)
            loss_mask = torch.clip(loss_mask,0,1)
            # print(loss_mask)
            loss_mask = torch.where(split_mask == 1, 0, loss_mask) & prefix_out["attention_mask"]
            # print(split_mask)
            # print(loss_mask)
            # print(prefix_out)

        return prefix_out, loss_mask

    def forward(self, batch: dict[str, Tensor]):
        device = batch[OBS_STATE].device

        with record_function("create_input_tokens"):
            padded_outs, loss_mask  = self.create_input_tokens(
                states=batch[OBS_STATE],
                images=batch[OBS_IMAGE],
                lang_text=batch.get("task", ""),
                actions=batch[ACTION],
            )

        with record_function("forward"):
            outputs = self.vlm.forward(
                input_ids=padded_outs["input_ids"],
                attention_mask=padded_outs["attention_mask"],
                pixel_values=padded_outs["pixel_values"],
                pixel_attention_mask=padded_outs["pixel_attention_mask"],
                use_cache=self.config.use_cache,
            )

        with record_function("loss"):
            logits = outputs.logits

            loss_fct = nn.CrossEntropyLoss(reduction="none")

            # Shift left for next-step prediction
            logits = logits[:, :-1, :]
            targets = padded_outs["input_ids"][:, 1:].to(device)  # Shift targets
            loss_mask = loss_mask[:, 1:].to(device)  # Ensure correct shape

            # Compute per-token loss
            token_loss = loss_fct(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

            # Apply loss mask
            token_loss = token_loss * loss_mask.reshape(-1)

            # Compute final loss
            loss = token_loss.sum() / torch.clamp(loss_mask.sum(), min=1)

            # Return loss dictionary
            loss_dict = {"ce_loss": loss.item(), "loss": loss, "sequence_len": padded_outs["input_ids"].shape[-1]}
        return loss_dict

    def generate_actions(self, batch: dict[str, Tensor]):
        batch_size = batch[OBS_STATE].shape[0]

        padded_outs, _ = self.create_input_tokens(
            states=batch[OBS_STATE],
            images=batch[OBS_IMAGE],
            lang_text=batch.get("task", ""),
            actions=None,
        )

        input_len = padded_outs["input_ids"].shape[1]

        output_tokens = self.vlm.generate(
            input_ids=padded_outs["input_ids"],
            attention_mask=padded_outs["attention_mask"],
            pixel_values=padded_outs["pixel_values"],
            pixel_attention_mask=padded_outs["pixel_attention_mask"],
            use_cache=self.config.use_cache,
            max_new_tokens=self.config.max_decoding_steps,
            do_sample=False,
            num_beams=1,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )
        action_tokens = output_tokens[:, input_len:].tolist()
        # print(action_tokens)
        for seq in action_tokens:
            while seq and (seq[-1] == self.eos_token_id or seq[-1] == self.pad_token_id):
                seq.pop()

        decoded_actions = [self.processor.decode(tokens) for tokens in action_tokens]
        decoded_actions = [actions[1:].split(" ") for actions in decoded_actions]
        # print(decoded_actions)

        final_actions = []
        for actions in decoded_actions:
            valid = True
            
            if len(actions) != self.action_horizon * self.action_dim:
                valid = False

            for action in actions[:self.action_horizon * self.action_dim]:
                if action.isdigit():
                    if int(action) < 0 or int(action) >= self.config.n_state_bins:
                        valid = False
                        break
                else:
                    valid = False
                    break

            if not valid:
                final_actions.append([0] * self.action_horizon * self.action_dim)
            else:
                final_actions.append([int(act) for act in actions])
        
        return torch.tensor(final_actions, dtype=torch.float32).reshape(batch_size, -1, self.action_dim) / self.config.n_state_bins * 2 - 1
