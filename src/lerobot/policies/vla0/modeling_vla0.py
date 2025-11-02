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
from lerobot.policies.vla0.configuration_vla0 import VLA0Config
from lerobot.policies.vla0.monkey_patch import patch_SmolVLMProcessor
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE

PRECISION = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


class VLA0Policy(PreTrainedPolicy):
    """Wrapper class around VLA0 model to train and run inference within LeRobot."""

    config_class = VLA0Config
    name = "vla0"

    def __init__(
        self,
        config: VLA0Config,
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

        self.model = VLA0(config)

        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        return self.model.parameters()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        raise NotImplementedError("Currently not implemented for VLA0")

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


class VLA0(nn.Module):
    def __init__(self, config: VLA0Config):
        super().__init__()
        self.config = config

        self.precision = PRECISION.get(config.precision, torch.float32)
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            self.config.vlm_checkpoint, dtype=self.precision
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

        self.action_horizon = self.config.chunk_size
        self.action_dim = self.config.action_feature.shape[0]

        if config.freeze_vision_encoder:
            for param in self.vlm.model.vision_model.parameters():
                param.requires_grad = False

        self.pad_token_id = self.processor.tokenizer.pad_token_id
        self.eos_token_id = self.processor.tokenizer.eos_token_id

        self.image_keys = self.config.image_features.keys()

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
            if self.config.relative_actions:
                actions = actions - states.unsqueeze(1)
            discretized_actions = torch.bucketize(actions, bins) - 1  # shape: [B, state_dim]
            disc_actions_cpu = discretized_actions.detach().cpu().numpy()
        
        # Build strings in batch
        prompts = []
        for txt, disc_st, act in zip(lang_text, disc_states_cpu, disc_actions_cpu, strict=False):
            
            task_cleaned = txt.lower().strip().replace("_", " ")
            state_str = " ".join(map(str, disc_st.tolist()))

            if self.config.relative_actions:
                prefix = f"Task: {task_cleaned}, Actions: "
            else:
                prefix = f"Task: {task_cleaned}, State: {state_str}, Actions: "

            message = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image"} for _ in range(len(images))],
                        {
                            "type": "text",
                            "text": prefix,
                        },
                    ],
                }
            ]

            if actions is not None:
                act = act.reshape(1,-1)[0, ...]
                action_str = " ".join(map(str, act.tolist()))
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
            prompts.append(self.processor.apply_chat_template(message, add_generation_prompt=actions is None))

        images = {camera_name: list(torch.unbind(camera_images, dim=0)) for camera_name, camera_images in images.items()}

        images_reshaped = []
        for imgs in zip(*images.values()):
            if self.do_crop:
                crop_fn = self.random_crop_fn if self.training else self.center_crop_fn
                images_reshaped.append([crop_fn(img) for img in imgs])
            else:
                images_reshaped.append([img for img in imgs])

        prefix_out = self.processor(
            images=images_reshaped,
            text=prompts,
            do_resize=self.config.do_image_splitting,
            do_rescale=False,
            return_tensors="pt",
            padding=True,
            padding_side = "right" if actions is not None else "left"
        )

        return prefix_out

    def create_input_tokens(self, states, images, lang_text, actions=None):

        device = states.device

        prefix_out = self.create_prefix_tokens(states=states, images=images, lang_text=lang_text, actions=actions)
        prefix_out = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in prefix_out.items()}
       
        if actions is None:
            loss_mask = None
        else:
            split_mask = torch.where(prefix_out["input_ids"] == self.config.start_actions_token, 1, 0)
            loss_mask = torch.cumsum(split_mask, dim=-1).clamp(0, 1) & prefix_out["attention_mask"]

        return prefix_out, loss_mask
    
    def prepare_images(self, batch):
            """Preprocess LeRobot batch into inputs"""
            images = {}
            present_img_keys = [key for key in self.image_keys if key in batch]
            if len(present_img_keys) == 0:
                raise ValueError(
                    f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
                )

            for key in self.image_keys:
                if key in present_img_keys:
                    img = batch[key]

                images[key] = img
            return images
    
    def forward(self, batch: dict[str, Tensor]):
        device = batch[OBS_STATE].device

        with record_function("create_input_tokens"):

            images = self.prepare_images(batch)

            padded_outs, loss_mask  = self.create_input_tokens(
                states=batch[OBS_STATE],
                images=images,
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

    def generate_actions(self, batch: dict[str, torch.Tensor]):
        device = next(self.vlm.parameters()).device
        batch_size = batch[OBS_STATE].shape[0]

        images = self.prepare_images(batch)

        # --- 1. Prepare inputs directly on GPU
        padded_outs, _ = self.create_input_tokens(
            states=batch[OBS_STATE],
            images=images,
            lang_text=batch.get("task", ""),
            actions=None,
        )

        input_len = padded_outs["input_ids"].shape[1]

        # --- 2. Model inference on GPU (no gradient)
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

    
        # --- 3. Slice to generated part
        action_tokens = output_tokens[:, input_len:]

        # --- 4. Remove padding/eos tokens efficiently on GPU
        valid_mask = (action_tokens != self.eos_token_id) & (action_tokens != self.pad_token_id)
        # Replace invalid tokens with pad (or zero) for decoding
        action_tokens = torch.where(valid_mask, action_tokens, torch.tensor(self.pad_token_id, device=device))

        # --- 5. Decode in batch (vectorized)
        # Decode all at once instead of Python loop
        decoded_texts = self.processor.batch_decode(action_tokens, skip_special_tokens=True)

        # --- 6. Convert decoded strings to numeric actions (vectorized)
        final_actions = []
        n_expected = self.action_horizon * self.action_dim
        n_bins = self.config.n_state_bins

        for text in decoded_texts:
            actions = text.strip().split()
            if len(actions) != n_expected or not all(a.isdigit() and 0 <= int(a) < n_bins for a in actions):
                final_actions.append(torch.zeros(n_expected, device=device, dtype=torch.long))
            else:
                final_actions.append(torch.tensor([int(a) for a in actions], device=device))
   
        discretized_actions = torch.stack(final_actions, dim=0).reshape(batch_size, -1, self.action_dim)

        # Assuming same bin setup
        bins = torch.linspace(-1, 1, self.config.n_state_bins + 1, device=device)

        # Compute bin centers (midpoints between edges)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])  # shape: [n_state_bins]

        # Map discretized indices back to continuous states
        reconstructed_actions = bin_centers[discretized_actions.clamp(0, self.config.n_state_bins - 1)]
        if self.config.relative_actions:
            reconstructed_actions += batch[OBS_STATE].unsqueeze(1)
        
        return reconstructed_actions