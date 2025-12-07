import torch

from transformers.models.smolvlm.processing_smolvlm import SmolVLMProcessor
from transformers.models.smolvlm.modeling_smolvlm import SmolVLMModel


def patch_SmolVLMProcessor():  # noqa: N802
    SmolVLMProcessor.image_processor_class = "SmolVLMImageProcessorFast"


def patch_SmolVLM_amp(compilation_enabled: bool):
    if compilation_enabled:
        # compile friendly version
        def orig_inputs_merger(
            self, input_ids: torch.LongTensor, inputs_embeds: torch.Tensor, image_hidden_states: torch.Tensor
        ):
            _, patch_size, _ = image_hidden_states.shape
            image_mask = input_ids == self.config.image_token_id

            num_image_tokens = image_mask.sum(dim=1)
            blocks_per_sample = num_image_tokens // patch_size

            offsets = torch.nn.functional.pad(blocks_per_sample.cumsum(dim=0), (1, 0), value=0)
            block_offset = offsets[:-1]
            row_cum = image_mask.cumsum(dim=-1)
            chunk_idx = (row_cum - 1) // patch_size
            local_idx = (row_cum - 1) % patch_size
            block_idx = block_offset.unsqueeze(1) + chunk_idx

            max_blocks = image_hidden_states.shape[0] - 1
            safe_block_idx = torch.where(image_mask, block_idx, 0).clamp(max=max_blocks)
            safe_local_idx = torch.where(image_mask, local_idx, 0)

            gathered_image_embeds = image_hidden_states[safe_block_idx, safe_local_idx]
            merged_embeds = torch.where(image_mask.unsqueeze(-1), gathered_image_embeds, inputs_embeds)
            return merged_embeds
    else:
        orig_inputs_merger = SmolVLMModel.inputs_merger

    def patched_inputs_merger(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
        image_hidden_states: torch.Tensor
    ):
        image_hidden_states = image_hidden_states.to(inputs_embeds.dtype)
        return orig_inputs_merger(self, input_ids, inputs_embeds, image_hidden_states)

    SmolVLMModel.inputs_merger = patched_inputs_merger

