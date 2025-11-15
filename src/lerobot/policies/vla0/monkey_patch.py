import torch

from transformers.models.smolvlm.processing_smolvlm import SmolVLMProcessor
from transformers.models.smolvlm.modeling_smolvlm import SmolVLMModel


def patch_SmolVLMProcessor():  # noqa: N802
    SmolVLMProcessor.image_processor_class = "SmolVLMImageProcessorFast"


def patch_SmolVLM_amp():
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
