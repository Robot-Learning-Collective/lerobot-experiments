from __future__ import annotations
from contextlib import nullcontext
import torch


def run_model_inference(policy, observation, device):
    """Runs model inference using preprocessed observations from dataloader and returns the policy output,
     bypassing action queue / temporal ensembling."""
    policy.eval()

    use_amp = policy.config.use_amp
    batch = {}
    for name in observation:
        if name.startswith("observation"):
            observation[name] = observation[name].unsqueeze(0).to(device, non_blocking=True)
            batch[name] = observation[name]
        elif name == "task":
            batch[name] = observation[name]
    with (
            torch.inference_mode(),
            torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
        ):
        actions = policy.predict_action_chunk(batch)
    return actions.to("cpu")
