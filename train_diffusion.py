import torch
import torch.nn as nn
import torch.optim as optim
import os


from torch.optim.lr_scheduler import CosineAnnealingLR

from pathlib import Path
from tqdm import tqdm
import torch
import random
from lerobot.datasets.utils import cycle

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.smolandfast.tokenizer_with_diffusion import DiffusionAE

from lerobot.processor.normalize_processor import NormalizerProcessorStep
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

import wandb


def _build_min_max_normalizer(dataset: LeRobotDataset) -> NormalizerProcessorStep:
    action_shape = tuple(dataset.meta.shapes["action"])
    features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=action_shape)}
    norm_map = {FeatureType.ACTION: NormalizationMode.MIN_MAX}
    normalizer = NormalizerProcessorStep.from_lerobot_dataset(dataset, features, norm_map)
    return normalizer


def apply_affine_transform(points_tensor, transform_type, shift_amount=0.1):
    """
    Applies a specified affine transformation to a tensor of 2D points.

    The points are assumed to be in a space defined from [-1, 1] for both axes.
    Shift operations are only applied if the resulting points all remain
    within this [-1, 1] boundary.

    Args:
        points_tensor (torch.Tensor): A tensor of shape (N, 2) or (B, N, 2),
                                      where B is the batch size and N is
                                      the number of points.
        transform_type (str): The type of transformation to apply.
            Valid options are:
            - 'rotate_left_90'
            - 'rotate_right_90'
            - 'rotate_180'
            - 'mirror_y_axis' (reflect across the vertical axis, x -> -x)
            - 'mirror_x_axis' (reflect across the horizontal axis, y -> -y)
            - 'shift_left'
            - 'shift_right'
            - 'shift_up'
            - 'shift_down'
        shift_amount (float, optional): The magnitude of the shift for
                                        shift operations. Defaults to 0.1.

    Returns:
        torch.Tensor: The transformed tensor of points, with the same shape
                      as the input tensor. Returns the original tensor
                      if a shift is out of bounds or
                      if the transform_type is unknown.
    """
    
    # --- Input Validation ---
    if not isinstance(points_tensor, torch.Tensor):
        raise TypeError("Input 'points_tensor' must be a torch.Tensor")

    if points_tensor.numel() == 0:
        return points_tensor  # Return empty tensor if input is empty

    # --- Batch Handling ---
    is_batched = True
    if points_tensor.dim() == 2:
        if points_tensor.shape[1] != 2:
            raise ValueError(f"Input tensor must have shape (N, 2) or (B, N, 2), but got {points_tensor.shape}")
        is_batched = False
        points_tensor = points_tensor.unsqueeze(0) # Add a batch dimension: (1, N, 2)
    elif points_tensor.dim() == 3:
        if points_tensor.shape[2] != 2:
            raise ValueError(f"Input tensor must have shape (N, 2) or (B, N, 2), but got {points_tensor.shape}")
        # Already in (B, N, 2) format
    else:
        raise ValueError(f"Input tensor must have 2 or 3 dimensions (N, 2) or (B, N, 2), but got {points_tensor.dim()} dimensions")

    # Ensure tensor is float for calculations
    points_tensor = points_tensor.float()

    # --- Transformation Logic ---
    # Create a copy to avoid in-place modification of the original
    transformed_points = points_tensor.clone()

    # --- Result Formatting Helper ---
    def format_output(tensor_to_return):
        """Removes the batch dimension if it wasn't present in the input."""
        if not is_batched:
            return tensor_to_return.squeeze(0)
        return tensor_to_return

    if transform_type == 'rotate_left_90':
        # (x, y) -> (-y, x)
        transformed_points[..., 0] = -points_tensor[..., 1]
        transformed_points[..., 1] =  points_tensor[..., 0]
        return format_output(transformed_points)

    elif transform_type == 'rotate_right_90':
        # (x, y) -> (y, -x)
        transformed_points[..., 0] =  points_tensor[..., 1]
        transformed_points[..., 1] = -points_tensor[..., 0]
        return format_output(transformed_points)

    elif transform_type == 'rotate_180':
        # (x, y) -> (-x, -y)
        transformed_points = -points_tensor
        return format_output(transformed_points)

    elif transform_type == 'mirror_y_axis':
        # (x, y) -> (-x, y)
        transformed_points[..., 0] = -points_tensor[..., 0]
        return format_output(transformed_points)

    elif transform_type == 'mirror_x_axis':
        # (x, y) -> (x, -y)
        transformed_points[..., 1] = -points_tensor[..., 1]
        return format_output(transformed_points)

    # --- Shift Operations with Boundary Checks ---

    # Define shift vectors
    shifts = {
        'shift_left':  torch.tensor([-shift_amount, 0.0], device=points_tensor.device, dtype=points_tensor.dtype),
        'shift_right': torch.tensor([ shift_amount, 0.0], device=points_tensor.device, dtype=points_tensor.dtype),
        'shift_up':    torch.tensor([0.0,  shift_amount], device=points_tensor.device, dtype=points_tensor.dtype),
        'shift_down':  torch.tensor([0.0, -shift_amount], device=points_tensor.device, dtype=points_tensor.dtype)
    }

    if transform_type in shifts:
        shift_vector = shifts[transform_type]
        # Apply the shift (broadcasts correctly to (B, N, 2))
        candidate_points = points_tensor + shift_vector

        # Check if all points are within the [-1, 1] boundary
        # We use .all() to ensure every single point is valid across all batches
        if (candidate_points >= -1.0).all() and (candidate_points <= 1.0).all():
            return format_output(candidate_points)
        else:
            # print(f"Warning: Shift '{transform_type}' with amount {shift_amount} would move points "
            #       "out of the [-1, 1] bounds. Returning original tensor.")
            return format_output(points_tensor)  # Return original (but formatted)
    
    else:
        print(f"Warning: Unknown 'transform_type' specified: '{transform_type}'. "
              "Returning original tensor.")
        return format_output(points_tensor)  # Return original (but formatted)


TRANSFORMS = (
    'rotate_left_90',
    'rotate_right_90',
    'rotate_180',
    'mirror_y_axis',
    'mirror_x_axis',
    'shift_left',
    'shift_right',
    'shift_up',
    'shift_down',
)

if __name__ == '__main__':
    DATASET_PATH = "lerobot/pusht"
    output_directory = Path("outputs/train/example_pusht")
    output_directory.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    
    random.seed(42)

    # Store hyperparameters in a dictionary for easy logging
    hyperparameters = {
        "learning_rate": 3e-5,
        "epochs": 30000,
    
        "policy": {
            # encoder
            "base_features": 24,
            "ratios": [1, 2, 1],
            "num_residual_layers": 6,
            "num_lstm_layers": 4,
            "horizon": 8,
            
            # vq
            "encoded_dim": 2,
            "emdedding_dim": 128,
            "vocab_size": 512,
            
            # diffusion
            "n_layer": 12,
            "n_head": 8,
            "n_emb": 768,
            "n_cond_layers": 4,
            
            "num_train_timesteps": 50,
            "prediction_type": 'epsilon',
        },
    }

    use_wandb = True
    checkpoint_path = "diffusion_8_high_lr.pth"

    dataset_metadata = LeRobotDatasetMetadata(DATASET_PATH)
    horizon = hyperparameters['policy']['horizon']

    delta_timestamps = {
            "action": [i / dataset_metadata.fps for i in range(horizon)],
        }

    # We can then instantiate the dataset with these delta_timestamps configuration.
    dataset = LeRobotDataset(DATASET_PATH, delta_timestamps=delta_timestamps)
    normalizer = _build_min_max_normalizer(dataset)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=32,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=3,
    )
    dl_iter = cycle(dataloader)

    # --- CHECKPOINTING SETUP ---
    start_epoch = 0
    wandb_run_id = checkpoint_path.split(".")[0]

    # Model configuration
    model = DiffusionAE(**hyperparameters['policy']).to(device)

    optimizer = optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=hyperparameters['epochs'],
        eta_min=0.01 * hyperparameters['learning_rate'],
    )

    # CHECKPOINT: Load if checkpoint exists
    if os.path.exists(checkpoint_path):
        print("--- Resuming training from checkpoint ---")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        wandb_run_id = checkpoint['wandb_run_id']
    
    # WANDB: Initialize a new run or resume an existing one
    if use_wandb:
        wandb.init(
            project="seanet-autoencoder-quantized",
            config=hyperparameters,
            id=wandb_run_id,
            resume="allow" # Allows resuming if id is set
        )
    
    def velocity_loss(pred, target):
        """Calculates the MSE of the first-order differences (velocity)."""
        pred_vel = torch.diff(pred, dim=1) # Difference along the sequence axis
        target_vel = torch.diff(target, dim=1)
        return nn.MSELoss()(pred_vel, target_vel)

    def criterion(
        reconstructed,
        sample_data,
        l2_loss=[2, 4],
        l1_coeff=0.5,
        l2_coeff=1.0,
        vel_coeff=0.7,
    ):
        loss = l1_coeff * nn.L1Loss()(reconstructed, sample_data)
        if l2_loss:
            loss += l2_coeff * torch.sqrt(
                nn.MSELoss()(reconstructed[:, l2_loss, :], sample_data[:, l2_loss, :])
            )
        
        loss += vel_coeff * velocity_loss(reconstructed, sample_data)
        return loss

    # WANDB: Watch the model to log gradients and topology
    if use_wandb:
        wandb.watch(model, log="all", log_freq=100)

    print(f"--- Starting Training from Epoch {start_epoch} ---")
    for epoch in range(start_epoch, hyperparameters["epochs"]):
        raw_batch = next(dl_iter)
        a = raw_batch["action"] # shape: (horizon, action_dim)
        sample_data = normalizer._normalize_action(a, inverse=False).to("cuda")
        
        transform = TRANSFORMS[random.randint(0, len(TRANSFORMS)-1)]
        sample_data = apply_affine_transform(
            sample_data,
            transform,
            random.random() * 0.15 + 0.05,
        )

        reconstructed_output, quantized_codes, add_loss = model(sample_data)
        loss = criterion(reconstructed_output, sample_data) + torch.mean(add_loss)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Add this
        optimizer.step()
        
        # SCHEDULER: Step the scheduler after each epoch
        scheduler.step()

        # WANDB: Log metrics to your dashboard
        if use_wandb:
            wandb.log({
                "loss": loss.item(),
                "learning_rate": scheduler.get_last_lr()[0]
            }, step=epoch)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{hyperparameters['epochs']}], Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    # CHECKPOINT: Save checkpoint
    print("--- Saving checkpoint ---")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'wandb_run_id': wandb.run.id if use_wandb else None
    }, checkpoint_path)


    print("\n--- Training Complete ---")
    with torch.no_grad():
        _, final_codes, _ = model(sample_data)
    print(f"Final Quantized int32 Codes:\n{final_codes}")
    
    # WANDB: Finish the run
    if use_wandb:
        wandb.finish()
