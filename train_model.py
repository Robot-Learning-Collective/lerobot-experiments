import torch
import torch.nn as nn
import torch.optim as optim
import os


from torch.optim.lr_scheduler import CosineAnnealingLR

from pathlib import Path
from tqdm import tqdm
import torch
from lerobot.datasets.utils import cycle

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.smolandfast.configuration_smolandfast import SMOLANDFASTConfig
from lerobot.policies.smolandfast.modeling_smolandfast import SMOLANDFASTPolicy
from lerobot.policies.smolandfast.tokenizer import Autoencoder

from lerobot.policies.factory import make_pre_post_processors
import wandb


if __name__ == '__main__':
    DATASET_PATH = "lerobot/pusht"
    output_directory = Path("outputs/train/example_pusht")
    output_directory.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")

    dataset_metadata = LeRobotDatasetMetadata(DATASET_PATH)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    cfg = SMOLANDFASTConfig(input_features=input_features,
                            output_features=output_features)

    delta_timestamps = {
            "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
        }

    # We can then instantiate the dataset with these delta_timestamps configuration.
    dataset = LeRobotDataset(DATASET_PATH, delta_timestamps=delta_timestamps)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=32,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )
    dl_iter = cycle(dataloader)
    
    policy = SMOLANDFASTPolicy(cfg,
                           dataset_stats=dataset_metadata.stats)
    policy.to(device)

    preprocessor, _ = make_pre_post_processors(policy.config, dataset_stats=dataset.meta.stats)

    del policy

    # Store hyperparameters in a dictionary for easy logging
    hyperparameters = {
        "learning_rate": 0.0004,
        "epochs": 20000,
        "encoded_dim": 3,
        "vocab_size": 2048,
        "base_features": 32,
        "ratios": [2, 2, 1],
        "num_residual_layers": 3,
        "num_lstm_layers": 3,
    }
    use_wandb = True
    checkpoint_path = "auto_encoder_new.pth"

    # --- CHECKPOINTING SETUP ---
    start_epoch = 0
    wandb_run_id = None

    # Model configuration
    model = Autoencoder(
        encoded_dim=hyperparameters["encoded_dim"],
        base_features=hyperparameters["base_features"],
        ratios=hyperparameters["ratios"],
        num_residual_layers=hyperparameters["num_residual_layers"],
        num_lstm_layers=hyperparameters["num_lstm_layers"],
        vocab_size=hyperparameters["vocab_size"],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])
    scheduler = CosineAnnealingLR(optimizer, T_max=hyperparameters['epochs'], eta_min=0.01 * hyperparameters['learning_rate'])

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
        l2_loss=[0, 4, 9, 11],
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
        batch = preprocessor(raw_batch)
        sample_data = batch["action"]

        reconstructed_output, quantized_codes, add_loss = model(sample_data)
        loss = criterion(reconstructed_output, sample_data) + torch.mean(add_loss)

        optimizer.zero_grad()
        loss.backward()
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