# %%
from pathlib import Path
from tqdm import tqdm
import torch
from lerobot.datasets.utils import cycle

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.smolandfast.configuration_smolandfast import SMOLANDFASTConfig
from lerobot.policies.smolandfast.modeling_smolandfast import SMOLANDFASTPolicy

from lerobot.policies.factory import make_pre_post_processors

# %%
output_directory = Path("outputs/train/example_pusht")
output_directory.mkdir(parents=True, exist_ok=True)

device = torch.device("mps")

# %%
DATASET_PATH = "lerobot/pusht"

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
    batch_size=2,
    shuffle=True,
    pin_memory=device.type != "cpu",
    drop_last=True,
)
dl_iter = cycle(dataloader)



# %%
policy = SMOLANDFASTPolicy(cfg,
                           dataset_stats=dataset_metadata.stats)
policy.train()
policy.to(device)

preprocessor, postprocessor = make_pre_post_processors(policy.config, dataset_stats=dataset.meta.stats)

optimizer = torch.optim.Adam(policy.parameters(), lr=5e-5)

# %%
raw_batch = next(dl_iter)
batch = preprocessor(raw_batch)

# %%
for step in tqdm(range(50)):

    # batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    loss, _ = policy.forward(batch)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"step: {step} loss: {loss.item():.3f}")

# %% [markdown]
# 

# %%
decoded_actions = policy.model.generate_actions(batch)
decoded_actions = postprocessor(decoded_actions)
error:torch.tensor = torch.sqrt((decoded_actions - batch["action"])**2)

print(f"RMSE {(error.mean(dim=1)*100).tolist()}%")


