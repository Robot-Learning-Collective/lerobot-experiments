#!/usr/bin/env python

# Licensed under the Apache License, Version 2.0

"""Remove one or more episodes from a LeRobot dataset, reindex, and rebuild metadata.

```
cp -r ~/.cache/huggingface/lerobot/{$USER}/old_dataset_name ~/.cache/huggingface/lerobot/{$USER}/new_dataset_name
python -m lerobot.utils.remove_episodes_from_dataset \
    --repo-id {$USER}/new_dataset_name \
    --episodes 0 2  \
    --push-to-hub
```
1. Delete episodes data and videos
2. Reindex and move data and videos to keep contiguous indices
3. Rebuild metadata (episodes.jsonl, episodes_stats.jsonl, info.json)

Assumptions and scope:
- Data is stored per episode in separate parquet files, and videos are also per episode (one mp4 per episode per camera).
- This tool only handles videos. Image are not moved or rewritten.

"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi


from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    EPISODES_PATH,
    EPISODES_STATS_PATH,
    DEFAULT_FEATURES,
    load_episodes,
    load_episodes_stats,
    load_info,
    serialize_dict,
    write_info,
    write_jsonlines,
)


def _assert_in_cache(repo_id: str, root: Path) -> None:
    meta_dir = root / "meta"
    if not meta_dir.exists():
        raise FileNotFoundError(
            f"Dataset meta not found locally at {meta_dir}. Make sure the dataset exists or was downloaded."
        )


def _delete_episode_files(meta: LeRobotDatasetMetadata, root: Path, episode_index: int) -> None:
    # Delete parquet
    data_path = root / meta.get_data_file_path(episode_index)
    if data_path.is_file():
        data_path.unlink()
    else:
        logging.warning(f"Parquet file not found for episode {episode_index}: {data_path}")

    # Delete videos
    for vid_key in meta.video_keys:
        vid_path = root / meta.get_video_file_path(episode_index, vid_key)
        if vid_path.is_file():
            vid_path.unlink()
        else:
            logging.warning(f"Video file not found for episode {episode_index}: {vid_path}")


def _rewrite_episode_data(
    old_path: Path,
    new_path: Path,
    new_episode_index: int,
    new_global_start_index: int,
) -> int:
    """Read a parquet, update episode_index and index columns, write atomically to new_path.

    Returns the number of rows written (episode length).
    """
    table = pq.read_table(old_path)
    num_rows = table.num_rows

    # Canonical column names from DEFAULT_FEATURES
    assert "index" in DEFAULT_FEATURES and "episode_index" in DEFAULT_FEATURES
    INDEX_COL = "index"
    EPISODE_INDEX_COL = "episode_index"

    new_index_array = pa.array(
        np.arange(new_global_start_index, new_global_start_index + num_rows),
        type=table.schema.field(INDEX_COL).type,
    )
    new_episode_index_array = pa.array(
        np.full(num_rows, new_episode_index),
        type=table.schema.field(EPISODE_INDEX_COL).type,
    )

    # Replace columns (keep positions)
    idx_pos = table.schema.get_field_index(INDEX_COL)
    table = table.set_column(idx_pos, INDEX_COL, new_index_array)

    ep_pos = table.schema.get_field_index(EPISODE_INDEX_COL)
    table = table.set_column(ep_pos, EPISODE_INDEX_COL, new_episode_index_array)

    # Atomic write via temp file then replace
    new_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = new_path.with_suffix(new_path.suffix + ".tmp")
    pq.write_table(table, tmp_path)
    os.replace(tmp_path, new_path)

    # If destination differs from source, remove the old file
    if old_path.resolve() != new_path.resolve():
        old_path.unlink()

    return num_rows


def _move_episode_videos(meta: LeRobotDatasetMetadata, root: Path, old_ep: int, new_ep: int) -> None:
    # Videos
    for vid_key in meta.video_keys:
        old_vid = root / meta.get_video_file_path(old_ep, vid_key)
        new_vid = root / meta.get_video_file_path(new_ep, vid_key)
        if old_vid.exists():
            os.replace(old_vid, new_vid)
        else:
            logging.warning(f"Video file not found for episode {old_ep}: {old_vid}")


def _reindex_and_rewrite(
    meta: LeRobotDatasetMetadata,
    root: Path,
    episodes_to_delete: List[int],
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Delete requested episodes and rewrite remaining ones with contiguous indices.

    Returns:
        mapping_old_to_new: map from old episode index to new episode index
        new_lengths: map from new episode index to its length
    """
    all_eps = sorted(meta.episodes.keys())
    episodes_to_delete = sorted(episodes_to_delete)

    # Remove files for episodes to delete first (avoid path collisions)
    for ep in episodes_to_delete:
        _delete_episode_files(meta, root, ep)

    remaining = [ep for ep in all_eps if ep not in episodes_to_delete]
    mapping_old_to_new = {old: new for new, old in enumerate(remaining)}

    # Rewrite parquets and move media for remaining episodes
    new_lengths: Dict[int, int] = {}
    cumulative = 0
    for new_ep, old_ep in enumerate(remaining):
        old_parquet = root / meta.get_data_file_path(old_ep)
        new_parquet = root / meta.get_data_file_path(new_ep)

        if not old_parquet.is_file():
            raise FileNotFoundError(f"Missing parquet for episode {old_ep}: {old_parquet}")

        ep_len = _rewrite_episode_data(
            old_path=old_parquet,
            new_path=new_parquet,
            new_episode_index=new_ep,
            new_global_start_index=cumulative,
        )
        new_lengths[new_ep] = ep_len
        cumulative += ep_len

        _move_episode_videos(meta, root, old_ep=old_ep, new_ep=new_ep)

    return mapping_old_to_new, new_lengths


def _rebuild_metadata(
    meta: LeRobotDatasetMetadata,
    root: Path,
    mapping_old_to_new: Dict[int, int],
    new_lengths: Dict[int, int],
) -> None:
    # Rebuild episodes.jsonl
    old_episodes = load_episodes(root)
    episodes_items = []
    for old_ep, new_ep in sorted(mapping_old_to_new.items(), key=lambda x: x[1]):
        old_entry = old_episodes[old_ep]
        episodes_items.append(
            {
                "episode_index": new_ep,
                "tasks": old_entry.get("tasks", []),
                "length": int(new_lengths[new_ep]),
            }
        )
    write_jsonlines(episodes_items, root / EPISODES_PATH)

    # Rebuild episodes_stats.jsonl if present
    episodes_stats_items = []
    old_stats = load_episodes_stats(root)
    for old_ep, new_ep in sorted(mapping_old_to_new.items(), key=lambda x: x[1]):
        stats = old_stats.get(old_ep, None)
        episodes_stats_items.append({"episode_index": new_ep, "stats": serialize_dict(stats)})
    write_jsonlines(episodes_stats_items, root / EPISODES_STATS_PATH)


    # Update info.json
    info = load_info(root)
    total_episodes = len(new_lengths)
    total_frames = int(sum(int(v) for v in new_lengths.values()))
    chunks_size = info.get("chunks_size", meta.chunks_size)
    total_chunks = (total_episodes + chunks_size - 1) // chunks_size if total_episodes > 0 else 0
    total_videos = total_episodes * len(meta.video_keys)

    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_chunks"] = total_chunks
    info["total_videos"] = total_videos
    info["splits"] = {"train": f"0:{total_episodes}"}

    write_info(info, root)


def remove_episodes(repo_id: str, episodes: List[int], push_to_hub: bool) -> None:
    root = HF_LEROBOT_HOME / repo_id
    _assert_in_cache(repo_id, root)
    meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root)

    # Validate episodes
    invalid = [e for e in episodes if e not in meta.episodes]
    if invalid:
        raise ValueError(f"Unknown episode indices: {invalid}. Available range: 0..{meta.total_episodes - 1}")

    mapping_old_to_new, new_lengths = _reindex_and_rewrite(meta, root, episodes_to_delete=episodes)

    # Rebuild metadata
    _rebuild_metadata(meta, root, mapping_old_to_new, new_lengths)

    # Validate by loading
    ds = LeRobotDataset(repo_id=repo_id, root=root, force_cache_sync=False)
    assert ds.num_episodes == len(new_lengths)
    logging.info(f"Remaining episodes: {ds.num_episodes}")

    if push_to_hub:
        # Abort if remote repo exists already (no in-function imports)
        api = HfApi()
        if api.repo_exists(repo_id=repo_id, repo_type="dataset"):
            raise RuntimeError(f"Remote dataset '{repo_id}' already exists. Aborting push.")
        ds.push_to_hub()


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Remove one or more episodes from a LeRobot dataset.")
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face dataset repo id: user/name")
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        required=True,
        help="List of episode indices to remove (space-separated integers).",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        default=False,
        help="Push edits to the Hub (default: False)",
    )
    args = parser.parse_args()

    remove_episodes(repo_id=args.repo_id, episodes=args.episodes, push_to_hub=args.push_to_hub)


if __name__ == "__main__":
    main()


