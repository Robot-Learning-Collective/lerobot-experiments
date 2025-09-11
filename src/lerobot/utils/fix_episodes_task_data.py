#!/usr/bin/env python

# Licensed under the Apache License, Version 2.0

"""Print episode -> task mapping from a LeRobot dataset cached locally.

Example:
python -m lerobot.utils.fix_episodes_task_data --repo-id {$USER}/dataset_name

This will read `meta/episodes.jsonl` and `meta/tasks.jsonl` and print, for each
episode, its `episode_index`, the first task string (`tasks[0]`), and the
corresponding `task_index` as defined in `tasks.jsonl`.
"""

import argparse
import logging
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


def _update_task_index_for_episode(meta: LeRobotDatasetMetadata, episode_index: int, task_index: int) -> None:
    root = meta.root
    parquet_path = root / meta.get_data_file_path(episode_index)
    if not parquet_path.is_file():
        logging.warning(f"Parquet file not found for episode {episode_index}: {parquet_path}")
        return

    table = pq.read_table(parquet_path)
    num_rows = table.num_rows

    col_name = "task_index"
    existing_idx = table.schema.get_field_index(col_name)

    new_values = pa.array(np.full(num_rows, task_index), type=(table.schema.field(col_name).type if existing_idx != -1 else pa.int64()))

    if existing_idx != -1:
        table = table.set_column(existing_idx, col_name, new_values)
    else:
        table = table.append_column(col_name, new_values)

    tmp_path = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
    pq.write_table(table, tmp_path)
    os.replace(tmp_path, parquet_path)


def apply_task_indices(repo_id: str) -> None:
    meta = LeRobotDatasetMetadata(repo_id=repo_id)
    for episode_index, episode in sorted(meta.episodes.items()):
        tasks = episode.get("tasks", []) or []
        if not tasks:
            logging.warning(f"Episode {episode_index} has no tasks; skipping")
            continue
        task = tasks[0]
        task_index = meta.task_to_task_index.get(task, None)
        if task_index is None:
            logging.warning(f"Task '{task}' not found in tasks.jsonl; skipping episode {episode_index}")
            continue
        _update_task_index_for_episode(meta, episode_index, int(task_index))
        logging.info(f"Updated episode {episode_index}: task='{task}', task_index={task_index}")


def push_to_hub(repo_id: str) -> None:
    meta = LeRobotDatasetMetadata(repo_id=repo_id)
    ds = LeRobotDataset(repo_id=repo_id, root=meta.root, force_cache_sync=False)
    ds.push_to_hub()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Print per-episode task and task_index from cached LeRobot dataset metadata."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face dataset repo id: user/name",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="If set, update task_index column in each episode parquet.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        default=False,
        help="Push edits to the Hub (will fail if remote already exists).",
    )
    args = parser.parse_args()

    if args.apply:
        apply_task_indices(repo_id=args.repo_id)
    if args.push_to_hub:
        push_to_hub(repo_id=args.repo_id)


if __name__ == "__main__":
    main()


