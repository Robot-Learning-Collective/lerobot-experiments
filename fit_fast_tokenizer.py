f"""
CLI utility for working with FAST action tokenizers.

Subcommands:
- plot: Load a tokenizer (hub id or local dir) and a dataset (hub id or local),
        normalize actions, tokenize them, and save a bar plot of token counts.
- fit:  Fit a new FAST tokenizer on normalized action windows from a dataset
        and save it to an output directory.

Examples:
    # Plot frequencies for a hub tokenizer and hub dataset
    python ./fit_fast_tokenizer.py plot \
        --tokenizer physical-intelligence/fast \
        --dataset lerobot/pusht
        --horizon 15

    # Fit a 256-vocab tokenizer
    python ./fit_fast_tokenizer.py fit \
        --dataset lerobot/pusht \
        --name fast_custom_256 \
        --vocab-size 256 --horizon 15 --scale 10
"""

import argparse
import os
from collections import Counter
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoProcessor

import json
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.processor.normalize_processor import NormalizerProcessorStep
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature


def _require_local_meta(dataset_root: str) -> dict:
    info_path = os.path.join(dataset_root, "meta", "info.json")
    if not os.path.isfile(info_path):
        raise FileNotFoundError(
            f"Local dataset not found or missing meta: {info_path}. Point --dataset-root to a local LeRobot dataset."
        )
    with open(info_path, "r") as f:
        return json.load(f)


def _build_dataset_local(repo_id: str, horizon: int) -> LeRobotDataset:
    meta = LeRobotDatasetMetadata(repo_id)
    fps = meta.fps

    delta_timestamps = {
        "observation.state": [0],
        "action": [t / fps for t in range(horizon)],
    }
    dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps, download_videos=False)
    return dataset


def _build_normalizer(dataset: LeRobotDataset) -> NormalizerProcessorStep:
    action_shape = tuple(dataset.meta.shapes["action"])  # (action_dim,)
    features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=action_shape)}
    norm_map = {FeatureType.ACTION: NormalizationMode.MIN_MAX}
    normalizer = NormalizerProcessorStep.from_lerobot_dataset(dataset, features, norm_map)
    return normalizer


def _iterate_normalized_actions(
    dataset: LeRobotDataset,
    normalizer: NormalizerProcessorStep,
) -> Iterable[np.ndarray]:
    n = len(dataset)
    from tqdm import tqdm
    for i in tqdm(range(n), desc="Iterating dataset for actions"):
        a = dataset[i]["action"].numpy()  # shape: (horizon, action_dim)
        a_norm = normalizer._normalize_action(torch.as_tensor(a, dtype=torch.float32), inverse=False).cpu().numpy()
        yield a_norm


def _count_tokens(
    processor,
    actions_iter: Iterable[np.ndarray],
) -> Tuple[Counter, int]:
    count = Counter()
    for a_norm in actions_iter:
        toks = processor(a_norm)
        for seq in toks:
            for t in seq:
                count[int(t)] += 1
    vocab_size = int(getattr(processor, "vocab_size", 0))
    return count, vocab_size


def _save_bar_plot(count: Counter, vocab_size: int, title: str | None = None) -> None:
    xs = np.arange(vocab_size) if vocab_size else np.array(sorted(count.keys(), key=int))
    ys = np.array([count.get(int(i), 0) for i in xs])

    plt.figure()
    plt.bar(xs, ys)
    if title:
        plt.title(title)
    plt.xlabel("token_id")
    plt.ylabel("frequency")

    output_name = f"token_freq_{title.replace('/', '-')}.png"
    plt.savefig(output_name, bbox_inches="tight")
    print(f"Plot saved in {os.path.abspath(output_name)}")


def cmd_plot(args: argparse.Namespace) -> None:
    processor = AutoProcessor.from_pretrained(args.tokenizer, trust_remote_code=True)
    
    dataset = _build_dataset_local(args.dataset_root, args.horizon)
    normalizer = _build_normalizer(dataset)

    actions_iter = _iterate_normalized_actions(dataset, normalizer)
    count, vocab_size = _count_tokens(processor, actions_iter)

    _save_bar_plot(count, vocab_size, str(args.tokenizer))


def cmd_fit(args: argparse.Namespace) -> None:
    # Load dataset + normalizer
    dataset = _build_dataset_local(args.dataset_root, args.horizon)
    normalizer = _build_normalizer(dataset)

    # Determine action_dim
    action_dim = int(dataset.meta.shapes["action"][0])

    # Collect normalized action windows (optionally subsample)
    actions = list(_iterate_normalized_actions(dataset, normalizer))
    if len(actions) == 0:
        raise RuntimeError("No actions found for fitting. Check dataset and horizon.")

    # Get the class implementing FAST; use a base processor id if provided
    base_proc_id = args.base_tokenizer
    base_proc = AutoProcessor.from_pretrained(base_proc_id, trust_remote_code=True)
    FastClass = type(base_proc)

    tokeniser = FastClass.fit(
        actions,
        scale=int(args.scale),
        vocab_size=int(args.vocab_size),
        time_horizon=int(args.horizon),
        action_dim=action_dim,
    )

    out_dir = f"outputs/{args.name}"
    tokeniser.save_pretrained(out_dir)
    print(f"Tokenizer saved to {out_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FAST tokenizer utilities (plot and fit)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # plot subcommand
    p_plot = sub.add_parser("plot", help="Plot token frequency bar chart for a tokenizer on a dataset")
    p_plot.add_argument("--tokenizer", required=True, help="Tokenizer hub id or local directory")
    p_plot.add_argument("--dataset-root", required=True, help="Local dataset root directory")
    p_plot.add_argument("--horizon", type=int, help="Temporal horizon for action windows")
    p_plot.set_defaults(func=cmd_plot)

    # fit subcommand
    p_fit = sub.add_parser("fit", help="Fit a FAST tokenizer on a dataset and save it")
    p_fit.add_argument("--dataset-root", required=True, help="Local dataset root directory")
    p_fit.add_argument("--name", required=True, help="Tokenizer name")
    p_fit.add_argument("--vocab-size", type=int, help="Vocabulary size")
    p_fit.add_argument("--horizon", type=int, help="Temporal horizon for action windows")
    p_fit.add_argument("--scale", type=int, default=10, help="Quantization scale used by FAST")
    p_fit.add_argument(
        "--base-tokenizer",
        default="physical-intelligence/fast",
        help="Base FAST processor to import class/logic from (hub id or local)",
    )
    p_fit.set_defaults(func=cmd_fit)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


