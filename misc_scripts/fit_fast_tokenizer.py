f"""
CLI utility for working with FAST action tokenizers.

Subcommands:
- plot_tokens: Load a tokenizer (hub id or local dir) and a dataset (hub id or local),
        normalize actions, tokenize them, and save a bar plot of token counts.
- plot_coeffs: Load a dataset (hub id or local), normalize actions, compute DCT coefficients,
        and save a bar plot of coefficient counts.
- fit:  Fit a new FAST tokenizer on normalized action windows from a dataset
        and save it to an output directory.

Examples:
    # Plot tokens' frequencies for a tokenizer and adataset
    python ./misc_scripts/fit_fast_tokenizer.py plot_tokens \
        --tokenizer physical-intelligence/fast \
        --dataset lerobot/pusht
        --horizon 15

    # Plot coefficients for a dataset
    python ./misc_scripts/fit_fast_tokenizer.py plot_coeffs \
        --dataset lerobot/pusht
        --horizon 15
        --scale 10

    # Fit a 256-vocab tokenizer
    python ./misc_scripts/fit_fast_tokenizer.py fit \
        --dataset lerobot/pusht \
        --name fast_custom_256 \
        --vocab-size 256 --horizon 15 --scale 10
"""

import argparse
import os
from collections import Counter
from typing import Iterable, Tuple
from tqdm import tqdm
from scipy.fft import dct
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoProcessor

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.processor.normalize_processor import NormalizerProcessorStep
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature


BASE_TOKENIZER = "physical-intelligence/fast"


def _build_dataset_local(repo_id: str, horizon: int) -> LeRobotDataset:
    meta = LeRobotDatasetMetadata(repo_id)
    fps = meta.fps

    delta_timestamps = {
        "observation.state": [0], # Do we need observations?
        "action": [t / fps for t in range(horizon)],
    }
    dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps, download_videos=False)
    return dataset


def _build_min_max_normalizer(dataset: LeRobotDataset) -> NormalizerProcessorStep:
    action_shape = tuple(dataset.meta.shapes["action"])
    features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=action_shape)}
    norm_map = {FeatureType.ACTION: NormalizationMode.MIN_MAX}
    normalizer = NormalizerProcessorStep.from_lerobot_dataset(dataset, features, norm_map)
    return normalizer


def _iterate_normalized_actions(
    dataset: LeRobotDataset,
    normalizer: NormalizerProcessorStep,
) -> Iterable[np.ndarray]:
    n = len(dataset)
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
        # Count FAST tokens
        toks = processor(a_norm)
        for seq in toks:
            for t in seq:
                count[int(t)] += 1
    vocab_size = processor.vocab_size
    return count, vocab_size


def _count_coeffs(
    scale: int,
    actions_iter: Iterable[np.ndarray],
) -> Tuple[Counter, int]:
    count = Counter()

    # Ugly copy-paste from FAST tokenizer to count DCT coefficients
    dct_tokens = [dct(a_norm, axis=0, norm="ortho").flatten() for a_norm in actions_iter]

    # Quantize and find min token
    max_token = int(np.around(np.concatenate(dct_tokens) * scale).max())
    min_token = int(np.around(np.concatenate(dct_tokens) * scale).min())
    min_vocab_size = max_token - min_token
    print(f"{min_vocab_size=}")

    for tokens in dct_tokens:
        rounded_tokens = np.around(tokens * scale) - min_token
        rounded_tokens = rounded_tokens.astype(int)
        for token in rounded_tokens:
            count[token] += 1
    return count, min_vocab_size


def _save_bar_plot(count: Counter, vocab_size: int, title: str) -> None:
    xs = np.arange(vocab_size)
    ys = np.array([count.get(int(i), 0) for i in xs])

    plt.figure()
    plt.bar(xs, ys)
    plt.title(title)
    plt.xlabel("token_id")
    plt.ylabel("frequency")

    output_name = f"token_freq_{title.replace('/', '-')}.png"
    plt.savefig(output_name, bbox_inches="tight")
    print(f"Plot saved in {os.path.abspath(output_name)}")


def cmd_plot_tokens(args: argparse.Namespace) -> None:
    processor = AutoProcessor.from_pretrained(args.tokenizer, trust_remote_code=True)
    
    dataset = _build_dataset_local(args.dataset, args.horizon)
    normalizer = _build_min_max_normalizer(dataset)

    actions_iter = _iterate_normalized_actions(dataset, normalizer)
    count, vocab_size = _count_tokens(processor, actions_iter)

    _save_bar_plot(count, vocab_size, args.tokenizer)


def cmd_plot_coeffs(args: argparse.Namespace) -> None:
    
    dataset = _build_dataset_local(args.dataset, args.horizon)
    normalizer = _build_min_max_normalizer(dataset)

    actions_iter = _iterate_normalized_actions(dataset, normalizer)
    count, vocab_size = _count_coeffs(args.scale, actions_iter)

    _save_bar_plot(count, vocab_size, f"scale_{args.scale}")


def cmd_fit(args: argparse.Namespace) -> None:
    # Load dataset + normalizer
    dataset = _build_dataset_local(args.dataset, args.horizon)
    normalizer = _build_min_max_normalizer(dataset)

    # Determine action_dim
    action_dim = int(dataset.meta.shapes["action"][0])

    # Collect normalized action windows (optionally subsample)
    actions = list(_iterate_normalized_actions(dataset, normalizer))
    if len(actions) == 0:
        raise RuntimeError("No actions found for fitting. Check dataset and horizon.")

    # Get the class implementing FAST
    base_proc = AutoProcessor.from_pretrained(BASE_TOKENIZER, trust_remote_code=True)
    FastClass = type(base_proc)

    tokeniser = FastClass.fit(
        actions,
        scale=args.scale,
        vocab_size=args.vocab_size,
        time_horizon=args.horizon,
        action_dim=action_dim,
    )

    out_dir = f"outputs/{args.name}"
    tokeniser.save_pretrained(out_dir)
    print(f"Tokenizer saved to {out_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FAST tokenizer utilities (plot and fit)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # plot tokens subcommand
    p_plot_t = sub.add_parser("plot_tokens", help="Plot token frequency bar chart for a tokenizer on a dataset")
    p_plot_t.add_argument("--tokenizer", required=True, help="Tokenizer hub id or local directory")
    p_plot_t.add_argument("--dataset", required=True, help="Local dataset root directory")
    p_plot_t.add_argument("--horizon", type=int, help="Temporal horizon for action windows")
    p_plot_t.set_defaults(func=cmd_plot_tokens)

    # plot coeffs subcommand
    p_plot_c = sub.add_parser("plot_coeffs", help="Plot coeffs frequency bar chart for a dataset")
    p_plot_c.add_argument("--dataset", required=True, help="Local dataset root directory")
    p_plot_c.add_argument("--horizon", type=int, help="Temporal horizon for action windows")
    p_plot_c.add_argument("--scale", type=int, default=10, help="Quantization scale")
    p_plot_c.set_defaults(func=cmd_plot_coeffs)

    # fit subcommand
    p_fit = sub.add_parser("fit", help="Fit a FAST tokenizer on a dataset and save it")
    p_fit.add_argument("--dataset", required=True, help="Local dataset root directory")
    p_fit.add_argument("--name", required=True, help="Tokenizer name")
    p_fit.add_argument("--vocab-size", type=int, help="Vocabulary size")
    p_fit.add_argument("--horizon", type=int, help="Temporal horizon for action windows")
    p_fit.add_argument("--scale", type=int, default=10, help="Quantization scale used by FAST")
    p_fit.set_defaults(func=cmd_fit)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


