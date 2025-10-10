import argparse
import json
import os
from typing import Iterable, List

import matplotlib.pyplot as plt

METRIC_NAME = "max_rewards"


def load_max_rewards(eval_json_path: str) -> List[float]:
    with open(eval_json_path, "r") as f:
        data = json.load(f)

    assert len(data.get("per_task", [])) == 1, "Expected exactly one task entry"
    # Expected structure: { "per_task": [ { "metrics": { "max_rewards": [...] } }, ... ] }
    task_entry = data["per_task"][0]
    metrics = task_entry.get("metrics", {})
    rewards = metrics.get(METRIC_NAME, [])
    return rewards


def plot_histogram_excluding_ones(values: Iterable[float], output_path: str, bins: int = 30) -> None:
    values_list = list(values)
    ones = [x for x in values_list if x == 1.0]
    non_ones = [x for x in values_list if x != 1.0]
    count_ones = len(ones)

    plt.hist(non_ones, bins=bins)

    plt.title("Max Rewards Histogram (excluding 1.0) with separate 1.0 count")
    plt.xlabel("max_reward")
    plt.ylabel("frequency")

    # Add a green bar to the right of 1.0 to show the count of 1.0 values
    bar_width = 1 / bins
    if count_ones > 0:
        plt.bar(1., count_ones, width=1 / bins, color="green", align="edge")

    plt.xlim(0, 1.1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot histogram of max_rewards from an eval JSON file.")
    parser.add_argument("eval_json", help="Path to eval_info.json containing max_rewards")
    parser.add_argument(
        "--output",
        help="Output image path. Defaults to <eval_dir>/histogram_max_rewards.png",
        default=None,
    )
    args = parser.parse_args()

    eval_json_path = args.eval_json
    if args.output is None:
        base_dir = os.path.dirname(os.path.abspath(eval_json_path))
        output_path = os.path.join(base_dir, "histogram_max_rewards.png")
    else:
        output_path = args.output

    max_rewards = load_max_rewards(eval_json_path)
    plot_histogram_excluding_ones(max_rewards, output_path)
    print(f"Plot saved in {output_path}")


if __name__ == "__main__":
    main()