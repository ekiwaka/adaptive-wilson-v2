from __future__ import annotations
import sys
from pathlib import Path
import torch

from gnn_graph_sampling.config.schema import DatasetConfig
from gnn_graph_sampling.data.loading import load_dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

PLANETOID_DATASETS = ["cora", "citeseer"]
OTHER_DATASETS = [
    "webkb/cornell",
    "webkb/texas",
    "webkb/wisconsin",
    "wikipedia/chameleon",
    "wikipedia/squirrel",
]


def print_bundle_stats(name: str, bundle, split_label: str) -> None:
    data = bundle.data
    print(
        f"{name:24} {split_label:10} "
        f"{int(data.num_nodes):8d} {int(data.edge_index.size(1)):10d} "
        f"{int(bundle.num_features):8d} {int(bundle.num_classes):8d} "
        f"{int(bundle.train_nodes.numel()):8d} {int(bundle.val_nodes.numel()):8d} {int(bundle.test_nodes.numel()):8d}"
    )


def main() -> None:
    print(
        f"{'dataset':24} {'split':10} {'nodes':>8} {'edges':>10} "
        f"{'feat':>8} {'classes':>8} {'train':>8} {'val':>8} {'test':>8}"
    )
    print("-" * 110)

    # Check Planetoid public split first
    for name in PLANETOID_DATASETS:
        try:
            cfg = DatasetConfig(
                name=name,
                root=str(ROOT / "data"),
                split_idx=0,
                planetoid_split="public",
                feature_mode="enabled",
                standardize_features=False,
            )
            bundle = load_dataset(cfg, device=torch.device("cpu"), seed=0)
            print_bundle_stats(name, bundle, "public")
        except Exception as exc:
            print(f"{name:24} {'public':10} ERROR: {exc}")

    # Check all Geom-GCN splits for Planetoid datasets
    for name in PLANETOID_DATASETS:
        for split_idx in range(10):
            try:
                cfg = DatasetConfig(
                    name=name,
                    root=str(ROOT / "data"),
                    split_idx=split_idx,
                    planetoid_split="geom-gcn",
                    feature_mode="enabled",
                    standardize_features=False,
                )
                bundle = load_dataset(cfg, device=torch.device("cpu"), seed=0)
                print_bundle_stats(name, bundle, f"geom-{split_idx}")
            except Exception as exc:
                print(f"{name:24} {f'geom-{split_idx}':10} ERROR: {exc}")

    # Check the non-Planetoid datasets with their default split handling
    for name in OTHER_DATASETS:
        try:
            cfg = DatasetConfig(
                name=name,
                root=str(ROOT / "data"),
                split_idx=0,
                feature_mode="enabled",
                standardize_features=False,
            )
            bundle = load_dataset(cfg, device=torch.device("cpu"), seed=0)
            print_bundle_stats(name, bundle, "split-0")
        except Exception as exc:
            print(f"{name:24} {'split-0':10} ERROR: {exc}")


if __name__ == "__main__":
    main()