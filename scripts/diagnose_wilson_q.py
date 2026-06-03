from __future__ import annotations
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple
import matplotlib.pyplot as plt
import torch
from scipy.sparse.csgraph import shortest_path
from torch_geometric.utils import subgraph, to_scipy_sparse_matrix

from gnn_graph_sampling.gnn.pipeline import load_dataset_bundle, make_torch_generator, set_global_seed
from gnn_graph_sampling.sampling.wilson import WilsonSampler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wilson structural diagnostics across q values.")
    parser.add_argument("--dataset", choices=["flickr", "actor", "citeseer", "cora"], default="flickr")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--split-idx", type=int, default=0)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="train")

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-batches", type=int, default=64)
    parser.add_argument("--base-seed", type=int, default=0)

    parser.add_argument("--feature-standardize", action="store_true")

    parser.add_argument("--wilson-q-values", type=float, nargs="+", required=True)
    parser.add_argument("--max-sampling-attempts", type=int, default=8)
    parser.add_argument("--redraw-q-growth", type=float, default=1.2)

    parser.add_argument("--distance-source-cap", type=int, default=128)
    parser.add_argument("--pair-sample-size", type=int, default=256)
    parser.add_argument("--global-distance-pair-samples", type=int, default=2000)

    parser.add_argument("--results-dir", type=str, default="results/wilson_diagnostics")
    return parser


def resolve_candidate_nodes(bundle, split_name: str) -> torch.Tensor:
    if split_name == "train":
        return bundle.train_nodes
    if split_name == "val":
        return bundle.val_nodes
    if split_name == "test":
        return bundle.test_nodes
    if split_name == "all":
        return torch.arange(bundle.data.num_nodes, device=bundle.data.x.device)
    raise ValueError(f"Unsupported split: {split_name}")


def build_adjacency_sets(edge_index: torch.Tensor, num_nodes: int) -> List[set[int]]:
    adjacency: List[set[int]] = [set([i]) for i in range(num_nodes)]
    src = edge_index[0].detach().cpu().tolist()
    dst = edge_index[1].detach().cpu().tolist()
    for u, v in zip(src, dst):
        adjacency[u].add(v)
        adjacency[v].add(u)
    return adjacency


def compute_degrees(edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
    deg = torch.bincount(edge_index[0], minlength=num_nodes).float()
    return deg.to(device)


def sample_pair_indices(n: int, max_pairs: int) -> List[Tuple[int, int]]:
    if n < 2:
        return []
    all_pairs: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs.append((i, j))
    if len(all_pairs) <= max_pairs:
        return all_pairs

    step = max(1, len(all_pairs) // max_pairs)
    return all_pairs[::step][:max_pairs]


def jaccard(a: set[int], b: set[int]) -> float:
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


def mean_neighborhood_overlap(center_nodes: torch.Tensor, adjacency_sets: List[set[int]], max_pairs: int) -> float:
    centers = center_nodes.detach().cpu().tolist()
    pairs = sample_pair_indices(len(centers), max_pairs)
    if not pairs:
        return 0.0
    vals = []
    for i, j in pairs:
        vals.append(jaccard(adjacency_sets[centers[i]], adjacency_sets[centers[j]]))
    return float(sum(vals) / max(len(vals), 1))


def mean_pairwise_distance_for_centers(
    center_nodes: torch.Tensor,
    graph_csr,
) -> float:
    if center_nodes.numel() < 2:
        return 0.0

    centers_cpu = center_nodes.detach().cpu().numpy()
    dists = shortest_path(graph_csr, directed=False, unweighted=True, indices=centers_cpu)
    center_to_center = dists[:, centers_cpu]

    vals: List[float] = []
    n = center_to_center.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            d = float(center_to_center[i, j])
            if math.isfinite(d) and d > 0:
                vals.append(d)

    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def estimate_global_mean_pair_distance(
    num_nodes: int,
    graph_csr,
    num_pairs: int,
    generator: torch.Generator,
) -> float:
    if num_nodes < 2:
        return 1.0

    idx = torch.randint(0, num_nodes, (num_pairs, 2), generator=generator)
    vals = []
    for pair in idx.tolist():
        u, v = pair
        if u == v:
            continue
        d = shortest_path(graph_csr, directed=False, unweighted=True, indices=[u])[0, v]
        if math.isfinite(d) and d > 0:
            vals.append(float(d))

    if not vals:
        return 1.0
    return float(sum(vals) / len(vals))


def build_local_wilson(candidate_nodes: torch.Tensor, full_edge_index: torch.Tensor, num_nodes: int, device: torch.device):
    candidate_nodes = torch.unique(candidate_nodes, sorted=True)
    local_edge_index, _ = subgraph(
        subset=candidate_nodes,
        edge_index=full_edge_index,
        relabel_nodes=True,
        num_nodes=num_nodes,
    )
    local_wilson = WilsonSampler(
        edge_index=local_edge_index,
        num_nodes=int(candidate_nodes.numel()),
        make_undirected=False,
        device=device,
    )
    return candidate_nodes, local_wilson


def sample_wilson_batch(
    candidate_nodes_global: torch.Tensor,
    local_wilson: WilsonSampler,
    q: float,
    budget: int,
    *,
    generator: torch.Generator,
    max_sampling_attempts: int,
    redraw_q_growth: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    candidate_size = int(candidate_nodes_global.numel())
    budget = min(int(budget), candidate_size)

    selected_local_mask = torch.zeros(candidate_size, device=candidate_nodes_global.device, dtype=torch.bool)
    raw_root_counts: List[int] = []
    redraws_used = 0
    fallback_uniform = 0.0
    current_q = float(q)

    for _ in range(max_sampling_attempts):
        redraws_used += 1
        _, roots_mask, _, _ = local_wilson.sample_forest_with_q_factor(
            q=current_q,
            generator=generator,
            trace=False,
        )
        roots_local = torch.nonzero(roots_mask, as_tuple=False).view(-1)
        raw_root_counts.append(int(roots_local.numel()))

        if roots_local.numel() > 0:
            remaining_local = roots_local[~selected_local_mask[roots_local]]
            if remaining_local.numel() > 0:
                perm = torch.randperm(
                    remaining_local.numel(),
                    generator=generator,
                    device=candidate_nodes_global.device,
                )
                remaining_local = remaining_local[perm]
                need = budget - int(selected_local_mask.sum().item())
                take = remaining_local[:need]
                selected_local_mask[take] = True

        if int(selected_local_mask.sum().item()) >= budget:
            break

        current_q *= redraw_q_growth

    chosen_local = torch.nonzero(selected_local_mask, as_tuple=False).view(-1)

    if chosen_local.numel() < budget:
        remaining_local = torch.nonzero(~selected_local_mask, as_tuple=False).view(-1)
        if remaining_local.numel() > 0:
            perm = torch.randperm(
                remaining_local.numel(),
                generator=generator,
                device=candidate_nodes_global.device,
            )
            need = budget - chosen_local.numel()
            fill_local = remaining_local[perm[:need]]
            chosen_local = torch.cat([chosen_local, fill_local], dim=0)
            fallback_uniform = 1.0

    center_nodes = candidate_nodes_global[chosen_local]

    raw_mean = float(mean(raw_root_counts)) if raw_root_counts else 0.0
    raw_std = float(pstdev(raw_root_counts)) if len(raw_root_counts) > 1 else 0.0

    meta = {
        "raw_mean_roots_before_cap": raw_mean,
        "raw_std_roots_before_cap": raw_std,
        "redraws_used": float(redraws_used),
        "fallback_uniform": float(fallback_uniform),
        "realized_budget": float(center_nodes.numel()),
    }
    return center_nodes, meta


def aggregate_metric(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def plot_metrics(rows: List[Dict[str, float]], output_path: Path) -> None:
    qs = [row["q"] for row in rows]

    metrics = [
        ("raw_mean_roots_before_cap", "Raw mean roots"),
        ("raw_std_roots_before_cap", "Raw root std"),
        ("mean_redraws_per_batch", "Mean redraws"),
        ("fallback_fill_rate", "Fallback fill rate"),
        ("mean_center_degree", "Mean center degree"),
        ("mean_pairwise_center_distance", "Mean pairwise distance"),
        ("center_class_coverage_ratio", "Class coverage ratio"),
        ("center_degree_ratio_vs_global", "Degree ratio vs global"),
        ("mean_neighborhood_overlap_jaccard", "Neighborhood overlap"),
        ("root_diversity_unique_ratio", "Root diversity"),
        ("center_spread_ratio", "Spread ratio"),
    ]

    ncols = 2
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()

    for ax, (key, title) in zip(axes, metrics):
        ys = [row[key] for row in rows]
        ax.plot(qs, ys, marker="o")
        ax.set_title(title)
        ax.set_xlabel("q")
        ax.set_ylabel(key)

    for ax in axes[len(metrics):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    args = build_parser().parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    set_global_seed(args.base_seed)

    bundle = load_dataset_bundle(
        name=args.dataset,
        root=args.data_root,
        device=device,
        split_idx=args.split_idx,
        feature_standardize=args.feature_standardize,
    )
    data = bundle.data
    candidate_nodes = resolve_candidate_nodes(bundle, args.split)

    candidate_nodes_global, local_wilson = build_local_wilson(
        candidate_nodes=candidate_nodes,
        full_edge_index=data.edge_index,
        num_nodes=int(data.num_nodes),
        device=device,
    )

    deg = compute_degrees(data.edge_index, int(data.num_nodes), device=device)
    global_mean_degree = float(deg.mean().item())

    adjacency_sets = build_adjacency_sets(data.edge_index, int(data.num_nodes))
    graph_csr = to_scipy_sparse_matrix(data.edge_index.cpu(), num_nodes=int(data.num_nodes)).tocsr()

    distance_generator = make_torch_generator(device=torch.device("cpu"), seed=args.base_seed + 12345)
    global_mean_pair_distance = estimate_global_mean_pair_distance(
        num_nodes=int(data.num_nodes),
        graph_csr=graph_csr,
        num_pairs=args.global_distance_pair_samples,
        generator=distance_generator,
    )

    rows: List[Dict[str, float]] = []

    for q in args.wilson_q_values:
        generator = make_torch_generator(device=device, seed=args.base_seed + int(q * 1000))

        raw_mean_roots_before_cap_vals: List[float] = []
        raw_std_roots_before_cap_vals: List[float] = []
        redraws_vals: List[float] = []
        fallback_vals: List[float] = []

        center_degree_vals: List[float] = []
        pairwise_distance_vals: List[float] = []
        class_coverage_vals: List[float] = []
        class_coverage_ratio_vals: List[float] = []
        center_degree_ratio_vals: List[float] = []
        overlap_vals: List[float] = []

        all_centers_seen: List[int] = []
        total_centers_sampled = 0

        for _ in range(args.num_batches):
            center_nodes, meta = sample_wilson_batch(
                candidate_nodes_global=candidate_nodes_global,
                local_wilson=local_wilson,
                q=float(q),
                budget=args.batch_size,
                generator=generator,
                max_sampling_attempts=args.max_sampling_attempts,
                redraw_q_growth=args.redraw_q_growth,
            )

            raw_mean_roots_before_cap_vals.append(meta["raw_mean_roots_before_cap"])
            raw_std_roots_before_cap_vals.append(meta["raw_std_roots_before_cap"])
            redraws_vals.append(meta["redraws_used"])
            fallback_vals.append(meta["fallback_uniform"])

            center_deg = float(deg[center_nodes].mean().item())
            center_degree_vals.append(center_deg)
            center_degree_ratio_vals.append(center_deg / max(global_mean_degree, 1e-8))

            pairwise_d = mean_pairwise_distance_for_centers(center_nodes=center_nodes, graph_csr=graph_csr)
            pairwise_distance_vals.append(pairwise_d)

            labels = data.y[center_nodes]
            class_coverage = float(torch.unique(labels).numel())
            class_coverage_vals.append(class_coverage)
            class_coverage_ratio_vals.append(class_coverage / float(bundle.num_classes))

            overlap_vals.append(
                mean_neighborhood_overlap(
                    center_nodes=center_nodes,
                    adjacency_sets=adjacency_sets,
                    max_pairs=args.pair_sample_size,
                )
            )

            centers_cpu = center_nodes.detach().cpu().tolist()
            all_centers_seen.extend(centers_cpu)
            total_centers_sampled += len(centers_cpu)

        unique_centers = len(set(all_centers_seen))
        root_diversity_unique_ratio = float(unique_centers / max(total_centers_sampled, 1))
        root_repeat_rate = float(1.0 - root_diversity_unique_ratio)

        mean_pairwise_center_distance = aggregate_metric(pairwise_distance_vals)
        center_spread_ratio = mean_pairwise_center_distance / max(global_mean_pair_distance, 1e-8)

        row = {
            "q": float(q),
            "num_batches": float(args.num_batches),
            "budget": float(args.batch_size),

            "raw_mean_roots_before_cap": aggregate_metric(raw_mean_roots_before_cap_vals),
            "raw_std_roots_before_cap": aggregate_metric(raw_std_roots_before_cap_vals),
            "mean_redraws_per_batch": aggregate_metric(redraws_vals),
            "fallback_fill_rate": aggregate_metric(fallback_vals),

            "mean_center_degree": aggregate_metric(center_degree_vals),
            "mean_pairwise_center_distance": mean_pairwise_center_distance,
            "center_class_coverage": aggregate_metric(class_coverage_vals),
            "center_class_coverage_ratio": aggregate_metric(class_coverage_ratio_vals),

            "center_spread_ratio": center_spread_ratio,
            "center_degree_ratio_vs_global": aggregate_metric(center_degree_ratio_vals),
            "mean_neighborhood_overlap_jaccard": aggregate_metric(overlap_vals),

            "root_diversity_unique_ratio": root_diversity_unique_ratio,
            "root_repeat_rate": root_repeat_rate,

            "global_mean_degree": global_mean_degree,
            "global_mean_pair_distance": global_mean_pair_distance,
        }
        rows.append(row)

        print(
            f"q={q:.3f} "
            f"raw_mean_roots={row['raw_mean_roots_before_cap']:.2f} "
            f"redraws={row['mean_redraws_per_batch']:.2f} "
            f"fallback={row['fallback_fill_rate']:.3f} "
            f"center_deg={row['mean_center_degree']:.2f} "
            f"pair_dist={row['mean_pairwise_center_distance']:.2f} "
            f"class_cov={row['center_class_coverage_ratio']:.3f} "
            f"overlap={row['mean_neighborhood_overlap_jaccard']:.3f} "
            f"diversity={row['root_diversity_unique_ratio']:.3f}"
        )

    csv_path = results_dir / f"{args.dataset}_{args.split}_wilson_diagnostics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_path = results_dir / f"{args.dataset}_{args.split}_wilson_diagnostics.png"
    plot_metrics(rows, plot_path)

    print(f"\nSaved diagnostics CSV to {csv_path}")
    print(f"Saved diagnostics plot to {plot_path}")
