from __future__ import annotations
import argparse
from pathlib import Path
from gnn_graph_sampling.experiments.reporting import generate_sampler_report


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate sampler-comparison tables and plots from benchmark outputs.')
    parser.add_argument('--benchmark-root', action='append', default=[], type=Path, help='Benchmark output root; may be passed multiple times.')
    parser.add_argument('--q-sweep-root', action='append', default=[], type=Path, help='Q-sweep output root; may be passed multiple times.')
    parser.add_argument('--out-root', type=Path, required=True, help='Directory for the report tables and plots.')
    args = parser.parse_args()
    generate_sampler_report(
        benchmark_roots=args.benchmark_root,
        q_sweep_roots=args.q_sweep_root,
        out_root=args.out_root,
    )


if __name__ == '__main__':
    main()
