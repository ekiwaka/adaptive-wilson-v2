from __future__ import annotations

import sys
from pathlib import Path

# Allow running this script directly from a source checkout without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from gnn_graph_sampling.experiments.chunked_seed_runner import main


if __name__ == '__main__':
    main()
