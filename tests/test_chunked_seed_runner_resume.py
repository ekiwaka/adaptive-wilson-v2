from __future__ import annotations

from pathlib import Path

import pytest

from gnn_graph_sampling.experiments.chunked_seed_runner import (
    chunk_ranges,
    first_unfinished_chunk_key,
    make_chunk_config,
)


def test_chunk_ranges_and_resume_from_interrupted_chunk() -> None:
    chunks = chunk_ranges(start_seed=0, total_runs=100, chunk_size=25)
    assert chunks == [
        (0, 24, 25),
        (25, 49, 25),
        (50, 74, 25),
        (75, 99, 25),
    ]

    status = {
        "chunks": {
            "0000_0024": {"status": "done"},
            "0025_0049": {"status": "done"},
            # This is the state left behind if a session dies mid-chunk.
            "0050_0074": {"status": "running"},
        }
    }

    assert first_unfinished_chunk_key(chunks, status) == "0050_0074"


def test_done_chunks_are_all_finished() -> None:
    chunks = chunk_ranges(start_seed=10, total_runs=4, chunk_size=2)
    status = {
        "chunks": {
            "0010_0011": {"status": "done"},
            "0012_0013": {"status": "done"},
        }
    }

    assert first_unfinished_chunk_key(chunks, status) is None


def test_make_chunk_config_forces_cpu_and_disables_plots_by_default(tmp_path: Path) -> None:
    base_config = {
        "study_name": "roman_empire_smoke",
        "runtime": {"device": "cuda", "torch_num_threads": 8},
        "evaluation": {"device": "cuda"},
        "output": {"root": "outputs/old", "experiment_name": "old"},
        "benchmark": {"runs": 100, "base_seed": 0, "parallel_workers": 8, "save_plots": True},
    }

    cfg = make_chunk_config(
        base_config,
        seed_start=50,
        seed_end=74,
        run_count=25,
        output_root=tmp_path / "chunked",
        experiment_prefix="roman_empire_unweighted",
        parallel_workers=4,
        force_cpu=True,
        save_plots=False,
    )

    assert cfg["runtime"]["device"] == "cpu"
    assert cfg["runtime"]["torch_num_threads"] == 1
    assert cfg["evaluation"]["device"] == "cpu"
    assert cfg["benchmark"]["runs"] == 25
    assert cfg["benchmark"]["base_seed"] == 50
    assert cfg["benchmark"]["parallel_workers"] == 4
    assert cfg["benchmark"]["save_plots"] is False
    assert cfg["output"]["experiment_name"] == "roman_empire_unweighted_seed0050_0074"


def test_invalid_chunk_arguments_raise() -> None:
    with pytest.raises(ValueError):
        chunk_ranges(start_seed=0, total_runs=0, chunk_size=1)
    with pytest.raises(ValueError):
        chunk_ranges(start_seed=0, total_runs=1, chunk_size=0)
