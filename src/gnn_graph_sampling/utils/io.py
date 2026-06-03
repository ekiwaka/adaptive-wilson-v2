from __future__ import annotations
import csv
import json
import re
import shutil
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
import yaml

from ..exceptions import ConfigurationError


def _resolve_unique_path(path: Path) -> Path:
    """Return a unique path by appending a numeric suffix when needed."""
    if not path.exists():
        return path

    index = 1
    while True:
        candidate = path.parent / f"{path.name}-{index:02d}"
        if not candidate.exists():
            return candidate
        index += 1


def assert_pristine_run_dir(path: str | Path) -> Path:
    """Raise if a run directory already contains persisted run artifacts."""

    run_dir = Path(path)
    tracked = [
        run_dir / 'resolved_config.yaml',
        run_dir / 'metrics.json',
        run_dir / 'history.csv',
        run_dir / 'checkpoint.pt',
        run_dir / 'logs' / 'run.log',
    ]
    existing = [artifact for artifact in tracked if artifact.exists()]
    if existing:
        raise ConfigurationError(
            'Refusing to reuse an existing run directory with persisted artifacts: '
            + ', '.join(str(item) for item in existing)
        )
    return run_dir


def create_experiment_root(
    output_root: str | Path,
    experiment_name: str,
    run_name: str,
    *,
    overwrite: bool = False,
    append_timestamp: bool = False,
    timestamp_format: str = "%Y-%m-%d-%H-%M",
) -> Path:
    """Create and return the root directory for one experiment or study."""

    base_name = slugify(run_name)
    if append_timestamp:
        stamp = datetime.now().strftime(timestamp_format)
        base_name = f"{base_name}-{stamp}"

    root = Path(output_root) / slugify(experiment_name) / base_name

    if append_timestamp:
        root = _resolve_unique_path(root)
    elif root.exists() and overwrite:
        shutil.rmtree(root)
    elif root.exists() and not overwrite:
        raise ConfigurationError(
            f'Output directory already exists: {root}. '
            'Choose a different run name, enable append_timestamp, or set overwrite=true.'
        )

    return ensure_dir(root)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""

    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def slugify(value: str) -> str:
    """Return a filesystem-friendly slug."""

    return re.sub(r'[^a-zA-Z0-9._-]+', '-', value.strip()).strip('-').lower()


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write a mapping to a UTF-8 JSON file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)


def write_yaml(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write a mapping to a UTF-8 YAML file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        yaml.safe_dump(dict(payload), handle, sort_keys=False)


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a collection of rows to CSV.

    Unlike a strict rectangular writer, this version first computes the union
    of all keys across all rows, so heterogeneous experiment outputs
    (for example sampler-specific diagnostics) can still be written safely.
    """

    if not rows:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def dataclass_to_dict(instance: Any) -> dict[str, Any]:
    """Convert a nested dataclass instance into a plain mapping."""

    if not is_dataclass(instance):
        raise TypeError('Expected a dataclass instance.')
    return asdict(instance)
