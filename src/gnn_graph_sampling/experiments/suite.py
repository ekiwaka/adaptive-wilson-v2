from __future__ import annotations
from ..config.loader import load_config
from ..config.schema import BenchmarkConfig, DiagnosticsConfig, SuiteConfig
from ..utils.logging import get_logger
from .benchmark import run_benchmark_study
from .common import parse_config_arg
from .diagnostics import run_diagnostics_study

LOGGER = get_logger(__name__)



def main() -> None:
    """Load suite configuration and execute diagnostics followed by benchmarking."""

    args = parse_config_arg('Run the configured publication study suite.')
    suite_config = load_config(SuiteConfig, args.config)
    diagnostics_config = load_config(DiagnosticsConfig, suite_config.diagnostics_config)
    benchmark_config = load_config(BenchmarkConfig, suite_config.benchmark_config)

    LOGGER.info('Running diagnostics suite from %s', suite_config.diagnostics_config)
    run_diagnostics_study(diagnostics_config)
    LOGGER.info('Running benchmark suite from %s', suite_config.benchmark_config)
    run_benchmark_study(benchmark_config)


if __name__ == '__main__':  # pragma: no cover
    main()
