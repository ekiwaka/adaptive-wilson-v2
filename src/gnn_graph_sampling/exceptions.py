from __future__ import annotations
"""Project-specific exception hierarchy."""


class GnnGraphSamplingError(RuntimeError):
    """Base exception for project-specific runtime errors."""


class ConfigurationError(GnnGraphSamplingError):
    """Raised when a YAML configuration is invalid or inconsistent."""


class NativeBackendUnavailableError(GnnGraphSamplingError):
    """Raised when the native Wilson backend is requested but unavailable."""


class PyGUnavailableError(GnnGraphSamplingError):
    """Raised when torch_geometric is required but could not be imported."""
