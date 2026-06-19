from __future__ import annotations
from dataclasses import dataclass
from typing import List
import torch

from ..exceptions import ConfigurationError


@dataclass(slots=True)
class DeviceResolution:
    """Resolved device information.

    Attributes:
        requested: Device string requested in configuration.
        resolved: Actual device string used by the runtime.
        warnings: Human-readable warnings emitted during resolution.
    """

    requested: str
    resolved: str
    warnings: List[str]


def resolve_device(requested: str, policy: str) -> DeviceResolution:
    """Resolve a configured device string into a safe runtime device.

    Args:
        requested: One of cpu, cuda, gpu, mps, or auto.
        policy: Either warn_and_fallback or error.

    Returns:
        A DeviceResolution describing the effective device.

    Raises:
        ConfigurationError: If the request is invalid or the policy is error
            and the requested accelerator is unavailable.
    """

    request = requested.lower().strip()
    policy = policy.lower().strip()
    warnings: List[str] = []

    if request == 'gpu':
        request = 'cuda'

    if policy not in {'warn_and_fallback', 'error'}:
        raise ConfigurationError(f'Unsupported device policy: {policy}')

    if request == 'auto':
        if torch.cuda.is_available():
            return DeviceResolution(requested=requested, resolved='cuda', warnings=[])
        if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
            return DeviceResolution(requested=requested, resolved='mps', warnings=[])
        return DeviceResolution(requested=requested, resolved='cpu', warnings=[])

    if request == 'cpu':
        return DeviceResolution(requested=requested, resolved='cpu', warnings=[])

    if request == 'cuda':
        if torch.cuda.is_available():
            return DeviceResolution(requested=requested, resolved='cuda', warnings=[])
        message = 'CUDA was requested but is not available.'
        if policy == 'error':
            raise ConfigurationError(message)
        warnings.append(message + ' Falling back to CPU.')
        return DeviceResolution(requested=requested, resolved='cpu', warnings=warnings)

    if request == 'mps':
        mps_backend = getattr(torch.backends, 'mps', None)
        if mps_backend is not None and mps_backend.is_available():
            return DeviceResolution(requested=requested, resolved='mps', warnings=[])
        message = 'MPS was requested but is not available.'
        if policy == 'error':
            raise ConfigurationError(message)
        warnings.append(message + ' Falling back to CPU.')
        return DeviceResolution(requested=requested, resolved='cpu', warnings=warnings)

    raise ConfigurationError(
        f'Unsupported device request: {requested}. Use one of cpu, cuda, gpu, mps, auto.'
    )
