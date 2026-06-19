from __future__ import annotations
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints
import types
import yaml

from ..exceptions import ConfigurationError

T = TypeVar('T')


def _strip_optional(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin in {Union, types.UnionType}:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _convert_value(annotation: Any, value: Any) -> Any:
    annotation = _strip_optional(annotation)
    origin = get_origin(annotation)

    if is_dataclass(annotation):
        if not isinstance(value, dict):
            raise ConfigurationError(f'Expected mapping for nested config {annotation.__name__}.')
        return _build_dataclass(annotation, value)

    if origin is list:
        (inner_type,) = get_args(annotation)
        if not isinstance(value, list):
            raise ConfigurationError('Expected a list in YAML configuration.')
        return [_convert_value(inner_type, item) for item in value]

    return value


def _build_dataclass(cls: type[T], payload: dict[str, Any]) -> T:
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ConfigurationError(
            f'Unknown configuration keys for {cls.__name__}: ' + ', '.join(unknown)
        )

    type_hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for field in fields(cls):
        if field.name in payload:
            annotation = type_hints.get(field.name, field.type)
            kwargs[field.name] = _convert_value(annotation, payload[field.name])
        elif field.default is not MISSING or field.default_factory is not MISSING:  # type: ignore[attr-defined]
            continue
        else:
            raise ConfigurationError(f'Missing required configuration field: {field.name}')
    return cls(**kwargs)


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and validate that its top level is a mapping."""

    with Path(path).open('r', encoding='utf-8') as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ConfigurationError(f'Configuration file {path} must contain a top-level mapping.')
    return dict(payload)


def load_config(config_type: type[T], path: str | Path) -> T:
    """Load a YAML file into a typed configuration dataclass."""

    payload = load_yaml_mapping(path)
    instance = _build_dataclass(config_type, payload)
    validate = getattr(instance, 'validate', None)
    if callable(validate):
        instance = validate()
    return instance
