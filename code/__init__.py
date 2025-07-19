"""
Top-level package exports for the acoustic database + model training pipeline.

This namespace re-exports the most commonly used functions, classes, and helpers
so they can be imported directly from ``code`` (e.g., ``from code import model``).
"""

# Re-export core modeling utilities.
from .model import (
    model,
    create_early_stopping_callback,
    reshape_data,
    normalize_descriptors,
    prediction,
    descriptors_err,
    save_exp_data,
)

# Re-export database generation and reading utilities.
from .generate_database import DataBaseGenerator
from .data_reader import read_dataset

# Re-export dynamic config loader.
from .utils import import_configs_objs

# Public API surface.
__all__ = [
    'model',
    'create_early_stopping_callback',
    'reshape_data',
    'normalize_descriptors',
    'prediction',
    'descriptors_err',
    'save_exp_data',
    'DataBaseGenerator',
    'read_dataset',
    'import_configs_objs',
]