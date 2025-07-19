"""
Utility helpers for experiment configuration management.

Currently exposes a single function, :func:`import_configs_objs`, which loads a
Python configuration file *dynamically* (given its path), executes it in an
isolated module namespace, and returns a dictionary of its top‑level variables
with non‑configuration metadata keys removed.
"""

from importlib.machinery import SourceFileLoader
from types import ModuleType


def import_configs_objs(config_file):
    """
    Dynamically import a Python configuration file and return its namespace.

    The target file is executed in an ephemeral module context created via
    :class:`importlib.machinery.SourceFileLoader`. After execution, common
    module metadata entries (``__name__``, ``__doc__``, etc.) are stripped, and
    the remaining items are returned as a plain dictionary suitable for use as
    experiment configuration parameters.

    Parameters
    ----------
    config_file : str
        Filesystem path to the Python configuration file to load.

    Returns
    -------
    dict
        Dictionary mapping variable names to their values as defined in the
        configuration file.

    Raises
    ------
    ValueError
        If ``config_file`` is ``None``.
    """
    if config_file is None:
        raise ValueError("No config path")

    # Execute the file at ``config_file``.
    loader = SourceFileLoader('config', config_file)
    mod = ModuleType(loader.name)
    loader.exec_module(mod)

    # Convert module namespace to a dict and remove non‑configuration metadata.
    config_objs = vars(mod)
    config_objs.pop("__name__")
    config_objs.pop("__doc__")
    config_objs.pop("__package__")
    config_objs.pop("__loader__")
    config_objs.pop("__spec__")
    config_objs.pop("__builtins__")

    return config_objs