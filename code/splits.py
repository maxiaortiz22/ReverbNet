"""
Group-aware train/validation/test splits (BUG-04).

Records must never be split individually: the same room, source RIR or speaker would
then appear on both sides and the evaluation would measure memorization instead of
generalization. The groups are:

* **Real RIRs** → their **room** (``great_hall``, ``octagon``, ``classroom``). Every
  position of a room goes to the same split, and augmented variants follow their
  source RIR (they are generated from it inside its split).
* **Synthetic RIRs** (``sintetica_*``) → each file is its own group.
* **Speech** → the **speaker** (``F1``, ``M3``… from ``F1s3.wav``).
* **Validation** (inside the training set) → the **source RIR** of the recording.

All random choices use the project-wide seed through ``code/random_streams.py``, and
each item's assignment depends only on the seed and its own name.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .random_streams import item_rng

ROOMS = ('great_hall', 'octagon', 'classroom')
SYNTHETIC = 'synthetic'
_SYNTHETIC_TAG = 'sintetica'
_SPEAKER_RE = re.compile(r'^([FM]\d+)s\d+$')


# --------------------------------------------------------------------------- #
# File listings (always sorted: os.listdir order is not guaranteed, BUG-06)
# --------------------------------------------------------------------------- #

def list_rir_files(rir_dir: str | Path = 'data/RIRs') -> list[str]:
    """Sorted RIR file names in ``rir_dir``."""
    return sorted(f for f in os.listdir(rir_dir) if f.lower().endswith('.wav'))


def list_speech_files(speech_dir: str | Path = 'data/Speech') -> list[str]:
    """
    Sorted speech files from every subfolder of ``speech_dir``, as POSIX paths relative to
    it (e.g. ``'train/F1s3.wav'``). The original train/test folders are pooled so that the
    split can be made by speaker.
    """
    speech_dir = Path(speech_dir)
    return sorted(p.relative_to(speech_dir).as_posix() for p in speech_dir.glob('*/*.wav'))


# --------------------------------------------------------------------------- #
# Group keys
# --------------------------------------------------------------------------- #

def rir_group(rir_file: str) -> str:
    """
    Group of a RIR file: its room for real RIRs, ``'synthetic'`` for synthetic ones.

    Raises
    ------
    ValueError
        If the file belongs to no known room. New RIR datasets must register their rooms
        here so they can be split by room.
    """
    stem = Path(rir_file).stem
    if _SYNTHETIC_TAG in stem:
        return SYNTHETIC
    for room in ROOMS:
        if stem.startswith(room):
            return room
    raise ValueError(f"Cannot infer the room of RIR {rir_file!r}; register it in ROOMS")


def speaker_of(speech_file: str) -> str:
    """Speaker id of a speech file (``'F1'`` for ``'train/F1s3.wav'``)."""
    match = _SPEAKER_RE.match(Path(speech_file).stem)
    if match is None:
        raise ValueError(f"Cannot infer the speaker of {speech_file!r}")
    return match.group(1)


def recording_rir(recording_name: str) -> str:
    """Source RIR of a database record name ``'speech|rir|augmentation'``."""
    parts = recording_name.split('|')
    if len(parts) != 3:
        raise ValueError(f"Unexpected recording name {recording_name!r}")
    return parts[1]


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #

def split_rirs(rir_files: Iterable[str], test_rooms: Sequence[str],
               synthetic_test_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """
    Split RIR files into train and test.

    Real RIRs go to test if their room is in ``test_rooms``; each synthetic RIR goes to
    test with probability ``synthetic_test_fraction``.

    Returns
    -------
    (train, test) : tuple of sorted lists
    """
    unknown = set(test_rooms) - set(ROOMS)
    if unknown:
        raise ValueError(f"Unknown test rooms {sorted(unknown)}; known rooms: {ROOMS}")
    if set(test_rooms) == set(ROOMS):
        raise ValueError("At least one room must remain in the training set")
    if not 0.0 <= synthetic_test_fraction < 1.0:
        raise ValueError(f"`synthetic_test_fraction` must be in [0, 1), got {synthetic_test_fraction}")

    train, test = [], []
    for rir_file in sorted(rir_files):
        group = rir_group(rir_file)
        if group == SYNTHETIC:
            is_test = item_rng(seed, 'split', rir_file).random() < synthetic_test_fraction
        else:
            is_test = group in test_rooms
        (test if is_test else train).append(rir_file)
    return train, test


def split_speech(speech_files: Iterable[str],
                 test_speakers: Sequence[str]) -> tuple[list[str], list[str]]:
    """
    Split speech files by speaker.

    Returns
    -------
    (train, test) : tuple of sorted lists
    """
    speech_files = sorted(speech_files)
    speakers = {speaker_of(f) for f in speech_files}
    unknown = set(test_speakers) - speakers
    if unknown:
        raise ValueError(f"Unknown test speakers {sorted(unknown)}; available: {sorted(speakers)}")
    if set(test_speakers) == speakers:
        raise ValueError("At least one speaker must remain in the training set")

    train = [f for f in speech_files if speaker_of(f) not in test_speakers]
    test = [f for f in speech_files if speaker_of(f) in test_speakers]
    return train, test


def select_rirs_to_augment(rir_files: Iterable[str], per_room: int, seed: int) -> list[str]:
    """
    Pick ``per_room`` real RIRs of each room for RT/DRR augmentation.

    Each file gets a reproducible random key from the seed and its name; the
    ``per_room`` lowest keys of each room are selected independently, so dropping a room
    does not change the selection of the others. Only training RIRs are augmented
    (DATA-11): configs keep the training part of this selection and the generator never
    augments a test RIR.
    """
    by_room: dict[str, list[str]] = {}
    for rir_file in rir_files:
        group = rir_group(rir_file)
        if group != SYNTHETIC:
            by_room.setdefault(group, []).append(rir_file)

    selected = []
    for room in sorted(by_room):
        files = sorted(by_room[room])
        if per_room > len(files):
            raise ValueError(f"Room {room!r} has {len(files)} RIRs, cannot pick {per_room}")
        keys = [item_rng(seed, 'augment', f).random() for f in files]
        selected.extend(files[i] for i in np.argsort(keys, kind='stable')[:per_room])
    return sorted(selected)


def validation_mask(recording_names: Iterable[str], fraction: float, seed: int) -> np.ndarray:
    """
    Boolean mask selecting validation records inside a training set.

    Whole source RIRs (with all their speech files, augmentations and bands) are sent to
    validation with probability ``fraction``, so no RIR is on both sides. The decision for
    a RIR depends only on the seed and its name, hence it is identical for every band.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"`fraction` must be in (0, 1), got {fraction}")
    names = list(recording_names)
    decision = {}
    for rir in {recording_rir(n) for n in names}:
        decision[rir] = item_rng(seed, 'validation', rir).random() < fraction
    return np.array([decision[recording_rir(n)] for n in names], dtype=bool)
