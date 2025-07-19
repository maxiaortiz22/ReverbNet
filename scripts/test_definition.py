"""
Batch test for the C++ ``DefinitionCalculator`` extension.

This script discovers a subset of RIR WAV files, computes a clarity/definition
metric for each using the C++ backend, and prints the results. The workflow is:

1. Extend ``sys.path`` to include the compiled C++ extension (expected under
   ``code/cpp/build/Release`` relative to the project root).
2. Collect all ``.wav`` files under ``../data/RIRs/`` (relative to the script).
3. Keep only those whose pathname contains the substring ``'sintetica'``.
4. Limit to the first ``audios`` files.
5. Load each file at the target sampling rate ``fs`` using ``librosa.load``.
6. Compute the Definition metric via ``DefinitionCalculator.calculate``.
7. Print the file path and computed value.

Run from ./ReverbNet/scripts or adjust the relative paths as needed.
"""

import sys
import os

# Append the project build path (where the compiled extension lives) to sys.path.
build_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "code", "cpp", "build", "Release"
)
sys.path.append(build_path)

from audio_processing import DefinitionCalculator
import glob
from librosa import load

if __name__ == '__main__':
    files = glob.glob('../data/RIRs/*.wav')
    files = [audio for audio in files if 'sintetica' in audio]
    fs = 16000
    audios = 10  # Number of audio files to process.
    files = files[:audios]  # Limit to the first 'audios' files.

    for file in files:
        data, fs = load(file, sr=fs)
        print(file)

        definition_cpp = DefinitionCalculator.calculate(data, fs)

        print(f'Definition CPP: {definition_cpp}')