"""
Batch test for the C++ ``ClarityCalculator`` extension.

This script loads a subset of RIR WAV files, computes clarity metrics (C50 and
C80 windows, specified in milliseconds) using the C++ backend, and prints the
results.

Workflow
--------
1. Extend ``sys.path`` to include the compiled C++ extension (expected under
   ``code/cpp/build/Release`` relative to the project root).
2. Discover all ``.wav`` files in ``../data/RIRs/`` (relative to this script).
3. Keep only files whose path contains the substring ``'sintetica'``.
4. Limit processing to the first ``audios`` files.
5. Load each file at sampling rate ``fs`` via ``librosa.load``.
6. Compute clarity using :func:`ClarityCalculator.calculate` for 50 ms and
   80 ms integration times.
7. Print file path and computed clarity values.

Run from the repo root or adjust the relative paths if needed.
"""

import sys
import os

# Append the project build path (compiled C++ extension) to sys.path.
build_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "code", "cpp", "build", "Release"
)
sys.path.append(build_path)

from audio_processing import ClarityCalculator
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

        clarity_cpp = ClarityCalculator.calculate(50, data, fs)
        print(f'Clarity C50 CPP: {clarity_cpp}')

        clartity_80_cpp = ClarityCalculator.calculate(80, data, fs)
        print(f'Clarity C80 CPP: {clartity_80_cpp}')