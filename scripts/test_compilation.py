"""
Minimal smoke test to verify import and basic instantiation of the C++
``audio_processing`` extension module.

Workflow
--------
1. Extend ``sys.path`` to include the compiled extension located under
   ``code/cpp/build/Release`` relative to the project root.
2. Import the ``audio_processing`` module.
3. Instantiate an ``AudioProcessor`` object to confirm symbols are available.

Use this script as a quick sanity check after (re)building the C++ code.
"""

import sys
import os

# Append the project build path (compiled C++ extension) to sys.path.
build_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "code", "cpp", "build", "Release"
)
sys.path.append(build_path)

import audio_processing

# Example usage: instantiate the AudioProcessor class.
processor = audio_processing.AudioProcessor()
# Use other methods as needed.