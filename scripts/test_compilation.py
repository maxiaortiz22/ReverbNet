import sys
import os

# Añadir la ruta del proyecto a sys.path
build_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code", "cpp", "build", "Release")
sys.path.append(build_path)

import audio_processing

# Ejemplo de uso
processor = audio_processing.AudioProcessor()
# Usar otros métodos según necesites