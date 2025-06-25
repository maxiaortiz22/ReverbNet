"""
Wrapper para el módulo compilado audio_processing
"""

import os
import sys
from pathlib import Path

# Obtener la ruta del directorio actual
current_dir = Path(__file__).parent

# Buscar el módulo compilado en diferentes ubicaciones posibles
possible_paths = [
    current_dir / "build" / "Release" / "audio_processing.cp312-win_amd64.pyd",
    current_dir / "build" / "audio_processing.cp312-win_amd64.pyd",
    current_dir / "audio_processing.cp312-win_amd64.pyd",
    current_dir / "build" / "Release" / "audio_processing.pyd",
    current_dir / "build" / "audio_processing.pyd",
    current_dir / "audio_processing.pyd",
]

# Buscar el archivo compilado
module_path = None
for path in possible_paths:
    if path.exists():
        module_path = path
        break

if module_path is None:
    # Si no se encuentra, buscar archivos .pyd en el directorio build
    build_dir = current_dir / "build"
    if build_dir.exists():
        for root, dirs, files in os.walk(build_dir):
            for file in files:
                if file.startswith("audio_processing") and file.endswith(".pyd"):
                    module_path = Path(root) / file
                    break
            if module_path:
                break

if module_path is None:
    raise ImportError(
        f"No se pudo encontrar el módulo compilado audio_processing. "
        f"Busqué en: {[str(p) for p in possible_paths]}"
    )

# Agregar el directorio del módulo al path
module_dir = module_path.parent
if str(module_dir) not in sys.path:
    sys.path.insert(0, str(module_dir))

try:
    # Importar el módulo compilado
    import audio_processing
except ImportError as e:
    raise ImportError(
        f"Error importando el módulo compilado audio_processing desde {module_path}: {e}"
    )

# Re-exportar todo del módulo compilado
__all__ = []
for name in dir(audio_processing):
    if not name.startswith('_'):
        globals()[name] = getattr(audio_processing, name)
        __all__.append(name)