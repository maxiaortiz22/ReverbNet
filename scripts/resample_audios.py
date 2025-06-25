import os
import librosa
import resampy
import soundfile as sf
import numpy as np

# Configuración
input_folder = "../../../octagonOmni/Omni"  # ← Cambiar por tu ruta real
output_sample_rate = 16000          # Nuevo sample rate deseado
prefix = "octagon_"               # Prefijo que se agrega al nombre

# Crear carpeta de salida si no existe
output_folder = os.path.join(input_folder, "resampled")
os.makedirs(output_folder, exist_ok=True)

# Recorrer todos los archivos de la carpeta
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
        original_path = os.path.join(input_folder, filename)

        # Leer el audio
        y, sr = librosa.load(original_path, sr=None)  # sr=None para conservar sample rate original

        # Remuestrear usando resampy
        y_resampled = resampy.resample(y, sr, output_sample_rate)
        y_resampled = y_resampled/np.max(np.abs(y_resampled))  # Normalizar a [-1, 1]

        # Crear nuevo nombre con prefijo
        new_filename = prefix + filename
        output_path = os.path.join(output_folder, new_filename)

        # Guardar audio remuestreado
        sf.write(output_path, y_resampled, output_sample_rate)

        print(f"Procesado: {filename} → {new_filename} @ {output_sample_rate} Hz")

print("🎉 Todos los archivos fueron procesados correctamente.")
