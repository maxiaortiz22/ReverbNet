# ReverbNet
Red neuronal creada para estimar parámetros acústicos de una sala a partir de un audio de voz grabado en el lugar.

## 📦 Uso de Importaciones

El proyecto está configurado para usar importaciones relativas. En cualquier script que quieras usar las funciones de ReverbNet, simplemente agrega al inicio:

```python
import sys; sys.path.append('code')

# Luego puedes importar normalmente:
from code import model, DataBase, read_dataset
from code.parameters_calculation import TAE, tr_lundeby, pink_noise
```

### Ejemplo de uso:

```python
import sys; sys.path.append('code')

# Importaciones principales
from code import (
    model,                    # Modelo de red neuronal
    reshape_data,             # Función para redimensionar datos
    normalize_descriptors,    # Normalización de descriptores
    prediction,               # Función de predicción
    descriptors_err,          # Cálculo de errores
    save_exp_data,           # Guardar datos de experimento
    DataBase,                # Clase para generar base de datos
    read_dataset,            # Leer dataset
    import_configs_objs      # Importar configuraciones
)

# Importaciones de parámetros acústicos
from code.parameters_calculation import (
    TAE,                     # Time-domain Acoustic Energy
    tr_lundeby,              # TR Lundeby method
    pink_noise,              # Generación de ruido rosa
    colored_noise,           # Generación de ruido coloreado
    drr_aug,                 # Augmentación DRR
    tr_augmentation          # Augmentación TR
)
```

## ✅ Verificación de la Instalación

Para verificar que todo funciona correctamente:

```bash
python example_usage.py
```

Este script probará todas las importaciones y funciones principales del proyecto.

## 🔧 Configuración del Entorno

### Exportar el environment a un archivo:

```bash
conda env export --name ReverbNet > ReverbNet.yml
```

### Para crear el mismo environment desde otra compu:

```bash
conda env create -f ReverbNet.yml
```

### Para activar el environment:

```bash
conda activate ReverbNet
```

### Para instalar las dependencias:

```bash
pip install -r requirements.txt
```

### Para desactivar el environment:

```bash
conda deactivate
```

## 📁 Submódulo

Cuando clones todo el repositorio también vas a necesitar activar el submódulo que clona pybind11 para poder usarlo como librería.

Para activarlo hay que ejecutar este comando en el main root:

```bash
git submodule update --init --recursive
```

### Para correr los entrenamientos:

```bash
python run.py --config configs/exp1.py
```

## 🎯 Ventajas de esta Configuración

1. **Sin instalación**: No necesitas instalar el paquete en tu sistema
2. **Portabilidad**: Funciona en cualquier máquina que tenga las dependencias
3. **Simplicidad**: Solo necesitas agregar una línea al inicio de tus scripts
4. **Flexibilidad**: Puedes modificar el código y los cambios se reflejan inmediatamente
5. **Compatibilidad**: Funciona con versiones modernas de SciPy y otras librerías

## 🔧 Solución de Problemas

### Error con `hann` de SciPy
Si encuentras errores con la importación de `hann` desde `scipy.signal`, el código ya está preparado para manejar diferentes versiones de SciPy automáticamente.

### Error con módulo `audio_processing`
Si hay problemas con el módulo compilado `audio_processing`, asegúrate de que:
1. El submódulo pybind11 esté inicializado
2. El módulo esté compilado correctamente
3. El archivo `code/cpp/audio_processing.py` exista (wrapper automático)

## Uso

### Parámetros de línea de comando

- `--config`: Archivo de configuración del experimento
- `--save_batch_size`: Tamaño del lote para guardado incremental (por defecto: 1000)

### Ejemplo de uso

```bash
# Ejecutar experimento con guardado incremental de 500 muestras por lote
python run.py --config configs/exp1.py --save_batch_size 500
```

### Diferencias entre batch_size

**IMPORTANTE**: Hay dos tipos diferentes de `batch_size` en el código:

1. **`save_batch_size`** (parámetro de línea de comando):
   - Controla cuántas muestras se guardan en cada archivo temporal durante el procesamiento
   - Ayuda a controlar el uso de memoria RAM
   - Valores recomendados: 100-1000 (dependiendo de la RAM disponible)

2. **`batch_size`** (en archivos de configuración):
   - Controla el tamaño del lote para entrenamiento del modelo TensorFlow
   - Afecta la velocidad y convergencia del entrenamiento
   - Valores típicos: 32-2048 (dependiendo del modelo y GPU)

### Configuración de memoria

Para evitar errores de memoria (`BrokenProcessPool`):

1. **Reduce `save_batch_size`** si tienes poca RAM:
   ```bash
   python run.py --config configs/exp1.py --save_batch_size 5000
   ```

2. **Ajusta `batch_size` en el archivo de configuración** según tu GPU:
   ```python
   # En configs/exp1.py
   batch_size = 512  # Reducir si hay problemas de memoria GPU
   ```