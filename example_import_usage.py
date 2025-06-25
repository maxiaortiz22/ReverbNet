#!/usr/bin/env python3
"""
Ejemplo de uso de ReverbNet
Este script muestra cómo importar y usar las funciones de ReverbNet
"""

# Agregar el directorio code al path de Python
import sys; sys.path.append('code')

# Importaciones principales
from code import (
    model,
    reshape_data,
    normalize_descriptors,
    prediction,
    descriptors_err,
    save_exp_data,
    DataBase,
    read_dataset,
    import_configs_objs
)

# Importaciones de parámetros acústicos
from code.parameters_calculation import (
    TAE,
    tr_lundeby,
    pink_noise,
    colored_noise,
    drr_aug,
    tr_augmentation
)

def ejemplo_uso():
    """Ejemplo de cómo usar las funciones de ReverbNet"""
    
    print("🚀 Ejemplo de uso de ReverbNet")
    print("=" * 40)
    
    # Ejemplo 1: Crear un modelo
    print("\n1. Creando un modelo de red neuronal:")
    try:
        modelo = model(
            filters=[32, 18, 8, 4],
            kernel_size=[10, 5, 5, 5],
            activation=['relu', 'relu', 'relu', 'relu'],
            pool_size=[2, 2, 2],
            learning_rate=0.001
        )
        print("✅ Modelo creado exitosamente")
        print(f"   - Número de parámetros: {modelo.count_params():,}")
    except Exception as e:
        print(f"❌ Error creando modelo: {e}")
    
    # Ejemplo 2: Generar ruido rosa
    print("\n2. Generando ruido rosa:")
    try:
        ruido = pink_noise(1000)
        print("✅ Ruido rosa generado exitosamente")
        print(f"   - Longitud: {len(ruido)} muestras")
        print(f"   - Valor máximo: {ruido.max():.4f}")
        print(f"   - Valor mínimo: {ruido.min():.4f}")
    except Exception as e:
        print(f"❌ Error generando ruido: {e}")
    
    # Ejemplo 3: Mostrar funciones disponibles
    print("\n3. Funciones disponibles:")
    funciones_principales = [
        'model', 'reshape_data', 'normalize_descriptors',
        'prediction', 'descriptors_err', 'save_exp_data',
        'DataBase', 'read_dataset', 'import_configs_objs'
    ]
    
    funciones_parametros = [
        'TAE', 'tr_lundeby', 'pink_noise', 'colored_noise',
        'drr_aug', 'tr_augmentation'
    ]
    
    print("   Funciones principales:")
    for func in funciones_principales:
        print(f"   - {func}")
    
    print("   Funciones de parámetros acústicos:")
    for func in funciones_parametros:
        print(f"   - {func}")
    
    print("\n✅ Todas las importaciones funcionan correctamente!")
    print("\n💡 Para usar en tu propio script, simplemente agrega:")
    print("   import sys; sys.path.append('code')")
    print("   from code import model, DataBase, read_dataset")

if __name__ == "__main__":
    ejemplo_uso() 