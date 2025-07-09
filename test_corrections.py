#!/usr/bin/env python3
"""
Script de prueba para verificar las correcciones en generate_database.py
"""

import sys
import os
sys.path.append('code')

from code import import_configs_objs
from code import DataBase
import numpy as np

def check_audio_files():
    """Verificar que existen archivos de audio necesarios"""
    print("🔍 Verificando archivos de audio...")
    print("-" * 30)
    
    # Verificar RIRs
    rir_path = 'data/RIRs'
    if os.path.exists(rir_path):
        rir_files = [f for f in os.listdir(rir_path) if f.endswith('.wav')]
        print(f"✅ RIRs encontradas: {len(rir_files)} archivos")
        if len(rir_files) > 0:
            print(f"   Ejemplos: {rir_files[:3]}")
    else:
        print("❌ Carpeta data/RIRs no existe")
        return False
    
    # Verificar audios de speech para entrenamiento
    speech_train_path = 'data/Speech/train'
    if os.path.exists(speech_train_path):
        speech_train_files = [f for f in os.listdir(speech_train_path) if f.endswith('.wav')]
        print(f"✅ Audios de speech (train): {len(speech_train_files)} archivos")
        if len(speech_train_files) > 0:
            print(f"   Ejemplos: {speech_train_files[:3]}")
    else:
        print("❌ Carpeta data/Speech/train no existe")
        return False
    
    # Verificar audios de speech para testing
    speech_test_path = 'data/Speech/test'
    if os.path.exists(speech_test_path):
        speech_test_files = [f for f in os.listdir(speech_test_path) if f.endswith('.wav')]
        print(f"✅ Audios de speech (test): {len(speech_test_files)} archivos")
        if len(speech_test_files) > 0:
            print(f"   Ejemplos: {speech_test_files[:3]}")
    else:
        print("❌ Carpeta data/Speech/test no existe")
        return False
    
    # Verificar que hay suficientes archivos
    if len(speech_train_files) == 0:
        print("❌ No hay audios de speech para entrenamiento")
        return False
    
    if len(speech_test_files) == 0:
        print("❌ No hay audios de speech para testing")
        return False
    
    print("✅ Todos los archivos de audio están disponibles")
    return True

def test_database_creation():
    """Probar la creación de la base de datos con las correcciones"""
    
    print("🧪 Probando correcciones en generate_database.py")
    print("=" * 50)
    
    # Primero verificar archivos de audio
    if not check_audio_files():
        print("❌ Faltan archivos de audio necesarios")
        return False
    
    # Cargar configuración
    try:
        config = import_configs_objs('configs/exp1.py')
        print("✅ Configuración cargada correctamente")
    except Exception as e:
        print(f"❌ Error cargando configuración: {e}")
        return False
    
    # Crear instancia de DataBase
    try:
        database = DataBase(
            config['files_speech_train'], 
            config['files_speech_test'], 
            config['files_rirs'], 
            config['tot_sinteticas'], 
            config['to_augmentate'],
            config['rirs_for_training'], 
            config['rirs_for_testing'], 
            config['bands'], 
            config['filter_type'], 
            config['fs'], 
            config['max_ruido_dB'],
            config['order'], 
            config['add_noise'], 
            config['snr'], 
            config['tr_aug'], 
            config['drr_aug'], 
            batch_size=100  # Batch pequeño para pruebas
        )
        print("✅ Instancia de DataBase creada correctamente")
    except Exception as e:
        print(f"❌ Error creando DataBase: {e}")
        return False
    
    # Verificar que los sets se crearon correctamente
    print(f"📊 RIRs para entrenamiento: {len(database.rirs_for_training)}")
    print(f"📊 RIRs para testing: {len(database.rirs_for_testing)}")
    print(f"📊 RIRs para aumentar: {len(database.to_augmentate)}")
    
    # Verificar que hay datos de test
    if len(database.rirs_for_testing) == 0:
        print("❌ No hay RIRs para testing!")
        return False
    else:
        print("✅ Hay RIRs para testing")
    
    # Probar el filtro de octava
    try:
        from code.cpp import audio_processing
        bpfilter = audio_processing.OctaveFilterBank(filter_order=4)
        print("✅ Filtro de octava creado correctamente")
        
        # Probar procesamiento
        test_signal = np.random.randn(1000).astype(np.float32)
        filtered = bpfilter.process(test_signal)
        print(f"✅ Filtro procesó señal: {filtered.shape}")
        
    except Exception as e:
        print(f"❌ Error con filtro de octava: {e}")
        return False
    
    # Probar función _should_augment_rir
    try:
        # Tomar una RIR de ejemplo
        test_rir = list(database.rirs_for_training)[0]
        rir_name = test_rir.split('.wav')[0]
        
        should_augment = database._should_augment_rir(rir_name)
        print(f"✅ _should_augment_rir funciona: {should_augment}")
        
    except Exception as e:
        print(f"❌ Error en _should_augment_rir: {e}")
        return False
    
    # Probar procesamiento de una RIR real (opcional)
    try:
        print("\n🔬 Probando procesamiento de una RIR real...")
        # Tomar la primera RIR de training
        test_rir_file = list(database.rirs_for_training)[0]
        print(f"   RIR de prueba: {test_rir_file}")
        
        # Simular el procesamiento (sin guardar)
        result = database.calc_database_multiprocess(test_rir_file)
        if result is not None:
            print("✅ Procesamiento de RIR funciona correctamente")
        else:
            print("⚠️  Procesamiento de RIR devolvió None (posiblemente ya existe la BD)")
        
    except Exception as e:
        print(f"❌ Error en procesamiento de RIR: {e}")
        return False
    
    print("\n🎉 Todas las pruebas pasaron!")
    return True

if __name__ == "__main__":
    success = test_database_creation()
    if success:
        print("\n✅ Las correcciones están funcionando correctamente")
        print("🚀 Puedes ejecutar el experimento completo ahora")
    else:
        print("\n❌ Hay problemas que necesitan ser corregidos") 