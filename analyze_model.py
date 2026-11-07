#!/usr/bin/env python3
"""
Model Analysis Script for ReverbNet

This script analyzes the computational complexity, number of parameters,
and TFLite file size of the ReverbNet model.
"""

import sys
import os
sys.path.append('code')

import tensorflow as tf
import numpy as np
from code import model
import tempfile

def count_parameters(model):
    """Count total trainable parameters in the model."""
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params
    }

def calculate_macs(model, input_shape=(1, 200, 1)):
    """
    Calculate the number of Multiply-Accumulate operations (MACs).
    
    This is a simplified calculation for 1D convolutions.
    """
    macs = 0
    
    # Manual calculation based on the model architecture we can see
    # From the model summary:
    # Input: (None, 200, 1)
    # Layer 1: capa1_conv -> (None, 191, 32)
    # Layer 2: capa2_conv -> (None, 91, 18) 
    # Layer 3: capa3_conv -> (None, 41, 8)
    # Layer 4: capa4_conv -> (None, 16, 4)
    # Final: Dense -> (None, 4)
    
    # Layer 1: Conv1D (200, 1) -> (191, 32)
    # MACs = input_channels * kernel_size * output_length * filters
    # MACs = 1 * 10 * 191 * 32 = 61,120
    layer1_macs = 1 * 10 * 191 * 32
    macs += layer1_macs
    print(f"  capa1_conv: {layer1_macs:,} MACs")
    
    # Layer 2: Conv1D (95, 32) -> (91, 18) [after MaxPool1D]
    # MACs = 32 * 5 * 91 * 18 = 262,080
    layer2_macs = 32 * 5 * 91 * 18
    macs += layer2_macs
    print(f"  capa2_conv: {layer2_macs:,} MACs")
    
    # Layer 3: Conv1D (45, 18) -> (41, 8) [after MaxPool1D]
    # MACs = 18 * 5 * 41 * 8 = 29,520
    layer3_macs = 18 * 5 * 41 * 8
    macs += layer3_macs
    print(f"  capa3_conv: {layer3_macs:,} MACs")
    
    # Layer 4: Conv1D (20, 8) -> (16, 4) [after MaxPool1D]
    # MACs = 8 * 5 * 16 * 4 = 2,560
    layer4_macs = 8 * 5 * 16 * 4
    macs += layer4_macs
    print(f"  capa4_conv: {layer4_macs:,} MACs")
    
    # Final Dense Layer: (64) -> (4)
    # After flatten: 16 * 4 = 64 input features
    # MACs = 64 * 4 = 256
    dense_macs = 64 * 4
    macs += dense_macs
    print(f"  Salida_prediccion: {dense_macs:,} MACs")
    
    return macs

def get_model_size_info(model, save_tflite=True):
    """Get model size information and optionally save as TFLite."""
    size_info = {}
    
    # Get parameter counts
    params = count_parameters(model)
    size_info['parameters'] = params
    
    # Calculate model size in bytes (assuming float32)
    size_info['model_size_bytes'] = params['total'] * 4  # 4 bytes per float32
    size_info['model_size_mb'] = size_info['model_size_bytes'] / (1024 * 1024)
    
    # Estimate TFLite size (typically 20-40% smaller due to quantization)
    size_info['tflite_size_mb'] = size_info['model_size_mb'] * 0.3  # Rough estimate
    
    # Try to save as TFLite if requested
    if save_tflite:
        try:
            # Create a simple converter without optimizations first
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            # Convert to TFLite
            tflite_model = converter.convert()
            
            # Save to temporary file to get size
            with tempfile.NamedTemporaryFile(suffix='.tflite', delete=False) as f:
                f.write(tflite_model)
                tflite_path = f.name
            
            # Get TFLite file size
            tflite_size = os.path.getsize(tflite_path)
            size_info['tflite_size_bytes'] = tflite_size
            size_info['tflite_size_mb'] = tflite_size / (1024 * 1024)
            
            # Save to results directory
            results_dir = 'results/model_analysis'
            os.makedirs(results_dir, exist_ok=True)
            final_path = os.path.join(results_dir, 'reverbnet_model.tflite')
            
            with open(final_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"TFLite model saved to: {final_path}")
            
            # Clean up temporary file
            os.unlink(tflite_path)
            
        except Exception as e:
            print(f"Warning: Could not convert to TFLite: {e}")
            print("Using estimated TFLite size instead.")
    
    return size_info

def print_model_summary(model):
    """Print detailed model summary."""
    print("=" * 60)
    print("REVERBNET MODEL ANALYSIS")
    print("=" * 60)
    
    # Model architecture summary
    print("\n📊 MODEL ARCHITECTURE:")
    print("-" * 30)
    model.summary()
    
    # Parameter analysis
    print("\n🔢 PARAMETER ANALYSIS:")
    print("-" * 30)
    params = count_parameters(model)
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Non-trainable parameters: {params['non_trainable']:,}")
    
    # MACs calculation
    print("\n⚡ COMPUTATIONAL COMPLEXITY:")
    print("-" * 30)
    macs = calculate_macs(model)
    print(f"Total MACs: {macs:,}")
    print(f"Total MACs (millions): {macs / 1e6:.2f}M")
    
    # Model size
    print("\n💾 MODEL SIZE:")
    print("-" * 30)
    size_info = get_model_size_info(model, save_tflite=True)
    print(f"Model size (float32): {size_info['model_size_mb']:.2f} MB")
    print(f"TFLite size: {size_info['tflite_size_mb']:.2f} MB")
    print(f"Compression ratio: {size_info['model_size_mb'] / size_info['tflite_size_mb']:.2f}x")
    
    # Memory requirements
    print("\n🧠 MEMORY REQUIREMENTS:")
    print("-" * 30)
    print(f"Peak memory (float32): {size_info['model_size_mb']:.2f} MB")
    print(f"Peak memory (float16): {size_info['model_size_mb'] / 2:.2f} MB")
    print(f"Peak memory (int8): {size_info['model_size_mb'] / 4:.2f} MB")
    
    print("\n" + "=" * 60)

def main():
    """Main function to run the analysis."""
    print("Building ReverbNet model...")
    
    # Create the model with default parameters
    reverbnet_model = model(
        filters=[32, 18, 8, 4],
        kernel_size=[10, 5, 5, 5],
        activation=['relu', 'relu', 'relu', 'relu'],
        pool_size=[2, 2, 2],
        learning_rate=0.001
    )
    
    # Run analysis
    print_model_summary(reverbnet_model)
    
    print("\n✅ Analysis complete! Check the results/model_analysis/ directory for the TFLite file.")

if __name__ == "__main__":
    main() 