#!/usr/bin/env python3
"""
Manual MACs calculation for ReverbNet based on the model architecture.
"""

def calculate_macs_manual():
    """Calculate MACs manually based on the model architecture."""
    
    print("=" * 60)
    print("REVERBNET MANUAL MACs CALCULATION")
    print("=" * 60)
    
    # From the model summary we can see:
    # Input: (None, 200, 1)
    # Layer 1: capa1_conv -> (None, 191, 32)
    # Layer 2: capa2_conv -> (None, 91, 18) 
    # Layer 3: capa3_conv -> (None, 41, 8)
    # Layer 4: capa4_conv -> (None, 16, 4)
    # Final: Dense -> (None, 4)
    
    total_macs = 0
    
    print("\n🔢 LAYER-BY-LAYER MACs CALCULATION:")
    print("-" * 40)
    
    # Layer 1: Conv1D (200, 1) -> (191, 32)
    # MACs = input_channels * kernel_size * output_length * filters
    # MACs = 1 * 10 * 191 * 32 = 61,120
    layer1_macs = 1 * 10 * 191 * 32
    total_macs += layer1_macs
    print(f"Layer 1 (capa1_conv): {layer1_macs:,} MACs")
    print(f"  Input: (200, 1), Output: (191, 32), Kernel: 10")
    
    # Layer 2: Conv1D (95, 32) -> (91, 18) [after MaxPool1D]
    # MACs = 32 * 5 * 91 * 18 = 262,080
    layer2_macs = 32 * 5 * 91 * 18
    total_macs += layer2_macs
    print(f"Layer 2 (capa2_conv): {layer2_macs:,} MACs")
    print(f"  Input: (95, 32), Output: (91, 18), Kernel: 5")
    
    # Layer 3: Conv1D (45, 18) -> (41, 8) [after MaxPool1D]
    # MACs = 18 * 5 * 41 * 8 = 29,520
    layer3_macs = 18 * 5 * 41 * 8
    total_macs += layer3_macs
    print(f"Layer 3 (capa3_conv): {layer3_macs:,} MACs")
    print(f"  Input: (45, 18), Output: (41, 8), Kernel: 5")
    
    # Layer 4: Conv1D (20, 8) -> (16, 4) [after MaxPool1D]
    # MACs = 8 * 5 * 16 * 4 = 2,560
    layer4_macs = 8 * 5 * 16 * 4
    total_macs += layer4_macs
    print(f"Layer 4 (capa4_conv): {layer4_macs:,} MACs")
    print(f"  Input: (20, 8), Output: (16, 4), Kernel: 5")
    
    # Final Dense Layer: (64) -> (4)
    # After flatten: 16 * 4 = 64 input features
    # MACs = 64 * 4 = 256
    dense_macs = 64 * 4
    total_macs += dense_macs
    print(f"Final Dense: {dense_macs:,} MACs")
    print(f"  Input: 64 features, Output: 4 features")
    
    print(f"\n📈 SUMMARY:")
    print("-" * 40)
    print(f"Total MACs: {total_macs:,}")
    print(f"Total MACs (millions): {total_macs / 1e6:.2f}M")
    
    # Parameter count from the model summary
    total_params = 4634
    print(f"\n💾 MODEL SIZE:")
    print("-" * 40)
    print(f"Total parameters: {total_params:,}")
    model_size_mb = (total_params * 4) / (1024 * 1024)  # float32
    print(f"Model size (float32): {model_size_mb:.2f} MB")
    print(f"Model size (float16): {model_size_mb / 2:.2f} MB")
    print(f"Model size (int8): {model_size_mb / 4:.2f} MB")
    
    # TFLite size estimation
    tflite_size_mb = model_size_mb * 0.3  # Rough estimate
    print(f"Estimated TFLite size: {tflite_size_mb:.2f} MB")
    print(f"Estimated compression ratio: {model_size_mb / tflite_size_mb:.1f}x")
    
    print(f"\n" + "=" * 60)
    
    return {
        'total_macs': total_macs,
        'total_params': total_params,
        'model_size_mb': model_size_mb,
        'tflite_size_mb': tflite_size_mb
    }

if __name__ == "__main__":
    results = calculate_macs_manual()
    
    print(f"\n✅ Manual calculation complete!")
    print(f"Key metrics:")
    print(f"  • Parameters: {results['total_params']:,}")
    print(f"  • MACs: {results['total_macs']:,} ({results['total_macs'] / 1e6:.2f}M)")
    print(f"  • Model size: {results['model_size_mb']:.2f} MB")
    print(f"  • TFLite size: ~{results['tflite_size_mb']:.2f} MB") 