#!/usr/bin/env python3
"""
Simple Model Analysis for ReverbNet

This script calculates the computational complexity and model size
without requiring TensorFlow installation.
"""

def calculate_conv1d_params(input_channels, filters, kernel_size):
    """Calculate parameters for a Conv1D layer."""
    # Conv1D parameters: (kernel_size * input_channels + 1) * filters
    # +1 for bias term
    return (kernel_size * input_channels + 1) * filters

def calculate_conv1d_macs(input_length, input_channels, filters, kernel_size):
    """Calculate MACs for a Conv1D layer."""
    # Output length after convolution
    output_length = input_length - kernel_size + 1
    # MACs = input_channels * kernel_size * output_length * filters
    return input_channels * kernel_size * output_length * filters

def calculate_maxpool1d_output_length(input_length, pool_size):
    """Calculate output length after MaxPool1D."""
    return input_length // pool_size

def calculate_dense_params(input_features, output_features):
    """Calculate parameters for a Dense layer."""
    # Dense parameters: (input_features + 1) * output_features
    # +1 for bias term
    return (input_features + 1) * output_features

def calculate_dense_macs(input_features, output_features):
    """Calculate MACs for a Dense layer."""
    return input_features * output_features

def analyze_reverbnet_model():
    """Analyze the ReverbNet model architecture."""
    
    # Model configuration
    input_length = 200
    input_channels = 1
    filters = [32, 18, 8, 4]
    kernel_sizes = [10, 5, 5, 5]
    pool_sizes = [2, 2, 2]
    dense_output = 4
    
    print("=" * 60)
    print("REVERBNET MODEL ANALYSIS")
    print("=" * 60)
    
    print(f"\n📊 MODEL ARCHITECTURE:")
    print("-" * 30)
    print(f"Input shape: ({input_length}, {input_channels})")
    print(f"Filters: {filters}")
    print(f"Kernel sizes: {kernel_sizes}")
    print(f"Pool sizes: {pool_sizes}")
    print(f"Output: {dense_output} values")
    
    # Track feature map sizes through the network
    current_length = input_length
    current_channels = input_channels
    
    total_params = 0
    total_macs = 0
    
    print(f"\n🔢 LAYER-BY-LAYER ANALYSIS:")
    print("-" * 30)
    
    # Analyze each convolutional layer
    for i, (filt, kernel_size, pool_size) in enumerate(zip(filters, kernel_sizes, pool_sizes + [None])):
        layer_num = i + 1
        
        # Conv1D layer
        conv_params = calculate_conv1d_params(current_channels, filt, kernel_size)
        conv_macs = calculate_conv1d_macs(current_length, current_channels, filt, kernel_size)
        
        print(f"Layer {layer_num} Conv1D:")
        print(f"  Input: ({current_length}, {current_channels})")
        print(f"  Filters: {filt}, Kernel: {kernel_size}")
        print(f"  Parameters: {conv_params:,}")
        print(f"  MACs: {conv_macs:,}")
        
        total_params += conv_params
        total_macs += conv_macs
        
        # Update dimensions after convolution
        current_length = current_length - kernel_size + 1
        current_channels = filt
        
        # MaxPool1D layer (except for last conv layer)
        if pool_size is not None:
            current_length = calculate_maxpool1d_output_length(current_length, pool_size)
            print(f"  MaxPool1D: pool_size={pool_size}")
            print(f"  Output after pool: ({current_length}, {current_channels})")
        
        # BatchNorm parameters (4 per filter: gamma, beta, moving_mean, moving_var)
        batchnorm_params = 4 * filt
        total_params += batchnorm_params
        print(f"  BatchNorm parameters: {batchnorm_params:,}")
        
        print()
    
    # Dense layer
    # After the last conv layer, we flatten: current_length * current_channels
    dense_input_features = current_length * current_channels
    dense_params = calculate_dense_params(dense_input_features, dense_output)
    dense_macs = calculate_dense_macs(dense_input_features, dense_output)
    
    print(f"Final Dense Layer:")
    print(f"  Input features: {dense_input_features:,}")
    print(f"  Output features: {dense_output}")
    print(f"  Parameters: {dense_params:,}")
    print(f"  MACs: {dense_macs:,}")
    
    total_params += dense_params
    total_macs += dense_macs
    
    # Summary
    print(f"\n📈 SUMMARY:")
    print("-" * 30)
    print(f"Total parameters: {total_params:,}")
    print(f"Total MACs: {total_macs:,}")
    print(f"Total MACs (millions): {total_macs / 1e6:.2f}M")
    
    # Model size calculations
    print(f"\n💾 MODEL SIZE:")
    print("-" * 30)
    model_size_float32 = total_params * 4  # 4 bytes per float32
    model_size_mb = model_size_float32 / (1024 * 1024)
    
    print(f"Model size (float32): {model_size_mb:.2f} MB")
    print(f"Model size (float16): {model_size_mb / 2:.2f} MB")
    print(f"Model size (int8): {model_size_mb / 4:.2f} MB")
    
    # TFLite size estimation (typically 20-40% smaller due to quantization)
    tflite_size_mb = model_size_mb * 0.3  # Rough estimate
    print(f"Estimated TFLite size: {tflite_size_mb:.2f} MB")
    print(f"Estimated compression ratio: {model_size_mb / tflite_size_mb:.1f}x")
    
    # Memory requirements for inference
    print(f"\n🧠 INFERENCE MEMORY REQUIREMENTS:")
    print("-" * 30)
    print(f"Model weights (float32): {model_size_mb:.2f} MB")
    print(f"Model weights (float16): {model_size_mb / 2:.2f} MB")
    print(f"Model weights (int8): {model_size_mb / 4:.2f} MB")
    
    # Feature map memory (rough estimation)
    max_feature_maps = max(filters) * 200  # Maximum feature map size
    feature_map_memory_mb = (max_feature_maps * 4) / (1024 * 1024)  # float32
    print(f"Peak feature maps memory: ~{feature_map_memory_mb:.2f} MB")
    
    print(f"\n" + "=" * 60)
    
    return {
        'total_params': total_params,
        'total_macs': total_macs,
        'model_size_mb': model_size_mb,
        'tflite_size_mb': tflite_size_mb
    }

if __name__ == "__main__":
    results = analyze_reverbnet_model()
    
    print(f"\n✅ Analysis complete!")
    print(f"Key metrics:")
    print(f"  • Parameters: {results['total_params']:,}")
    print(f"  • MACs: {results['total_macs']:,} ({results['total_macs'] / 1e6:.2f}M)")
    print(f"  • Model size: {results['model_size_mb']:.2f} MB")
    print(f"  • TFLite size: ~{results['tflite_size_mb']:.2f} MB") 