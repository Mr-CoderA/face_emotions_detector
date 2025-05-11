"""
Utility module for GPU checks and memory tracking
"""
import os
import torch
import gc

def check_gpu_availability():
    """
    Check if GPU is available and verify CUDA is properly configured.
    Raises RuntimeError if no GPU is available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This application requires a GPU.")
    
    # Force GPU-only execution
    torch.device("cuda")
    
    # Print GPU info
    gpu_name = torch.cuda.get_device_name(0)
    cuda_version = torch.version.cuda
    
    print(f"Using GPU: {gpu_name}")
    print(f"CUDA Version: {cuda_version}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    return True

def estimate_memory_usage():
    """
    Estimate current GPU memory usage.
    Returns memory usage in bytes.
    """
    # Empty cache to get accurate measurement
    torch.cuda.empty_cache()
    
    # Get current memory allocation
    memory_allocated = torch.cuda.memory_allocated()
    memory_reserved = torch.cuda.memory_reserved()
    
    return memory_allocated, memory_reserved

def cleanup_gpu_memory():
    """
    Perform thorough GPU memory cleanup.
    """
    # Clear PyTorch cache
    torch.cuda.empty_cache()
    
    # Run garbage collector
    gc.collect()
    
    # Release all unoccupied cached memory
    torch.cuda.empty_cache()
    
    return True

def log_memory_usage(label="Current"):
    """
    Log current GPU memory usage.
    """
    allocated, reserved = estimate_memory_usage()
    print(f"{label} - GPU Memory: Allocated: {allocated / (1024**3):.2f} GB, "
          f"Reserved: {reserved / (1024**3):.2f} GB") 