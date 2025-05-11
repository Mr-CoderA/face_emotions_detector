"""
Advanced GPU memory manager for optimizing performance on systems with limited VRAM.
"""
import os
import torch
import gc
import time
import threading
from typing import Dict, Any, Optional, Callable

import src.config as config
from src.utils.gpu_check import log_memory_usage

class GPUMemoryManager:
    """
    Advanced GPU memory manager for systems with limited VRAM.
    Provides memory monitoring, cleanup, and optimization strategies.
    """
    
    def __init__(self, device="cuda", optimization_level="aggressive"):
        """
        Initialize the GPU memory manager.
        
        Args:
            device: CUDA device to monitor
            optimization_level: Memory optimization aggressiveness 
                                ("normal", "aggressive", "conservative")
        """
        self.device = device
        self.optimization_level = optimization_level
        self.memory_pressure = 0.0  # 0.0 to 1.0, representing memory pressure
        self.last_cleanup_time = 0
        self.cleanup_interval = 5.0  # seconds between cleanup attempts
        self.mem_history = []
        self.warning_threshold = 0.85  # 85% memory usage triggers warning
        self.critical_threshold = 0.95  # 95% memory usage is critical
        
        # Set up background monitoring thread
        self.monitoring_active = False
        self.monitor_thread = None
    
    def start_monitoring(self, callback: Optional[Callable[[float], None]] = None):
        """
        Start background memory monitoring.
        
        Args:
            callback: Optional callback function to be called with memory pressure parameter
        """
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        
        def monitor_task():
            while self.monitoring_active:
                try:
                    self.update_memory_pressure()
                    
                    # Call callback if provided
                    if callback and callable(callback):
                        callback(self.memory_pressure)
                    
                    # Log warnings if memory pressure is high
                    if self.memory_pressure > self.critical_threshold:
                        log_memory_usage("CRITICAL: Very high memory usage")
                        self.emergency_cleanup()
                    elif self.memory_pressure > self.warning_threshold:
                        log_memory_usage("WARNING: High memory usage")
                        
                    # Sleep for a bit
                    time.sleep(2.0)
                except Exception as e:
                    print(f"Error in memory monitoring: {e}")
                    time.sleep(5.0)  # Sleep longer on error
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=monitor_task, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop background memory monitoring"""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
    
    def update_memory_pressure(self) -> float:
        """
        Calculate current memory pressure (0.0 to 1.0).
        
        Returns:
            Memory pressure value between 0.0 (no pressure) and 1.0 (extreme pressure)
        """
        try:
            # Get current memory stats
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            
            # Calculate memory pressure
            # We consider both allocated memory and how much of the reserved pool is used
            reserved_pressure = allocated / max(reserved, 1) if reserved > 0 else 0
            total_pressure = allocated / total
            
            # Combine metrics (weighting allocated/total higher)
            self.memory_pressure = 0.7 * total_pressure + 0.3 * reserved_pressure
            
            # Keep history for trend analysis
            self.mem_history.append(self.memory_pressure)
            if len(self.mem_history) > 10:
                self.mem_history.pop(0)
            
            return self.memory_pressure
            
        except Exception as e:
            print(f"Error updating memory pressure: {e}")
            return 0.0
    
    def get_memory_pressure(self) -> float:
        """Get the current memory pressure value"""
        return self.memory_pressure
    
    def is_memory_critical(self) -> bool:
        """Check if memory usage is at a critical level"""
        return self.memory_pressure > self.critical_threshold
    
    def is_memory_warning(self) -> bool:
        """Check if memory usage is at a warning level"""
        return self.memory_pressure > self.warning_threshold
    
    def cleanup(self, force=False) -> bool:
        """
        Perform memory cleanup if needed or forced.
        
        Args:
            force: Force cleanup regardless of interval
            
        Returns:
            True if cleanup was performed
        """
        current_time = time.time()
        
        # Check if we need to perform cleanup
        if not force and (current_time - self.last_cleanup_time < self.cleanup_interval):
            return False
        
        # Update memory pressure
        self.update_memory_pressure()
        
        # Decide if cleanup is needed based on pressure and level
        cleanup_needed = force
        
        if self.optimization_level == "aggressive":
            cleanup_needed = cleanup_needed or (self.memory_pressure > 0.7)
        elif self.optimization_level == "normal":
            cleanup_needed = cleanup_needed or (self.memory_pressure > 0.8)
        elif self.optimization_level == "conservative":
            cleanup_needed = cleanup_needed or (self.memory_pressure > 0.9)
        
        if cleanup_needed:
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            # Run garbage collector
            gc.collect()
            
            # Update last cleanup time
            self.last_cleanup_time = current_time
            
            return True
        
        return False
    
    def emergency_cleanup(self):
        """
        Perform aggressive emergency cleanup when memory is critically low.
        """
        # Log current state
        log_memory_usage("EMERGENCY CLEANUP - Before")
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        # Run garbage collector multiple times
        for _ in range(3):
            gc.collect()
        
        # Force another CUDA cache clear
        torch.cuda.empty_cache()
        
        # Log results
        log_memory_usage("EMERGENCY CLEANUP - After")
        
        # Update last cleanup time
        self.last_cleanup_time = time.time()
    
    def optimize_for_inference(self, model):
        """
        Apply memory optimizations to a PyTorch model for inference.
        
        Args:
            model: PyTorch model to optimize
            
        Returns:
            Optimized model
        """
        # Make sure model is in eval mode
        model.eval()
        
        # Use half precision if enabled
        if config.USE_FP16:
            model = model.half()
        
        # Enable CUDA graph capture for faster inference if supported
        # This is an advanced optimization that captures the compute graph
        if hasattr(torch, 'cuda') and hasattr(torch.cuda, 'is_current_stream_capturing'):
            # Only do this if we have the latest PyTorch with CUDA graphs
            try:
                # Only basic models can use this optimization
                simple_model = not hasattr(model, 'generate')
                if simple_model and torch.cuda.is_available() and not torch.cuda.is_current_stream_capturing():
                    torch.cuda.optimize_model(model)
            except (AttributeError, RuntimeError):
                # This is an experimental feature, so ignore if not available
                pass
        
        return model
    
    def release(self):
        """Release all resources used by the memory manager"""
        self.stop_monitoring()
        # Perform final cleanup
        self.cleanup(force=True) 