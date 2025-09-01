"""Inspired by https://github.com/huggingface/accelerate/blob/main/examples/by_feature/fsdp_with_peak_mem_tracking.py

but not a context manager
"""
import gc
import os
import threading
import torch
import psutil
from accelerate.utils import is_npu_available, is_xpu_available

# New Code #
# Converting Bytes to Megabytes
def b2mb(x):
    return int(x / 2**20)

class TorchTracemalloc:
    """Lightweight memory tracker that can be started/stopped multiple times."""
    
    def __init__(self, do_tracking: bool=True, device='cpu'):
        if not do_tracking:
            self.do_tracking = False
        else:
            self.do_tracking = True
            self.process = psutil.Process()
            self.is_tracking = False
            self.peak_monitoring = False
            self.monitor_thread = None
            self.device = device

    def start(self):
        """Start memory tracking."""
        if not self.do_tracking:
            return
        if self.is_tracking:
            return  # Already tracking
            
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
            self.gpu_begin = torch.cuda.memory_allocated()
        elif is_xpu_available():
            torch.xpu.empty_cache()
            torch.xpu.reset_max_memory_allocated()
            self.gpu_begin = torch.xpu.memory_allocated()
        elif is_npu_available():
            torch.npu.empty_cache()
            torch.npu.reset_max_memory_allocated()
            self.gpu_begin = torch.npu.memory_allocated()
        else:
            self.gpu_begin = 0
            
        self.cpu_begin = self.process.memory_info().rss
        self.cpu_peak = self.cpu_begin
        self.is_tracking = True
        self.peak_monitoring = True
        
        # Start peak monitoring thread
        self.monitor_thread = threading.Thread(target=self._peak_monitor)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop(self):
        """Stop tracking and return stats."""
        if not self.do_tracking:
            return None
        if not self.is_tracking:
            return {}
            
        # Stop peak monitoring
        self.peak_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gpu_end = torch.cuda.memory_allocated()
            gpu_peak = torch.cuda.max_memory_allocated()
        else:
            gpu_end = 0
            gpu_peak = 0
            
        cpu_end = self.process.memory_info().rss
        
        stats = {
            'gpu_used_mb': torch.tensor([b2mb(gpu_end - self.gpu_begin)], device=self.device),
            'gpu_peak_mb': torch.tensor([b2mb(gpu_peak - self.gpu_begin)], device=self.device),
            'gpu_total_peak_mb': torch.tensor([b2mb(gpu_peak)], device=self.device),
            'cpu_used_mb': torch.tensor([b2mb(cpu_end - self.cpu_begin)], device=self.device),
            'cpu_peak_mb': torch.tensor([b2mb(self.cpu_peak - self.cpu_begin)], device=self.device),
            'cpu_total_peak_mb': torch.tensor([b2mb(self.cpu_peak)], device=self.device),
        }
        
        self.is_tracking = False
        return stats
        
    def _peak_monitor(self):
        """Monitor peak CPU usage in background thread."""
        while self.peak_monitoring:
            current = self.process.memory_info().rss
            self.cpu_peak = max(current, self.cpu_peak)