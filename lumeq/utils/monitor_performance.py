import os, sys, time, psutil
import functools
import logging

# optional GPU support (PyTorch)
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False

# --- global debug performance flag ---
# toggle debugging mode via environment variable
#_DEBUG_MODE = os.getenv("DEBUG_PERFORMANCE", "False").lower() == "true"
# it can also be set via set_performance_log() function below
_DEBUG_PERFORMANCE = False

# --- global logger setup ---
_logger_performance = logging.getLogger("performance_monitor")
_logger_performance.setLevel(logging.INFO)

_default_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
# default handler: stdout
_default_handler = logging.StreamHandler(sys.stdout)
_default_handler.setFormatter(_default_formatter)
_logger_performance.addHandler(_default_handler)

### do not change _DEBUG_PERFORMANCE and _logger_performance outside of this module ###


def set_performance_log(debug=True, filename=None):
    """
    turn on/off performance monitoring and optionally
    change the output destination for performance logs.
    - debug = True/False: enable/disable performance monitoring
    - filename = none: set log to stdout (default)
    """
    global _DEBUG_PERFORMANCE ### dangerous: modifying global flag
    _DEBUG_PERFORMANCE = debug

    if filename is None:
        return # keep default stdout handler

    global _logger_performance ### dangerous: modifying global logger
    _logger_performance.handlers.clear() # remove existing handlers
    handler = logging.FileHandler(filename)
    handler.setFormatter(_default_formatter)
    _logger_performance.addHandler(handler)


def monitor_performance(func):
    """
    decorator for monitoring memory of CPU (GPU) and timing of wall, CPU (GPU).
    use only as @monitor_performance above function definitions.
    """
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        if not _DEBUG_PERFORMANCE:
            return func(*args, **kwargs)  # bypass monitoring when disabled

        process = psutil.Process(os.getpid())

        # --- Start metrics ---
        mem_before = process.memory_info().rss / 1024 / 1024 # in MB
        wall_start = time.time()
        cpu_start = time.process_time()

        if GPU_AVAILABLE:
            torch.cuda.synchronize()
            gpu_mem_before = torch.cuda.memory_allocated() / 1024 / 1024
            gpu_time_start = time.perf_counter()
        else:
            gpu_mem_before = gpu_time_start = 0

        # --- execute target function ---
        result = func(*args, **kwargs)

        # --- end metrics ---
        mem_after = process.memory_info().rss / 1024 / 1024 # in MB
        wall_end = time.time()
        cpu_end = time.process_time()

        if GPU_AVAILABLE:
            torch.cuda.synchronize()
            gpu_mem_after = torch.cuda.memory_allocated() / 1024 / 1024
            gpu_time_end = time.perf_counter()
        else:
            gpu_mem_after = gpu_time_end = 0

        # --- logging results ---
        msg = [f"Performance report for function: {func.__name__}()",
               f"    Wall time: %.4f s  CPU time: %.4f s  GPU time: %.4f s" % (wall_end-wall_start, cpu_end-cpu_start, gpu_time_end-gpu_time_start),
               f"    RAM memory usage: %.2f MB GPU memory usage: %.2f\n" % (mem_after, gpu_mem_after)]

        _logger_performance.info('\n' + '\n'.join(msg))
        return result

    return decorator


# ---------------------------------------------------------
# exported public interface only
# ---------------------------------------------------------
__all__ = ['monitor_performance', 'set_performance_log']
