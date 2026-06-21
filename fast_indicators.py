import ctypes
import os
import numpy as np
import pandas as pd

import subprocess

# Determine OS and library path
base_dir = os.path.dirname(__file__)
is_windows = os.name == 'nt'
lib_name = 'fast_math.dll' if is_windows else 'fast_math.so'
lib_path = os.path.join(base_dir, lib_name)

# Auto-compile for Linux (Streamlit Cloud) if it doesn't exist
if not is_windows and not os.path.exists(lib_path):
    print("Linux environment detected. Auto-compiling C++ engine...")
    cpp_file = os.path.join(base_dir, 'fast_math.cpp')
    compile_cmd = f"g++ -O3 -shared -fPIC -o {lib_path} {cpp_file}"
    try:
        subprocess.run(compile_cmd, shell=True, check=True)
        print("C++ engine compiled successfully!")
    except Exception as e:
        print(f"Failed to auto-compile C++ engine: {e}")

try:
    fast_math = ctypes.CDLL(lib_path)
    
    # Define argument types for the C++ function
    fast_math.calculate_ema_cpp.argtypes = [
        ctypes.POINTER(ctypes.c_double), 
        ctypes.POINTER(ctypes.c_double), 
        ctypes.c_int, 
        ctypes.c_int
    ]
    fast_math.calculate_ema_cpp.restype = None
    C_EXT_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not load {lib_name}. Falling back to Pandas. Error: {e}")
    C_EXT_AVAILABLE = False


def get_fast_ema(series, window=20):
    """
    Calculates Exponential Moving Average using ultra-fast C++ extension.
    Falls back to pandas if C++ library is not available.
    """
    if not C_EXT_AVAILABLE:
        return series.ewm(span=window, adjust=False).mean()
        
    prices = series.to_numpy(dtype=np.float64)
    length = len(prices)
    
    # Prepare output array
    result = np.zeros(length, dtype=np.float64)
    
    # Cast numpy arrays to ctypes pointers
    prices_ptr = prices.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    # Call C++ function
    fast_math.calculate_ema_cpp(prices_ptr, result_ptr, length, window)
    
    return pd.Series(result, index=series.index)
