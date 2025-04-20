import os
import numpy as np
import random
import pickle
from src import utils
from typing import List, Tuple
from sklearn.decomposition import PCA

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_same_len(data, max_len):
    """
    Interpolate multidimensional time series data to a specified length.

    Parameters:
    data (np.array): Input data with shape (N, D, L)
    max_len (int): Desired length of the time series

    Returns:
    np.array: Interpolated data with shape (N, D, max_len)
    """
    N, D, L = data.shape
    
    # Create a new array to store the interpolated data
    data_interpolated = np.zeros((N, D, max_len))
    
    # Create the new time points
    old_time = np.linspace(0, 1, num=L)
    new_time = np.linspace(0, 1, num=max_len)
    
    # Interpolate each dimension of each sample
    for n in range(N):
        for d in range(D):
            data_interpolated[n, d, :] = np.interp(new_time, old_time, data[n, d, :])
    
    return data_interpolated


for data_name in ['ECG', 'EMG', 'Epilepsy', 'FD-B', 'Gesture', 'SleepEEG']:
    context_len = 256
    horizon_len = 0
    output_filename = f'preprocessed_data/_DA_{data_name}_{context_len}_{horizon_len:02d}.pkl'
    
    # Check if output file already exists
    if os.path.exists(output_filename):
        print(f"Skipping {output_filename}: Output file already exists.")
        continue
    
    with open(f'data/Domain_ts/{data_name}.pkl', 'rb') as f:
        X_train, X_val, X_test, y_train_np, y_val_np, y_test_np = pickle.load(f)
    
    # N, D, L = X_train.shape
    # context_len = min(context_len, (L // 32 + 1) * 32)
    # print(data_name, L, context_len, horizon_len)

    X_train_intp = get_same_len(X_train, context_len)
    X_val_intp = get_same_len(X_val, context_len)
    X_test_intp = get_same_len(X_test, context_len)

    # Save processed data
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)  # Ensure directory exists
    with open(output_filename, 'wb') as f:
        pickle.dump([X_train_intp, None, None, y_train_np, 
                     X_val_intp, None, None, y_val_np, 
                     X_test_intp, None, None, y_test_np], f)
    
    print(f"Processed data saved to {output_filename}")
