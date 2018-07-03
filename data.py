import pandas as pd
import numpy as np

np_dtype = np.float32

def load_data(filename, n_groups, t0 = 1):
    data = pd.read_csv('dataset1.csv',header=None).values
    # impute negative values using softplus
    # data[data < 0] = np.log(1 + np.exp(data[data < 0]))
    n_pop = data.shape[1]
    n_obs = data.shape[0] - t0
    y0 = data[t0-1]
    obs = np.transpose(data[t0:])
    obs_times = np.arange(n_obs, dtype = np_dtype)    
    return obs_times, obs, y0
