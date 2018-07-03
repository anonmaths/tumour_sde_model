import pandas as pd
import numpy as np

def extract_features(obs, obs_times, dt):
    l = int(obs_times[-1]/dt)
    next_obs = np.empty((obs.shape[0],1))
    prev_obs = np.empty((obs.shape[0],1))
    next_time = np.array([])
    time_diff = obs_times[1:] - obs_times[:-1]
    for i in range(len(time_diff)):
        n = int(time_diff[i] / dt)
        next_obs = np.append(next_obs, np.tile(
            np.reshape(obs[:,i+1],(-1,1)), (1,n)), axis = 1)
        prev_obs = np.append(prev_obs, np.tile(
            np.reshape(obs[:,i],(-1,1)), (1,n)), axis = 1)
        next_time = np.append(next_time, np.linspace(
            time_diff[i], dt, n))
    next_obs = next_obs[:,1:]
    prev_obs = prev_obs[:,1:]
    return next_time, next_obs, prev_obs