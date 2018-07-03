import numpy as np
import tensorflow as tf


tfd = tf.contrib.distributions
tfb = tfd.bijectors

def init_param(log_mean, log_std, fix_mean = False):
    '''
    Initialise variational hyperparameters and distribution
    '''
    # Map back lognormal moments to normal moments
    mean = tf.log(tf.square(log_mean) / tf.sqrt(tf.square(log_std) + tf.square(log_mean)))
    std = tf.sqrt(tf.log(1 + tf.square(log_std)/tf.square(log_mean)))
    # Inverse softplus
    std = tf.log(tf.exp(log_std) - 1)
    
    if fix_mean:
        param_mean = mean
    else:
        param_mean = tf.Variable(mean)
    param_std = tf.Variable(std)
    param_distr = tfd.TransformedDistribution(distribution=tfd.MultivariateNormalDiagWithSoftplusScale(
            loc=tf.reshape(param_mean,[-1]), scale_diag=tf.reshape(param_std,[-1])), bijector=tfb.Exp())
    return param_distr, param_mean, param_std

def draw_sample(distr, p, l):
    '''
    Reshape param sample for use in ELBO
    '''
    sample = distr.sample([p, 1])
    return sample, tf.reshape(tf.tile(sample, [1, l]), [-1, 1])




