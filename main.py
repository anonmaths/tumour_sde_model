import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import scipy.stats as stats
from datetime import datetime


from lotka_volterra_data_augmentation import *
from lotka_volterra_loss import ELBO
from network_utils import Weight, Bias

tfd = tf.contrib.distributions
tfb = tfd.bijectors

tf_dtype = tf.float32
np_dtype = np.float32


if __name__ == "__main__":
    with tf.Session() as sess:
        lotka_volterra = Model(network_params=NETWORK_PARAMS, p=P,
                               dt=DT, obs=obs, params=params, priors=PRIORS, features=features)
        sess.run(tf.global_variables_initializer())
        # desired number of iterations. currently no implementation of a
        # convergence criteria.
        lotka_volterra.train(25000, PATH_TO_TENSORBOARD_OUTPUT)
