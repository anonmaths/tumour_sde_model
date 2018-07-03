import tensorflow as tf
import numpy as np

tfd = tf.contrib.distributions
tfb = tfd.bijectors

tf_dtype = tf.float32
np_dtype = np.float32
np_inttype = np.int32

def alpha(x1, x2, pars):
    '''
    returns drift vector for approx p(x)
    '''
    a = tf.concat([pars['c1_strech'] * x1 - pars['c2_strech'] * x1 * x2,
                   pars['c2_strech'] * x1 * x2 - pars['c3_strech'] * x2], 1)
    return a


def beta(x1, x2, pars):
    '''
    returns diffusion matrix for approx p(x)
    '''
    a = tf.expand_dims(tf.sqrt(pars['c1_strech'] * x1 + pars['c2_strech']* x1 * x2), 1)
    b = tf.expand_dims(- pars['c2_strech'] * x1 * x2, 1) / a
    c = tf.sqrt(tf.expand_dims(pars['c3_strech'] * x2 + pars['c2_strech'] * x1 * x2, 1) - tf.square(b))
    zeros = tf.zeros(tf.shape(a))
    beta_chol = tf.concat([tf.concat([a, zeros], 2), tf.concat([b, c], 2)], 1)
    return beta_chol

# ELBO loss function
def ELBO(obs, obs_times v_paths, v_mu, v_sigma, pars, priors, p, dt, T):
    '''
    Calculates the evidence lower bound under the SDE model
    :param obs: observations of SDE
    :param obs_times: observation times of SDE
    :param v_paths: diffusion paths produced by generative VI approx
    :param v_mu: drift vectors produced by generative VI approx
    :param v_sigma: diffusion matrices produced by generative VI approx
    :param pars: current pars of model
    :param priors: the prior distribution moments set over the parameters
    :param p: number of samples used for monte-carlo estimate
    :param dt: discretisation used
    :param T: total timeframe
    '''

    time_index = np_inttype(obs_times / dt)
    

    for i in range(len(time_index)):
        obs_likl = tfd.MultivariateNormalDiag(v_paths[:, :, time_index[i]], scale_identity_multiplier = tf.sqrt([obs['tau']]))
        obs_loglikl = obs_lik.log_prob(tf.tile(tf.expand_dims(obs['obs'][:, i], 0), [p, 1]))
        if i == 0:
            obs_logprob_store = tf.expand_dims(obs_loglik, 1)
        else:
            obs_logprob_store = tf.concat(
                [obs_logprob_store, tf.expand_dims(obs_loglik, 1)], 1)
    obs_logprob = tf.reduce_sum(obs_logprob_store, 1)

    x1_path = v_paths[:, 0, :]
    x2_path = v_paths[:, 1, :]

    x_path_diff = v_paths[:, :, 1:] - v_paths[:, :, :-1]
    x_diff = tf.concat([tf.reshape(x_path_diff[:, 0, :], [-1, 1]),
                        tf.reshape(x_path_diff[:, 1, :], [-1, 1])], 1)

    x_path_mean = tf.concat(
        [tf.reshape(v_paths[:, 0, :-1], [-1, 1]), tf.reshape(v_paths[:, 1, :-1], [-1, 1])], 1)
    x_path_eval = tf.concat(
        [tf.reshape(v_paths[:, 0, 1:], [-1, 1]), tf.reshape(v_paths[:, 1, 1:], [-1, 1])], 1)

    gen_dist = tfd.TransformedDistribution(distribution=tfd.MultivariateNormalTriL(
        loc=x_path_mean + dt * v_mu, scale_tril=tf.sqrt(dt) * v_sigma), bijector=tfb.Softplus(event_ndims=1))

    gen_logprob = gen_dist.log_prob(x_path_eval)

    x1_head = tf.reshape(x1_path[:, :-1], [-1, 1])
    x2_head = tf.reshape(x2_path[:, :-1], [-1, 1])

    alpha_eval = alpha(x1_head, x2_head, pars)
    beta_eval = beta(x1_head, x2_head, pars)

    sde_dist = tfd.MultivariateNormalTriL(
        loc=dt * alpha_eval, scale_tril=tf.sqrt(dt) * beta_eval)
    sde_logprob = sde_dist.log_prob(x_diff)

    theta_cat = tf.log(tf.concat([pars['c1'], pars['c2'], pars['c3']], 1))

    prior_dist = tfd.MultivariateNormalDiag(loc=[priors['c1_mean'], priors['c2_mean'], priors['c3_mean']],
                                            scale_diag=[priors['c1_std'], priors['c2_std'], priors['c3_std']])
    gen_dist = tfd.MultivariateNormalDiag(loc=[pars['c1_mean'], pars['c2_mean'], pars['c3_mean']],
                                          scale_diag=[pars['c1_std'], pars['c2_std'], pars['c3_std']])

    prior_loglik = prior_dist.log_prob(theta_cat)
    gen_loglik = gen_dist.log_prob(theta_cat)

    sum_eval = tf.reduce_sum(tf.reshape(gen_logprob - sde_logprob, [p, -1]), 1)
    loss = sum_eval - obs_logprob + gen_loglik - prior_loglik
    mean_loss = tf.reduce_mean(loss, 0)

    return mean_loss
