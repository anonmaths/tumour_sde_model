import tensorflow as tf
import numpy as np

from network_utils import Weight, Bias

tfd = tf.contrib.distributions
tfb = tfd.bijectors

tf_dtype = tf.float32
np_dtype = np.float32

class Model():
    def __init__(self, obs, features, priors, sigma, dt, T, p, pars, network_pars):
        weights = {}
        
        # Initialise RNN layers
        with tf.variable_scope('input_layer'):
            weights['w0'] = Weight(
                [1, network_pars['n_inputs'], network_pars['hidden_layer_width']], tf_dtype).tile(p)
            weights['b0'] = Bias([1, 1, network_pars['hidden_layer_width']], tf_dtype).tile(p)
        
        for i in range(1, network_pars['n_hidden_layers'] + 1):
            with tf.variable_scope('hidden_layer_%d' % i):
                weights['w%i' % i] = Weight(
                    [1, network_pars['hidden_layer_width'], network_pars['hidden_layer_width']], tf_dtype).tile(p)
                weights['b%i' % i] = Bias(
                    [1, 1, network_pars['hidden_layer_width']], tf_dtype).tile(p)

        with tf.variable_scope('output_layer'):
            weights['wn'] = Weight(
                [1, network_pars['hidden_layer_width'], network_pars['n_outputs']], tf_dtype).tile(p)
            weights['bn'] = Bias([1, 1, network_pars['n_outputs']], tf_dtype).tile(p)
    
        self.sigma = sigma
        self.weights = weights
        self.n = obs.shape[0]
        self.network_pars = network_pars
        self.obs = obs
        self.pars = pars
        self.priors = priors
        self.features = features
        self.p = p
        self.dt = dt
        self.T = dt

        # Build the TensorFlow graph
        self._build()

    def _build(self):
        '''
        Build model graph
        '''
        print("Building graph...")
        
        # Build diffusion bridge functions using RNN for all p simulations at once
        with tf.name_scope('diffusion_bridge'):
            v_paths, v_mu, v_sigma = self._diff_bridge()

        with tf.name_scope('ELBO'):
            loss_estimate = ELBO(
                self.obs, v_paths, v_mu, v_sigma, self.pars, self.priors, self.p, self.dt, self.T)
            tf.summary.scalar('mean_loss', mean_loss)

        # specifying optimizer and gradient clipping for backprop
        with tf.name_scope('optimize'):
            opt = tf.train.AdamOptimizer(1e-3)
            gradients, variables = zip(
                *opt.compute_gradients(mean_loss))
            global_norm = tf.global_norm(gradients)
            gradients, _ = tf.clip_by_global_norm(gradients, 4e3)
            self.train_step = opt.apply_gradients(
                zip(gradients, variables))
            tf.summary.scalar('global_grad_norm', global_norm)

        # mean-field approx pars to tensorboard
        with tf.name_scope('variables'):
            with tf.name_scope('theta1'):
                tf.summary.scalar('theta1_mean', c1_mean)
                tf.summary.scalar('theta1_std', c1_std)
            with tf.name_scope('theta2'):
                tf.summary.scalar('theta2_mean', c2_mean)
                tf.summary.scalar('theta2_std', c2_std)
            with tf.name_scope('theta3'):
                tf.summary.scalar('theta3_mean', c3_mean)
                tf.summary.scalar('theta3_std', c3_std)

        self.merged = tf.summary.merge_all()

    def _diff_bridge(self):
        '''
        Roll out rnn cell across the time series
        '''
        x01 = tf.expand_dims(self.params['x0'].sample(p), axis = 2)
        zeros = tf.zeros_like(x01)
        x0 =  tf.concat((x01, zeros), axis = 2)
        t1 = tf.expand_dims(self.features['next_times'][...,0], axis = 2) 
        y1 = tf.expand_dims(self.features['next_obs'][...,0], axis = 2)
        
        input_features = tf.concat([x0, zeros, t1, y1, y1 - x0], axis = 2)
        
        v_mu, v_sigma = self._rnn_cell(input_features)
        v_mu_store = v_mu
        v_sigma_store = tf.reshape(v_sigma, [self.p, self.n, 1, 4])
        
        v_path = self._path_sampler(tf.expand_dims(x0, axis = 3), v_mu, v_sigma)
        v_path_store = tf.concat((x0, v_path), axis = 3)
        
        for i in range(1, int(self.T / self.dt)):
            xt = tf.squeeze(v_path)
            y_next = tf.expand_dims(self.features['next_obs'][...,i], axis = 2)
            t = self.dt + input_features[...,2]
            t_next = tf.expand_dims(self.features['next_times'][...,i], axis = 2)
            y_curr_exp = tf.expand_dims(tf.reduce_sum(xt, axis = 2), axis = 2)
            
            input_features = tf.concat([xt, t, t_next, y_next, y_next - y_curr_exp], axis = 2)
            
            v_mu, v_sigma = self._rnn_cell(input_features)
            v_mu_store = tf.concat((v_mu_store, v_mu), axis = 3)
            v_sigma_store = tf.concat((v_sigma_store, tf.reshape(v_sigma, [self.p, self.n, 1, 4])), axis = 2)
            
            v_path = self._path_sampler(tf.expand_dims(xt, axis = 3), v_mu, v_sigma)
            v_path_store = tf.concat((v_path_store, v_path), axis = 3)
            
        return v_path_store, v_mu_store, v_sigma_store

    # the rnn cell called by diff_bridge
    def _rnn_cell(self, features_input, eps_identity=1e-3):
        '''
        rnn cell for supplying Gaussian state transitions
        :param eps: eps * identity added to diffusion matrix to control numerical stability
        '''
        hidden_layer = tf.nn.relu(
            tf.add(tf.matmul(features_input, self.weights['w0']), self.weights['b0']))

        for i in range(1, self.network_pars['n_hidden_layers'] + 1):
            hidden_layer = tf.nn.relu(
                tf.add(tf.matmul(hidden_layer, self.weights['w%i' % i]), self.weights['b%i' % i]))

        output = tf.add(
            tf.matmul(hidden_layer, self.weights['wn']), self.weights['bn'])

        v_mu, v_sigma_11, v_sigma_21, v_sigma_22 = tf.split(output, [2, 1, 1, 1], 2)
        zeros = tf.zeros((self.p,self.n,1,1), dtype = tf_dtype)
        v_sigma_chol = tf.concat((tf.concat((tf.expand_dims(tf.nn.softplus(v_sigma_11), axis = 3), zeros), axis = 3),
                                tf.concat((tf.expand_dims(v_sigma_21, axis = 3),
                                    tf.expand_dims(tf.nn.softplus(v_sigma_22), axis = 3)), axis = 3)), axis = 2)
        v_sigma_chol = tf.cholesky(tf.matmul(v_sigma_chol, v_sigma_chol, transpose_b = True) +
            eps_identity * tf.tile(tf.reshape(tf.eye(2), [1,1,2,2]),[self.p, self.n, 1, 1]))
        return tf.reshape(v_mu, [self.p, self.n, 2, 1]), v_sigma_chol

    # functions to return p simulations of a diffusion bridge
    def _path_sampler(self, xt, v_mu, v_sigma):
        '''
        sample new state using learned Gaussian state transitions
        :param inp: current state of system
        :param mu_nn: drift vector from RNN
        :param sigma_nn: diffusion matrix from RNN as cholesky factor
        '''
        xtplus_dist = tfd.TransformedDistribution(distribution=tfd.MultivariateNormalTriL(
            loc = xt + self.dt * v_mu, scale_tril = tf.sqrt(self.dt) * v_sigma),
            bijector=tfb.Softplus(event_ndims = 2))
        
        xtplus = tf.reshape(xtplus_dist.sample(), [self.p, self.n, 2, 1])
        return xtplus

    # train the model
    def train(self, niter, path):
        '''
        trains model
        :pars niter: number of iterations
        :pars PATH: path to tensorboard output
        '''
        print("Training model...")
        writer = tf.summary.FileWriter(
            '%s/%s' % (path, datetime.now().strftime("%d:%m:%y-%H:%M:%S")), sess.graph)
        for i in range(niter):
            self.train_step.run()
            if i % 10 == 0:
                summary = sess.run(self.merged)
                writer.add_summary(summary, i)

    def save(self, path):
        '''
        save model
        '''
        saver = tf.train.Saver()
        saver.save(sess, path)
        print("Model Saved")

    def load(self, path):
        '''
        load model
        '''
        saver = tf.train.Saver()
        saver.restore(sess, path)
        print("Model Restored")
