import tensorflow as tf
import numpy as np
from scipy.cluster.vq import kmeans2

from .base_model import BaseModel
from . import conditionals
from .utils import get_rand

def set_seed(seed=0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)


class Strauss(object):

    def __init__(self, gamma=0.5, R=0.5):
        self.gamma = tf.constant(gamma, dtype=tf.float64)
        self.R = tf.constant(R, dtype=tf.float64)

    def _euclid_dist(self, X):
        Xs = tf.reduce_sum(tf.square(X), axis=-1, keepdims=True)
        dist = -2 * tf.matmul(X, X, transpose_b=True)
        dist += Xs + tf.matrix_transpose(Xs)
        return tf.sqrt(tf.maximum(dist, 1e-40))

    def _get_Sr(self, X):
        """
        Get the # elements in distance matrix dist that are < R
        """
        dist = self._euclid_dist(X)
        val = tf.where(dist <= self.R)
        Sr = tf.shape(val)[0] # number of points satisfying the constraint above
        dim = tf.shape(dist)[0]
        Sr = (Sr - dim)/2  # discounting diagonal and double counts
        return Sr

    def logp(self, X):
        return self._get_Sr(X) * tf.math.log(self.gamma)


class Layer(object):
    def __init__(self, kern, outputs, n_inducing, fixed_mean, X, full_cov, prior_type="uniform"): # uniform / normal / point
        self.inputs, self.outputs, self.kernel = kern.input_dim, outputs, kern
        self.M, self.fixed_mean = n_inducing, fixed_mean
        self.full_cov = full_cov
        self.prior_type = prior_type
        self.X = X
        if prior_type == "strauss":
            self.pZ = Strauss(R=0.5)

        if len(X) > 1000000:
            perm = np.random.permutation(100000)
            X = X[perm]
        self.Z = tf.Variable(kmeans2(X, self.M, minit='points')[0], dtype=tf.float64, trainable=False, name='Z')
        if self.inputs == outputs:
            self.mean = np.eye(self.inputs)
        elif self.inputs < self.outputs:
            self.mean = np.concatenate([np.eye(self.inputs), np.zeros((self.inputs, self.outputs - self.inputs))], axis=1)
        else:
            _, _, V = np.linalg.svd(X, full_matrices=False)
            self.mean = V[:self.outputs, :].T

        self.U = tf.Variable(np.zeros((self.M, self.outputs)), dtype=tf.float64, trainable=False, name='U')
        # self.U = tf.Variable(np.random.randn(self.M, self.outputs), dtype=tf.float64, trainable=False, name='U')
        self.Lm = None


    def conditional(self, X):

        mean, var, self.Lm = conditionals.conditional(X, self.Z, self.kernel, self.U, white=True, full_cov=self.full_cov, return_Lm=True)
        
        if self.fixed_mean:
            mean += tf.matmul(X, tf.cast(self.mean, tf.float64))
        return mean, var

    def prior_Z(self):
        if self.prior_type == "uniform":
            return 0.
        if self.prior_type == "normal":
            return -tf.reduce_sum(tf.square(self.Z)) / 2.0
            

        if self.prior_type == "strauss":
            return self.pZ.logp(self.Z)

        #if self.Lm is not None: # determinantal;
        if self.prior_type == "determinantal":
            self.Lm = tf.linalg.cholesky(self.kernel.K(self.Z) + tf.eye(self.M, dtype=tf.float64) * 1e-7)
            pZ = tf.reduce_sum(tf.math.log(tf.square(tf.linalg.diag_part(self.Lm))))
            return pZ

        else: #
            raise Exception("Invalid prior type")

    def prior_hyper(self):
        return -tf.reduce_sum(tf.square(self.kernel.loglengthscales)) / 2.0 - tf.reduce_sum(tf.square(self.kernel.logvariance - np.log(0.05))) / 2.0

    def prior(self):
        return -tf.reduce_sum(tf.square(self.U)) / 2.0 + self.prior_hyper()  + self.prior_Z()

    def __str__(self):
        str = [
            '============ GP Layer ',
            ' Input dim = %d' % self.inputs,
            ' Output dim = %d' % self.outputs,
            ' Num inducing = %d' % self.M,
            ' Prior on inducing positions = %s' % self.prior_type,
            '\n'.join(map(lambda s: ' |' + s, self.kernel.__str__().split('\n')))
        ]

        return '\n'.join(str)


class DGP(BaseModel):
    def propagate(self, X):
        Fs = [X, ]
        Fmeans, Fvars = [], []

        for l, layer in enumerate(self.layers):
            mean, var = layer.conditional(Fs[-1])
            # eps = tf.random_normal(tf.shape(mean), dtype=tf.float64)
            # F = mean + eps * tf.sqrt(var)
            if l+1 < len(self.layers):
                F = self.rand([mean, var])
            else:
                F = get_rand([mean, var], False)
                
            Fs.append(F)
            Fmeans.append(mean)
            Fvars.append(var)

        return Fs[1:], Fmeans, Fvars

    def reset_Lm(self):
        for layer in self.layers:
            layer.Lm = None

    def __init__(self, X, Y, n_inducing, kernels, likelihood, minibatch_size, window_size, output_dim=None,
                 adam_lr=0.01, prior_type="uniform", full_cov=False, epsilon=0.01, mdecay=0.05,):
        self.n_inducing = n_inducing
        self.kernels = kernels
        self.likelihood = likelihood
        self.minibatch_size = minibatch_size
        self.window_size = window_size

        self.rand = lambda x: get_rand(x, full_cov)
        self.output_dim = output_dim or Y.shape[1]

        n_layers = len(kernels)
        N = X.shape[0]

        self.layers = []
        X_running = X.copy()
        for l in range(n_layers):
            outputs = self.kernels[l+1].input_dim if l+1 < n_layers else self.output_dim#Y.shape[1]
            self.layers.append(Layer(self.kernels[l], outputs, n_inducing, fixed_mean=(l+1 < n_layers), X=X_running, full_cov=full_cov if l+1<n_layers else False, prior_type=prior_type))
            X_running = np.matmul(X_running, self.layers[-1].mean)

        variables = []
        for l in self.layers:
            variables += [l.U, l.Z, l.kernel.loglengthscales, l.kernel.logvariance]

        super().__init__(X, Y, variables, minibatch_size, window_size)
        self.f, self.fmeans, self.fvars = self.propagate(self.X_placeholder)
        self.y_mean, self.y_var = self.likelihood.predict_mean_and_var(self.fmeans[-1], self.fvars[-1])

        self.prior = tf.add_n([l.prior() for l in self.layers])
        self.log_likelihood = self.likelihood.predict_density(self.fmeans[-1], self.fvars[-1], self.Y_placeholder)

        # self.varexps = self.likelihood.variational_expectations(self.fmeans[-1], self.fvars[-1], self.Y_placeholder)
        # self.nll = - tf.reduce_sum(self.varexps) / tf.cast(tf.shape(self.X_placeholder)[0], tf.float64) \
        #            - (self.prior / N)

        self.nll = - tf.reduce_sum(self.log_likelihood) / tf.cast(tf.shape(self.X_placeholder)[0], tf.float64) \
                       - (self.prior / N)

        self.generate_update_step(self.nll, epsilon, mdecay)
        self.adam = tf.compat.v1.train.AdamOptimizer(adam_lr)
        try:
            self.hyper_train_op = self.adam.minimize(self.nll)
        except ValueError:
            pass

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.compat.v1.Session(config=config)
        set_seed()
        init_op = tf.compat.v1.global_variables_initializer()
        try:
            self.session.run(init_op, feed_dict=self.likelihood.initializable_feeds )
        except AttributeError:
            self.session.run(init_op)


    def predict_y(self, X, S, posterior=True):
        # assert S <= len(self.posterior_samples)
        ms, vs = [], []
        for i in range(S):
            feed_dict = {self.X_placeholder: X}
            feed_dict.update(self.posterior_samples[i]) if posterior else feed_dict.update(self.window[-(i+1)])
            m, v = self.session.run((self.y_mean, self.y_var), feed_dict=feed_dict)
            ms.append(m)
            vs.append(v)
        return np.stack(ms, 0), np.stack(vs, 0)

    def predict_f_samples(self, X, S):
        assert S <= len(self.posterior_samples)
        fs = []
        for i in range(S):
            feed_dict = {self.X_placeholder: X}
            feed_dict.update(self.posterior_samples[i])
            f = self.session.run(self.f[-1], feed_dict=feed_dict)
            fs.append(f)
        return np.stack(fs, 0)

    def __str__(self):
        str = [
            '================= DGP',
            ' Input dim = %d' % self.layers[0].inputs,
            ' Output dim = %d' % self.layers[-1].outputs,
            ' Depth = %d' % len(self.layers)
        ]
        return '\n'.join(str + ['\n'.join(map(lambda s: ' |' + s, l.__str__().split('\n'))) for l in self.layers])