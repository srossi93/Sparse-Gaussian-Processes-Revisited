# Credit to GPFlow.

import tensorflow as tf
import numpy as np
from .quadrature import ndiagquad

class Gaussian(object):
    def logdensity(self, x, mu, var):
        return -0.5 * (np.log(2 * np.pi) + tf.math.log(var) + tf.square(mu-x) / var)

    def __init__(self, variance=1.0, **kwargs):
        self.variance = tf.exp(tf.Variable(np.log(variance), dtype=tf.float64, name='lik_log_variance'))

    def logp(self, F, Y):
        return self.logdensity(Y, F, self.variance)

    def conditional_mean(self, F):
        return tf.identity(F)

    def conditional_variance(self, F):
        return tf.fill(tf.shape(F), tf.squeeze(self.variance))

    def predict_mean_and_var(self, Fmu, Fvar):
        return tf.identity(Fmu), Fvar + self.variance

    def predict_density(self, Fmu, Fvar, Y):
        return self.logdensity(Y, Fmu, Fvar + self.variance)

    def variational_expectations(self, Fmu, Fvar, Y):
        return -0.5 * np.log(2 * np.pi) - 0.5 * tf.log(self.variance) \
               - 0.5 * (tf.square(Y - Fmu) + Fvar) / self.variance


def inv_probit(x):
    jitter = 1e-3  # ensures output is strictly between 0 and 1
    return 0.5 * (1.0 + tf.math.erf(x / np.sqrt(2.0))) * (1 - 2 * jitter) + jitter


class Bernoulli(object):
    def __init__(self, invlink=inv_probit, **kwargs):
        self.invlink = invlink
        self.num_gauss_hermite_points = 20

    def logdensity(self, x, p):
        return tf.math.log(tf.where(tf.equal(x, 1), p, 1-p))

    def logp(self, F, Y):
        return self.logdensity(Y, self.invlink(F))

    def conditional_mean(self, F):
        return self.invlink(F)

    def conditional_variance(self, F):
        p = self.conditional_mean(F)
        return p - tf.square(p)

    def predict_density(self, Fmu, Fvar, Y):
        p = self.predict_mean_and_var(Fmu, Fvar)[0]
        return self.logdensity(Y, p)

    def predict_mean_and_var(self, Fmu, Fvar):
        if self.invlink is inv_probit:
            p = inv_probit(Fmu / tf.sqrt(1 + Fvar))
            return p, p - tf.square(p)
        else:
            # for other invlink, use quadrature
            integrand2 = lambda *X: self.conditional_variance(*X) + tf.square(self.conditional_mean(*X))
            E_y, E_y2 = ndiagquad([self.conditional_mean, integrand2],
                                  self.num_gauss_hermite_points,
                                  Fmu, Fvar)
            V_y = E_y2 - tf.square(E_y)
            return E_y, V_y

    def variational_expectations(self, Fmu, Fvar, Y):
        r"""
        Compute the expected log density of the datasets, given a Gaussian
        distribution for the function values.

        if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes

           \int (\log p(y|f)) q(f) df.
        """
        return ndiagquad(self.logp, self.num_gauss_hermite_points, Fmu, Fvar, Y=Y)

