# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

from datetime import datetime
from time import time
from scipy.linalg.special_matrices import toeplitz
import numpy as np


def features_normal_cov_toeplitz(n_samples, n_features, rho=0.5):
    """Features obtained as samples of a centered Gaussian vector
    with a toeplitz covariance matrix

    Parameters
    ----------
    rho: float
        correlation coefficient of the toeplitz correlation matrix

    Returns
    -------
    output: ndarray, shape=[n_samples, n_features]
    """
    cov = toeplitz(rho ** np.arange(0, n_features))
    return np.random.multivariate_normal(
        np.zeros(n_features), cov, size=n_samples)


def simulation_method(simulate_method):
    """A decorator for simulation methods.
    It simply calls _start_simulation and _end_simulation methods
    """

    def decorated_simulate_method(self):
        self._start_simulation()
        result = simulate_method(self)
        self._end_simulation()
        self.data = result
        return result

    return decorated_simulate_method


class Simulation:
    """This is an abstract simulation class that inherits form BaseClass
    It does nothing besides printing stuff and verbosing
    """

    def __init__(self, seed=None, verbose=True):
        # Set default parameters
        self.seed = seed
        self.verbose = verbose
        if self.seed is not None:
            self._set_seed()
        # No data simulated yet
        self.features = None
        self.labels = None

    def _set_seed(self):
        np.random.seed(self.seed)
        return self

    @staticmethod
    def _get_now():
        return str(datetime.now()).replace(" ", "_").replace(":", "-")

    def _start_simulation(self):
        self.time_start = Simulation._get_now()
        self._numeric_time_start = time()
        if self.verbose:
            msg = "Launching simulation using {class_}..." \
                .format(class_=self.__class__.__name__)
            print("-" * len(msg))
            print(msg)

    def _end_simulation(self):
        self.time_end = self._get_now()
        t = time()
        self.time_elapsed = t - self._numeric_time_start
        if self.verbose:
            msg = "Done simulating using {class_} in {time:.2e} seconds." \
                .format(class_=self.__class__.__name__,
                        time=self.time_elapsed)
            print(msg)


class CensoredGeomMixtureRegression(Simulation):
    """Class for the simulation of Censored Geometric Mixture Model

    Parameters
    ----------
    verbose: boolean
        Verbose mode to detail or not ongoing tasks
    n_samples: int
        Number of patients
    n_features: int
        Number of features
    nb_active_features: int
        Number of active features
    K: float
        Value of the active coefficients
    rho: float
        Coefficient of the Toeplitz correlation matrix
    pi_0: float
        Proportion of desired low risk patients rate
    gap: float
        Gap value to create high/low risk groups
    r_c: float
        Censoring rate
    r_cf: float
        Confusion factors rate
    p0: float
        Geometric parameter for low risk patients
    p1: float
        Geometric parameter for high risk patients
    seed: int
        For reproducible simulation

    Attributes
    ----------
    Y: ndarray
        Temporal data
    Z: ndarray
        Latent variable
    delta: ndarray
        Censoring indicator
    """

    def __init__(self, verbose, n_samples, n_features, nb_active_features,
                 K, rho, pi_0, gap, r_c, r_cf, p0, p1, seed=None):
        Simulation.__init__(self, seed=seed, verbose=verbose)
        self.n_samples = n_samples
        self.n_features = n_features
        self.nb_active_features = nb_active_features
        self.K = K
        self.rho = rho
        self.pi_0 = pi_0
        self.gap = gap
        self.r_c = r_c
        self.r_cf = r_cf
        self.p0 = p0
        self.p1 = p1

        # Attributes that will be instantiated afterwards
        self.Z = None
        self.Y = None
        self.delta = None

    @staticmethod
    def logistic_grad(z):
        """Overflow proof computation of 1 / (1 + exp(-z)))
        """
        idx_pos = np.where(z >= 0.)
        idx_neg = np.where(z < 0.)
        res = np.empty(z.shape)
        res[idx_pos] = 1. / (1. + np.exp(-z[idx_pos]))
        res[idx_neg] = 1 - 1. / (1. + np.exp(z[idx_neg]))
        return res

    @staticmethod
    def poldeg2_solver(a=0, b=0, c=0):
        """2nd order polynomial solver
        """
        if a == 0:
            if b == 0:
                return None
            return -c / b
        delta = b * b - 4 * a * c
        if delta < 0:
            return None
        if delta == 0:
            return -b / (2 * a)
        if delta > 0:
            sqrt_delta = np.sqrt(delta)
            return [(-b - sqrt_delta) / (2 * a), (-b + sqrt_delta) / (2 * a)]

    @simulation_method
    def simulate(self):
        """Launch simulation of the data.
        """
        n_samples = self.n_samples
        n_features = self.n_features
        nb_active_features = self.nb_active_features
        K = self.K
        pi_0 = self.pi_0
        gap = self.gap
        p0 = self.p0
        p1 = self.p1
        r_c = self.r_c
        r_cf = self.r_cf
        rho = self.rho

        coeffs = np.zeros(n_features)
        coeffs[0:nb_active_features] = K

        features = features_normal_cov_toeplitz(n_samples, n_features, rho)

        # Add class relative information on the design matrix    
        A = np.random.choice(range(n_samples), size=int((1 - pi_0) * n_samples),
                             replace=False)
        A_ = np.delete(range(n_samples), A)

        index_plus_gap = nb_active_features + int(
            (n_features - nb_active_features) * r_cf)
        features[A, :index_plus_gap] += gap
        features[A_, :index_plus_gap] -= gap

        self.features = features
        xc = features.dot(coeffs)

        # Simulation of latent variables
        pi = self.logistic_grad(-xc)
        u = np.random.rand(n_samples)
        Z = (u <= 1 - pi)
        self.Z = Z

        # Simulation of true times
        n_samples_class_1 = np.sum(Z)
        n_samples_class_0 = n_samples - n_samples_class_1
        T = np.empty(n_samples)
        pi_0_est = 1 - Z.mean()
        T[Z == 0] = np.random.geometric(p0, size=n_samples_class_0)

        # Compute p_c to obtain censoring rate r_c
        r_c_ = 1 - r_c
        p0_ = 1 - p0
        p1_ = 1 - p1
        pi_0_ = 1 - pi_0_est
        a = r_c_ * p0_ * p1_
        b = p0 * pi_0_est * p1_ + p1 * pi_0_ * p0_ - r_c_ * (p1_ + p0_)
        c = r_c_ - p0 * pi_0_est - p1 * pi_0_
        res = self.poldeg2_solver(a=a, b=b, c=c)
        if isinstance(res, list):
            if res[0] > 0:
                pc = 1 - res[0]
            else:
                pc = 1 - res[1]
        else:
            pc = 1 - res
        T[Z == 1] = np.random.geometric(p1, size=n_samples_class_1)

        # Simulation of the censoring
        C = np.random.geometric(pc, size=n_samples)

        # Censoring indicator: 1 if it is a time of failure, 0 if it's 
        # censoring.
        delta = (T <= C).astype(int)

        # Observed time
        Y = np.minimum(T, C).astype(int)
        if np.sum(Y == 0) > 0:
            Y += 1
        self.delta = delta
        self.Y = Y

<<<<<<< HEAD
        return features, Y, delta, Z, pi
=======
        return features, Y, delta, Z, pi
>>>>>>> 563bcf79c3de9c930f695693923967f0ca43595c
