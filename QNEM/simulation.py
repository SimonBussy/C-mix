# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

from time import time
from .base import BaseClass
from scipy.linalg.special_matrices import toeplitz
import numpy as np

def features_normal_cov_toeplitz(n_samples, n_features, rho=0.5):
    """
    Features obtained as samples of a centered Gaussian vector
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
            np.zeros(n_features),cov,size=n_samples)


def simulationmethod(simulate_method):
    """
    A decorator for simulation methods.
    It simply calls _start_simulation and _end_simulation methods
    """
    def decorated_simulate_method(self):
        self._start_simulation()
        result = simulate_method(self)
        self._end_simulation()
        self.data = result
        return result
    return decorated_simulate_method
    

class Simulation(BaseClass):
    """
    This is an abstract simulation class that inherits form BaseClass
    It does nothing besides printing stuff and verbosing
    """
    def __init__(self, **kwargs):
        BaseClass.__init__(self, **kwargs)
        # Set default parameters
        self.set_params(seed=None, verbose=True)
        self.set_params(**kwargs)
        if self.get_params("seed") is not None:
            self._set_seed()
        # No data simulated yet
        self.features = None
        self.labels = None

    def _set_seed(self):
        np.random.seed(self.get_params("seed"))
        return self

    def _start_simulation(self):
        self._set_info(time_start=self._get_now())
        self.__time_start = time()
        if self.get_params("verbose"):
            msg = "Launching simulation using {class_}..." \
                    .format(class_=self.get_info("class"))
            print("-" * len(msg))
            print(msg)

    def _end_simulation(self):
        self._set_info(time_end=self._get_now())
        t = time()
        self._set_info(time_elapsed=t - self.__time_start)
        if self.get_params("verbose"):
            msg = "Done simulating using {class_} in {time:.2e} seconds." \
                .format(class_=self.get_info("class"),
                        time=self.get_info("time_elapsed"))
            print(msg)


class CensoredGeomMixtureRegression(Simulation):
    """
    Class for the simulation of Censored Geometric Mixture Model
    """
    def __init__(self,verbose,n_samples,n_features,nb_active_features,
                 K,rho,pi0,gap,r_c,r_cf,p0,p1,**kwargs):
        Simulation.__init__(self, **kwargs)
        self.set_params(verbose=verbose,features_type="cov_toeplitz",
                        n_samples=n_samples,n_features=n_features,
                        nb_active_features=nb_active_features,K=K,
                        rho=rho,pi0=pi0,gap=gap,r_c=r_c,r_cf=r_cf,
                        p0=p0,p1=p1)
        self.set_params(**kwargs)
        self.p0 = p0
        self.p1 = p1
        self.r_c = r_c
        self.r_cf = r_cf

    @staticmethod
    def logistic_grad(z):
        """
        Overflow proof computation of 1 / (1 + exp(-z)))
        """
        idx_pos = np.where(z >= 0.)
        idx_neg = np.where(z < 0.)
        res = np.empty(z.shape)
        res[idx_pos] = 1. / (1. + np.exp(-z[idx_pos]))
        res[idx_neg] = 1 - 1. / (1. + np.exp(z[idx_neg]))
        return res

    @staticmethod
    def poldeg2_solver(a=0,b=0,c=0):
        """
        2nd order polynomial solver
        """
        if a==0:
            if b==0:
                return None
            return -c/b
        Delta=b*b-4*a*c 
        if Delta <0:
            return None
        if Delta ==0:
            return -b/(2*a) 
        if Delta >0:
            sqrt_Delta=np.sqrt(Delta) 
            return [(-b-sqrt_Delta)/(2*a) , (-b+sqrt_Delta)/(2*a)] 

    @simulationmethod
    def simulate(self):
        """
        Launch simulation of the data.
        """
        n_samples = self.get_params("n_samples")
        n_features = self.get_params("n_features")
        features_type = self.get_params("features_type")
        nb_active_features = self.get_params("nb_active_features")
        K = self.get_params("K")
        pi0 = self.get_params("pi0")
        gap = self.get_params("gap")
        p0 = self.p0        
        p1 = self.p1
        r_c = self.r_c
        r_cf = self.r_cf
        
        coeffs = np.zeros(n_features)
        coeffs[0:nb_active_features] = K
        if features_type == "cov_toeplitz":
            rho = self.get_params("rho")
            features = features_normal_cov_toeplitz(n_samples,n_features,rho)
        # Add class relative information on the design matrix    
        A = np.random.choice(range(n_samples), size=int((1-pi0)*n_samples), 
                             replace=False)
        A_ = np.delete(range(n_samples),A)
        features[A,:(nb_active_features + int((n_features - \
                     nb_active_features)*r_cf))] += gap
        features[A_,:(nb_active_features + int((n_features - \
                     nb_active_features)*r_cf))] -= gap
        
        self.features = features
        xc = features.dot(coeffs)
        
        # Simulation of latent variables
        pi = self.logistic_grad(-xc)
        u = np.random.rand(n_samples)
        Z = (u <= 1-pi)
        self.Z = Z

        # Simulation of true times
        n_samples1 = np.sum(Z)
        n_samples0 = n_samples - n_samples1
        T = np.empty(n_samples)
        pi0_est = 1-Z.mean()
        T[Z==0] = np.random.geometric(p0, size=n_samples0)
        # Compute p_c to obtain censoring rate r_c
        r_c_ = 1 - r_c
        p0_ = 1 - p0
        p1_ = 1 - p1
        pi0_ = 1 - pi0_est
        a = r_c_ * p0_ * p1_
        b = p0 * pi0_est * p1_ + p1 * pi0_ * p0_ - r_c_ * (p1_ + p0_) 
        c = r_c_ - p0 * pi0_est - p1 * pi0_
        res = self.poldeg2_solver(a=a,b=b,c=c)
        if isinstance(res, list):
            if res[0]>0:
                pc = 1 - res[0]
            else:
                pc = 1 - res[1]
        else:
            pc = 1 - res
        T[Z==1] = np.random.geometric(p1, size=n_samples1)       

        # Simulation of the censoring
        C = np.random.geometric(pc, size=n_samples)
            
        # Censoring indicator: 1 if it is a time of failure, 0 if it's 
        # censoring.
        delta = (T <= C).astype(int)
        # Observed time
        Y = np.minimum(T, C).astype(int) 
        if np.sum(Y==0)>0:
            Y += 1
        self.delta = delta
        self.Y = Y
        
        return features, Y, delta, Z, pi