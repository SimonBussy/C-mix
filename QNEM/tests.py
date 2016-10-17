import unittest
import rpy2.robjects as ro
import numpy as np
import pylab as pl
import pandas as pd
from QNEM.inference import qnem
from QNEM.simulation import CensoredGeomMixtureRegression
from sklearn.cross_validation import ShuffleSplit
from sklearn.preprocessing import scale
from time import time

class Test(unittest.TestCase):
    def test_global_behavior(self):
        ## Choose parameters ##
        n_samples = 20  # number of patients
        n_features = 10  # number of covariables
        nb_active_features = 3  # number of active covariables
        K = 1.  # value of the active coefficients
        gap = .3  # gap value to create high/low risk groups
        rho = 0.5  # coefficient of the toeplitz correlation matrix
        r_cf = .5  # confusion factors rate
        r_c = 0.5  # censoring rate
        pi0 = 0.75  # proportion of desired low risk patients rate
        p0 = .01  # geometric parameter for low risk patients
        p1 = 0.5  # geometric parameter for high risk patients
        verbose = True  # verbose mode to detail or not ongoing tasks

        simu = CensoredGeomMixtureRegression(verbose, n_samples, n_features,
                                             nb_active_features, K, rho, pi0,
                                             gap, r_c, r_cf, p0, p1)
        X, Y, delta, Z, pi = simu.simulate()

        ## Assign index for each feature ##
        features_names = range(X.shape[1])
        n_samples, n_features = X.shape

        ## Split data into training and test sets ##
        test_size = .3  # proportion of data used for testing
        rs = ShuffleSplit(n_samples, n_iter=1, test_size=test_size, random_state=0)
        for train_index, test_index in rs:
            X_test = X[test_index]
            delta_test = delta[test_index]
            Y_test = Y[test_index]

            X = X[train_index]
            Y = Y[train_index]
            delta = delta[train_index]

        print "%d%% for training, %d%% for testing." % \
              ((1 - test_size) * 100, test_size * 100)

        ## Choose parameters ##
        tol = 1e-6  # tolerance for the convergence stopping criterion
        eta = 0.2  # parameter controlling the trade-off between l1
        # and l2 regularization in the elasticNet
        intercept = True  # whether or not an intercept term is fitted
        gammaChosen = '1se'  # way to select l_elasticNet_chosen: '1se' or 'min'
        warm_start = True  # at each L-BGFS-B iteration, reset beta to 0 or take
        # the previous value
        grid_size = 3  # grid size for the cross validation procedure
        metric = 'C-index'  # cross-validation metric: 'log_lik' or 'C-index'
        verbose = True

        ## Choose between C-mix or CURE model ##
        model = "C-mix"  # "C-mix", "CURE"

        if verbose:
            print ' '
            print "Launching %s..." % model
            print ' '

        learner = qnem(l_elastic_net=0., eta=eta, max_iter=100, tol=tol,
                       warm_start=warm_start, verbose=verbose, model=model,
                       intercept=intercept)
        learner.n_features = n_features

        ## Cross-validation ##
        learner.cross_validate(X, Y, delta, n_folds=5, verbose=False, eta=eta,
                               grid_size=grid_size, metric=metric)
        avg_scores = learner.scores.mean(axis=1)
        l_elastic_net_best = learner.l_elastic_net_best
        if gammaChosen == '1se':
            l_elastic_net_chosen = learner.l_elastic_net_chosen
        if gammaChosen == 'min':
            l_elastic_net_chosen = l_elastic_net_best

        grid_elasticNet = learner.grid_elastic_net  # get the cross-validation grid
        # to plot learning curves

        ## Run selected model with l_elasticNet_chosen ##
        learner = qnem(l_elastic_net=l_elastic_net_chosen, eta=eta, tol=tol,
                       warm_start=warm_start, verbose=verbose, model=model,
                       intercept=intercept)
        learner.n_features = n_features
        learner.fit(X, Y, delta)


if __name__ == "main":
    unittest.main()
