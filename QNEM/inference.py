# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

import numpy as np
from datetime import datetime
from QNEM.history import History
from time import time
from scipy.optimize import fmin_l_bfgs_b
from lifelines.utils import concordance_index as c_index_score


class Learner:
    """The base class for a Solver.
    Not intended for end-users, but for development only.
    It should be sklearn-learn compliant

    Parameters
    ----------
    verbose : `bool`, default=True
        If `True`, we verbose things, otherwise the solver does not
        print anything (but records information in history anyway)

    print_every : `int`, default=10
        Print history information when ``n_iter`` (iteration number) is
        a multiple of ``print_every``
    """

    def __init__(self, verbose=True, print_every=10):
        self.verbose = verbose
        self.print_every = print_every
        self.history = History()

    def _init_coeffs(self, n_features):
        self.coeffs = np.empty(n_features)
        self.n_features = n_features

    def _start_solve(self):
        # Reset history
        self.history.clear()
        self.time_start = Learner._get_now()
        self._numeric_time_start = time()

        if self.verbose:
            print("Launching the solver " + self.__class__.__name__ + "...")

    def _end_solve(self):
        self.time_end = self._get_now()
        t = time()
        self.time_elapsed = t - self._numeric_time_start

        if self.verbose:
            print("Done solving using " + self.__class__.__name__ + " in "
                  + "%.2e seconds" % self.time_elapsed)

    @staticmethod
    def _get_now():
        return str(datetime.now()).replace(" ", "_").replace(":", "-")

    def get_history(self, key=None):
        """Return history of the solver

        Parameters
        ----------
        key : str [optional, default=None]
            if None all history is returned as a dict
            if str then history of the required key is given

        Returns
        -------
        output : dict or list
            if key is None or key is not in history then output is
                dict containing history of all keys
            if key is not None and key is in history, then output is a list
            containing history for the given key
        """
        val = self.history.values.get(key, None)
        if val is None:
            return self.history.values
        else:
            return val


class MixtureGeoms(Learner):
    """EM Algorithm for fitting a censoring mixture of geometric distributions
        with two components

    Parameters
    ----------
    tol : `float`, default=1e-5
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it). By default the solver does ``max_iter``
        iterations

    max_iter : `int`, default=100
        Maximum number of iterations of the solver

    verbose : `bool`, default=True
        If `True`, we verbose things, otherwise the solver does not
        print anything (but records information in history anyway)

    print_every : `int`, default=10
        Print history information when ``n_iter`` (iteration number) is
        a multiple of ``print_every``
    """

    def __init__(self, max_iter=100, verbose=True, print_every=10, tol=1e-5):
        Learner.__init__(self, verbose=verbose, print_every=print_every)
        self.max_iter = max_iter
        self.verbose = verbose
        self.print_every = print_every
        self.tol = tol
        self._init_coeffs(4)

        # Attributes that will be instantiated afterwards
        self.p0 = None
        self.p1 = None
        self.pc = None
        self.pi = None

    @staticmethod
    def log_lik(Y, delta, coeffs):
        """Computes the log-likelihood of the censoring mixture of two
        geometric distributions

        Parameters
        ----------
        Y : `np.ndarray`, shape=(n_samples,)
            Times of the event of interest

        delta : `np.ndarray`, shape=(n_samples,)
            Censoring indicator

        coeffs : `np.ndarray`, shape=(4,)
            coeffs[0] : Shape parameter of the geometric distribution for the
            first component
            coeffs[1] : Shape parameter of the geometric distribution for the
            second component
            coeffs[2] : Shape parameter of the geometric distribution for the
            censoring component
            coeffs[3] : The mixture parameter

        Returns
        -------
            The value of the log-likelihood
        """
        p0, p1, pc, pi = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
        prb = ((pi * p0 * (1. - p0) ** (Y - 1.)
                + (1. - pi) * p1 * (1. - p1) ** (Y - 1.)
                ) * (1. - pc) ** Y
               ) ** delta \
              * ((pi * (1 - p0) ** Y
                  + (1. - pi) * (1. - p1) ** Y
                  ) * pc * (1. - pc) ** (Y - 1.)
                 ) ** (1. - delta)
        return np.mean(np.log(prb))

    def fit(self, Y, delta, model='C-mix'):
        """Fit the censoring mixture of geometric distributions
        with two components

        Parameters
        ----------
        Y : `np.ndarray`, shape=(n_samples,)
            Times of the event of interest

        delta : `np.ndarray`, shape=(n_samples,)
            Censoring indicator

        model : 'C-mix', 'CURE', default='C-mix'
            The model to be fitted
        """
        verbose = self.verbose
        max_iter = self.max_iter
        print_every = self.print_every
        tol = self.tol
        self._start_solve()

        # Split at random the sample with probability 0.5
        n_samples = Y.shape[0]
        pi = 0.5
        Z = np.random.binomial(1, pi, size=n_samples)
        p1 = 1. / np.mean(Y[(delta == 1) + (Z == 1)])
        p0 = 1. / np.mean(Y[(delta == 1) + (Z == 0)])
        if p0 > p1:
            tmp = p0
            p0 = p1
            p1 = tmp

        if model == 'CURE':
            p0 = 0
        pc = 1. / np.mean(Y[delta == 0])

        log_lik = self.log_lik(Y, delta, np.array([p0, p1, pc, pi]))
        obj = -log_lik
        rel_obj = 1.
        self.history.update(n_iter=0, obj=obj, rel_obj=rel_obj)
        if verbose:
            self.history.print_history()

        for n_iter in range(max_iter):
            if n_iter % print_every == 0:
                self.history.update(n_iter=n_iter, obj=obj,
                                    rel_obj=rel_obj)
                if verbose:
                    self.history.print_history()
            # E-Step
            a = ((1. - p1) ** (Y - 1.) * p1) ** delta * ((1. - p1) ** Y) ** (
                1. - delta) * (1. - pi)
            b = ((1. - p0) ** (Y - 1.) * p0) ** delta * ((1. - p0) ** Y) ** (
                1. - delta) * pi
            q = a / (a + b)
            # M-Step
            if model == 'C-mix':
                p0 = ((1. - q) * delta).mean() / ((1. - q) * Y).mean()
            p1 = (delta * q).mean() / (q * Y).mean()
            pi = 1. - np.mean(q)
            prev_obj = obj
            log_lik = self.log_lik(Y, delta, np.array([p0, p1, pc, pi]))
            obj = -log_lik
            rel_obj = abs(obj - prev_obj) / abs(prev_obj)
            if (n_iter > max_iter) or (rel_obj < tol):
                break

        n_iter += 1
        self.history.update(n_iter=n_iter, obj=obj, rel_obj=rel_obj)
        if verbose:
            self.history.print_history()
        self._end_solve()
        self.p0 = p0
        self.p1 = p1
        self.pc = pc
        self.pi = pi
        self.coeffs[:] = np.array([p0, p1, pc, pi])


class QNEM(Learner):
    """QNEM Algorithm for fitting a censoring mixture of geometric distributions
    with two components and elasticNet regularization

    Parameters
    ----------
    model : 'C-mix', 'CURE', default='C-mix'
        The model to be fitted

    fit_intercept : `bool`, default=True
        If `True`, include an intercept in the model

    l_elastic_net : `float`, default=0
        Level of ElasticNet penalization

    eta: `float`, default=0.1
        The ElasticNet mixing parameter, with 0 <= eta <= 1.
        For eta = 0 this is ridge (L2) regularization
        For eta = 1 this is lasso (L1) regularization
        For 0 < eta < 1, the regularization is a linear combination
        of L1 and L2

    max_iter : `int`, default=100
        Maximum number of iterations of the solver

    verbose : `bool`, default=True
        If `True`, we verbose things, otherwise the solver does not
        print anything (but records information in history anyway)

    print_every : `int`, default=10
        Print history information when ``n_iter`` (iteration number) is
        a multiple of ``print_every``

    tol : `float`, default=1e-5
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it). By default the solver does ``max_iter``
        iterations

    warm_start : `bool`, default=False
        If true, learning will start from the last reached solution
    """

    def __init__(self, model="C-mix", fit_intercept=False, l_elastic_net=0.,
                 eta=.1, max_iter=100, verbose=True, print_every=1, tol=1e-5,
                 warm_start=False):
        Learner.__init__(self, verbose=verbose, print_every=print_every)
        self.l_elastic_net = l_elastic_net
        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.model = model
        self.fit_intercept = fit_intercept

        # Attributes that will be instantiated afterwards
        self.coeffs = None
        self.coeffs_ext = None
        self.p1 = None
        self.p0 = None
        self.pc = None
        self.pi = None
        self.avg_scores = None
        self.scores = None
        self.l_elastic_net_best = None
        self.l_elastic_net_chosen = None
        self.grid_elastic_net = None
        self.n_features = None
        self.n_samples = None
        self.adaptative_grid = None
        self.grid_size = None

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
    def logistic_loss(z):
        """Overflow proof computation of log(1 + exp(-z))
        """
        idx_pos = np.where(z >= 0.)
        idx_neg = np.where(z < 0.)
        res = np.empty(z.shape)
        res[idx_pos] = np.log(1. + np.exp(-z[idx_pos]))
        z_neg = z[idx_neg]
        res[idx_neg] = -z_neg + np.log(1. + np.exp(z_neg))
        return res

    def _func_pen(self, coeffs_ext):
        """Computes the elasticNet penalization of the global objective to be
        minimized by the QNEM algorithm

        Parameters
        ----------
        coeffs_ext: `np.ndarray`, shape=(2*n_features,)
            The parameters of the mixture decompose
            on positive and negative parts

        Returns
        -------
        output : `float`
            The value of the penalization of the global objective
        """
        l_elastic_net = self.l_elastic_net
        eta = self.eta
        n_features = self.n_features
        coeffs = coeffs_ext[:n_features] - coeffs_ext[n_features:]
        return l_elastic_net * ((1. - eta) * coeffs_ext.sum()
                                + 0.5 * eta * np.linalg.norm(coeffs) ** 2)

    def _grad_pen(self, coeffs):
        """Computes the gradient of the elasticNet penalization of the global
        objective to be minimized by the QNEM algorithm

        Parameters
        ----------
        coeffs : `np.ndarray`, shape=(n_features,)
            The parameters of the mixture

        Returns
        -------
        output : `float`
            The gradient of the penalization of the global objective
        """
        l_elastic_net = self.l_elastic_net
        eta = self.eta
        n_features = self.n_features
        grad = np.zeros(2 * n_features)
        # Gradient of lasso penalization
        grad += l_elastic_net * (1 - eta)
        # Gradient of ridge penalization
        grad_pos = (l_elastic_net * eta)
        grad[:n_features] += grad_pos * coeffs
        grad[n_features:] -= grad_pos * coeffs
        return grad

    def _log_lik(self, X, Y, delta):
        """Computes the likelihood of the censoring mixture model

        Parameters
        ----------
        X : `np.ndarray`, shape=(n_samples, n_features)
            The features matrix

        Y : `np.ndarray`, shape=(n_samples,)
            Times of the event of interest

        delta : `np.ndarray`, shape=(n_samples,)
            Censoring indicator

        Returns
        -------
        output : `float`
            The log-likelihood computed on the given data
        """
        pi = self.predict_proba(X, self.fit_intercept, self.coeffs)
        p0, p1, pc = self.p0, self.p1, self.pc
        prb = ((pi * p0 * (1. - p0) ** (Y - 1.)
                + (1. - pi) * p1 * (1. - p1) ** (Y - 1.)
                ) * (1. - pc) ** Y
               ) ** delta \
              * ((pi * (1 - p0) ** Y
                  + (1. - pi) * (1. - p1) ** Y
                  ) * pc * (1. - pc) ** (Y - 1.)
                 ) ** (1. - delta)
        return np.mean(np.log(prb))

    def _func_obj(self, X, Y, delta, coeffs_ext):
        """The global objective to be minimized by the QNEM algorithm
        (including penalization)

        Parameters
        ----------
        X : `np.ndarray`, shape=(n_samples, n_features)
            The features matrix

        Y : `np.ndarray`, shape=(n_samples,)
            Times of the event of interest

        delta : `np.ndarray`, shape=(n_samples,)
            Censoring indicator

        coeffs_ext : `np.ndarray`, shape=(2*n_features,)
            The parameters of the mixture decompose
            on positive and negative parts

        Returns
        -------
        output : `float`
            The value of the global objective to be minimized
        """
        n_features = self.n_features
        if self.fit_intercept:
            coeffs = coeffs_ext[:n_features + 1] - coeffs_ext[n_features + 1:]
            coeffs_ext = np.delete(coeffs_ext, [0, n_features + 1])
        else:
            coeffs = coeffs_ext[:n_features] - coeffs_ext[n_features:]
        self.coeffs = coeffs
        log_lik = self._log_lik(X, Y, delta)
        pen = self._func_pen(coeffs_ext)
        return -log_lik + pen

    def _func_sub_obj(self, X, q, coeffs_ext):
        """Computes the sub objective, namely the function to be minimized at
        each QNEM iteration using fmin_l_bfgs_b, for the incidence part 
        of the model. It computes
            mean(q_i x_i^Y beta + log(1 + exp(-x_i^Y beta))
            + penalization(beta)

        Parameters
        ----------
        X : `np.ndarray`, shape=(n_samples, n_features)
            The features matrix

        q : `np.ndarray`, shape=(n_samples,)
            The soft-assignments obtained by the E-step

        coeffs_ext : `np.ndarray`, shape=(2*n_features,)
            The parameters of the mixture decompose 
            on positive and negative parts

        Returns
        -------
        output : `float`
            The value of the sub objective to be minimized at each QNEM step
        """
        n_features = self.n_features
        if self.fit_intercept:
            coeffs = coeffs_ext[:n_features + 1] - coeffs_ext[n_features + 1:]
            coeffs_0 = coeffs[0]
            coeffs = coeffs[1:]
            coeffs_ext = np.delete(coeffs_ext, [0, n_features + 1])
        else:
            coeffs_0 = 0
            coeffs = coeffs_ext[:n_features] - coeffs_ext[n_features:]
        pen = self._func_pen(coeffs_ext)
        u = coeffs_0 + X.dot(coeffs)
        sub_obj = (q * u + self.logistic_loss(u)).mean()
        return sub_obj + pen

    def _grad_sub_obj(self, X, q, coeffs_ext):
        """Computes the gradient of the sub objective used in fmin_l_bfgs_b

        Parameters
        ----------
        X : `np.ndarray`, shape=(n_samples, n_features)
            The features matrix

        q : `np.ndarray`, shape=(n_samples,)
            The soft-assignments obtained by the E-step

        coeffs_ext : `np.ndarray`, shape=(2*n_features,)
            The parameters of the mixture decompose
            on positive and negative parts

        Returns
        -------
        output : `float`
            The value of the sub objective gradient at each QNEM step
        """
        n_features = self.n_features
        n_samples = self.n_samples
        if self.fit_intercept:
            coeffs = coeffs_ext[:n_features + 1] - coeffs_ext[n_features + 1:]
            coeffs_0 = coeffs[0]
            coeffs = coeffs[1:]
        else:
            coeffs_0 = 0
            coeffs = coeffs_ext[:n_features] - coeffs_ext[n_features:]
        grad_pen = self._grad_pen(coeffs)
        u = coeffs_0 + X.dot(coeffs)
        if self.fit_intercept:
            X = np.concatenate((np.ones(n_samples).reshape(1, n_samples).T, X),
                               axis=1)
            grad_pen = np.concatenate([[0], grad_pen[:n_features], [0],
                                       grad_pen[n_features:]])
        grad = (X * (q - self.logistic_grad(-u)).reshape(n_samples, 1)).mean(
            axis=0)
        grad_sub_obj = np.concatenate([grad, -grad])
        return grad_sub_obj + grad_pen

    def fit(self, X, Y, delta):
        """Fit the supervised censoring mixture of geometric distributions.
        After the call to the method, trained parameters are saved
        in self.p0, self.p1 and self.coeffs

        Parameters
        ----------
        X : `np.ndarray`, shape=(n_samples, n_features)
            The features matrix

        Y : `np.ndarray`, shape=(n_samples,)
            Times of the event of interest

        delta : `np.ndarray`, shape=(n_samples,)
            Censoring indicator
        """
        verbose = self.verbose
        max_iter = self.max_iter
        print_every = self.print_every
        tol = self.tol
        warm_start = self.warm_start
        model = self.model
        fit_intercept = self.fit_intercept

        n_samples, n_features = X.shape
        self.n_samples = n_samples
        self.n_features = n_features
        self._start_solve()

        # Initialize coeffs to 0. which makes pi all equal to 0.5
        if fit_intercept:
            n_features += 1
        coeffs = np.zeros(n_features)
        coeffs_ext = np.zeros(2 * n_features)

        func_obj = self._func_obj
        func_sub_obj = self._func_sub_obj
        grad_sub_obj = self._grad_sub_obj

        # We initialize p0 and p1 by fitting a censoring mixture of geometrics
        mixt_geoms = MixtureGeoms(max_iter=max_iter, verbose=False,
                                  print_every=print_every, tol=tol)
        mixt_geoms.fit(Y, delta, model)
        p0 = mixt_geoms.p0
        p1 = mixt_geoms.p1
        pc = mixt_geoms.pc

        self.p0, self.p1, self.pc = p0, p1, pc
        if verbose:
            print("init: p0=%s" % p0)
            print("init: p1=%s" % p1)

        obj = func_obj(X, Y, delta, coeffs_ext)
        rel_obj = 1.

        # Bounds vector for the L-BGFS-B algorithm
        bounds = [(0, None)] * n_features * 2

        for n_iter in range(max_iter):
            if n_iter % print_every == 0:
                self.history.update(n_iter=n_iter, obj=obj, rel_obj=rel_obj)
                if verbose:
                    self.history.print_history()
            pi = self.predict_proba(X, fit_intercept, coeffs)

            # E-Step
            a = ((1. - p1) ** (Y - 1.) * p1) ** delta * ((1. - p1) ** Y) ** (
                1. - delta) * (1. - pi)
            b = ((1. - p0) ** (Y - 1.) * p0) ** delta * ((1. - p0) ** Y) ** (
                1. - delta) * pi
            q = a / (a + b)

            # M-Step
            if model == 'C-mix':
                p0 = ((1. - q) * delta).mean() / ((1. - q) * Y).mean()
            p1 = (delta * q).mean() / (q * Y).mean()
            self.p0, self.p1 = p0, p1

            if warm_start:
                x0 = coeffs_ext
            else:
                x0 = np.zeros(2 * n_features)
            coeffs_ext = fmin_l_bfgs_b(
                func=lambda coeffs_ext_: func_sub_obj(X, q, coeffs_ext_),
                x0=x0,
                fprime=lambda coeffs_ext_: grad_sub_obj(X, q, coeffs_ext_),
                disp=False,
                bounds=bounds,
                maxiter=60,
                # pgtol=1e-20
                pgtol=1e-5
            )[0]

            coeffs = coeffs_ext[:n_features] - coeffs_ext[n_features:]
            prev_obj = obj

            obj = func_obj(X, Y, delta, coeffs_ext)
            rel_obj = abs(obj - prev_obj) / abs(prev_obj)
            if (n_iter > max_iter) or (rel_obj < tol):
                break

        n_iter += 1
        self.history.update(n_iter=n_iter, obj=obj, rel_obj=rel_obj)
        if verbose:
            self.history.print_history()
            print("At the end: p0=%s" % p0)
            print("At the end: p1=%s" % p1)

        self._end_solve()
        self.p0, self.p1 = p0, p1
        self.pi = pi
        # self.coeffs = -coeffs
        self.coeffs = coeffs

    @staticmethod
    def predict_proba(X, fit_intercept, coeffs):
        """Probability estimates for being on the high-risk group.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : `np.ndarray`, shape=(n_samples, n_features)
            Input features matrix

        fit_intercept : `bool`
            If `True`, include an intercept in the model

        coeffs : `np.ndarray`, shape=(n_features,)
            The parameters of the mixture

        Returns
        -------
        output : `np.ndarray`, shape=(n_samples,)
            Returns the probability of the sample for being on
            the high-risk group
        """
        if fit_intercept:
            coeffs_0 = coeffs[0]
            coeffs = coeffs[1:]
        else:
            coeffs_0 = 0
        u = coeffs_0 + X.dot(coeffs)
        return QNEM.logistic_grad(u)

    def score(self, X, Y, delta, metric):
        """Computes the score with the trained parameters on the given data,
        either log-likelihood or C-index

        Parameters
        ----------
        X : `np.ndarray`, shape=(n_samples, n_features)
            The features matrix

        Y : `np.ndarray`, shape=(n_samples,)
            Times of the event of interest

        delta : `np.ndarray`, shape=(n_samples,)
            Censoring indicator

        metric : 'log_lik', 'C-index'
            Either computes log-likelihood or C-index

        Returns
        -------
        output : `float`
            The score computed on the given data
        """
        if metric == 'log_lik':
            return self._log_lik(X, Y, delta)

        if metric == 'C-index':
            return c_index_score(Y, self.predict_proba(X, self.fit_intercept,
                                                       self.coeffs), delta)

    def cross_validate(self, X, Y, delta, n_folds=3, eta=0.1,
                       adaptative_grid=True, grid_size=50,
                       grid_elastic_net=np.array([0]), shuffle=True,
                       verbose=True, metric='log_lik'):
        """Apply n_folds cross-validation using the given data, to select the
        best penalization parameter

        Parameters
        ----------
        X : `np.ndarray`, shape=(n_samples, n_features)
            The features matrix

        Y : `np.ndarray`, shape=(n_samples,)
            Times of the event of interest

        delta : `np.ndarray`, shape=(n_samples,)
            Censoring indicator

        n_folds : `int`, default=3
            Number of folds. Must be at least 2.

        eta : `float`, default=0.1
            The ElasticNet mixing parameter, with 0 <= eta <= 1.
            For eta = 0 this is ridge (L2) regularization
            For eta = 1 this is lasso (L1) regularization
            For 0 < eta < 1, the regularization is a linear combination
            of L1 and L2

        adaptative_grid : `bool`, default=True
            If `True`, adapt the ElasticNet strength parameter grid using the
            KKT conditions

        grid_size : `int`, default=50
            Grid size if adaptative_grid=`True`

        grid_elastic_net : `np.ndarray`, default=np.array([0])
            Grid of ElasticNet strength parameters to be run through, if
            adaptative_grid=`False`

        shuffle : `bool`, default=True
            Whether to shuffle the data before splitting into batches

        verbose : `bool`, default=True
            If `True`, we verbose things, otherwise the solver does not
            print anything (but records information in history anyway)

        metric : 'log_lik', 'C-index', default='log_lik'
            Either computes log-likelihood or C-index
        """
        from sklearn.model_selection import KFold
        n_samples = Y.shape[0]
        cv = KFold(n_splits=n_folds, shuffle=shuffle)
        self.grid_elastic_net = grid_elastic_net
        self.adaptative_grid = adaptative_grid
        self.grid_size = grid_size
        tol = self.tol
        warm_start = self.warm_start
        model = self.model

        if adaptative_grid:
            # from KKT conditions
            gamma_max = 1. / np.log(10.) * np.log(
                1. / (1. - eta) * (.5 / n_samples)
                * np.absolute(X).sum(axis=0).max())
            grid_elastic_net = np.logspace(gamma_max - 4, gamma_max, grid_size)

        learners = [
            QNEM(verbose=False, tol=tol, eta=eta, warm_start=warm_start,
                 model=model, fit_intercept=self.fit_intercept)
            for _ in range(n_folds)
        ]

        n_grid_elastic_net = grid_elastic_net.shape[0]
        scores = np.empty((n_grid_elastic_net, n_folds))
        if verbose is not None:
            verbose = self.verbose
        for idx_elasticNet, l_elastic_net in enumerate(grid_elastic_net):
            if verbose:
                print("Testing l_elastic_net=%.2e" % l_elastic_net, "on fold ",
                      end="")
            for n_fold, (idx_train, idx_test) in enumerate(cv.split(X)):
                if verbose:
                    print(" " + str(n_fold), end="")
                X_train, X_test = X[idx_train], X[idx_test]
                Y_train, Y_test = Y[idx_train], Y[idx_test]
                delta_train, delta_test = delta[idx_train], delta[idx_test]
                learner = learners[n_fold]
                learner.l_elastic_net = l_elastic_net
                learner.fit(X_train, Y_train, delta_train)
                scores[idx_elasticNet, n_fold] = learner.score(
                    X_test, Y_test, delta_test, metric)
            if verbose:
                print(": avg_score=%.2e" % scores[idx_elasticNet, :].mean())

        avg_scores = scores.mean(1)
        std_scores = scores.std(1)
        idx_best = avg_scores.argmax()
        l_elastic_net_best = grid_elastic_net[idx_best]
        idx_chosen = max([i for i, j in enumerate(
            list(avg_scores >= avg_scores.max() - std_scores[idx_best])) if j])
        l_elastic_net_chosen = grid_elastic_net[idx_chosen]

        self.grid_elastic_net = grid_elastic_net
        self.l_elastic_net_best = l_elastic_net_best
        self.l_elastic_net_chosen = l_elastic_net_chosen
        self.scores = scores
        self.avg_scores = avg_scores
