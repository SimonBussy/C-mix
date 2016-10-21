# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

import numpy as np
from datetime import datetime
from QNEM.history import History
from time import time
from scipy.optimize import fmin_l_bfgs_b
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()


class Learner:
    """The base class for a Solver. This class should not be used by end-users.
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

    Class attributes

    coeffs: ndarray, shape=(4,)
        The parameters p0, p1 and pi of the mixture and pc the censoring 
        parameter
    """

    def __init__(self, max_iter, verbose, print_every, tol):
        Learner.__init__(self, verbose=verbose, print_every=print_every)
        self.max_iter = max_iter
        self.print_every = print_every
        self.verbose = verbose
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

        :param Y: ndarray
            Temporal data
        :param delta: ndarray
            Censoring indicator
        :param coeffs: ndarray, shape=(4,)
            coeffs[0]: Shape parameter of the geometric distribution for the
            first component
            coeffs[1]: Shape parameter of the geometric distribution for the
            second component
            coeffs[2]: Shape parameter of the geometric distribution for the
            censoring component
            coeffs[3]: The mixture parameter
        :return: float
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

    def fit(self, Y, delta, model):
        """Fit the censoring mixture of geometric distributions
        with two components
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
        return None


class QNEM(Learner):
    """QNEM Algorithm for fitting a censoring mixture of geometric distributions
    with two components and elasticNet regularization
    """

    def __init__(self, l_elastic_net=0., eta=.1, max_iter=100, verbose=True,
                 print_every=1, tol=1e-5, warm_start=False, model="C-mix",
                 intercept=False):
        Learner.__init__(self, verbose=verbose, print_every=print_every)
        self.l_elastic_net = l_elastic_net
        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.model = model
        self.intercept = intercept

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
        """
        pi = self.predict_proba(X, self.intercept, self.coeffs)
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

    @staticmethod
    def _c_index(Y, delta, marker):
        """Compute the C-index score for a given marker vector and the
        corresponding times and censoring indicator vectors, using the R 
        package survival
        """
        n_samples_test = Y.shape[0]
        data = np.concatenate((Y.reshape(n_samples_test, 1),
                               delta.reshape(n_samples_test, 1),
                               marker.reshape(n_samples_test, 1)),
                              axis=1)
        ro.globalenv['data'] = data
        ro.r('Y = data[,1]')
        ro.r('delta = data[,2]')
        ro.r('marker = data[,3]')
        ro.r('library(survival)')
        ro.r('surv <- Surv(time=Y, event=delta, type="right")')
        C_index = ro.r('survConcordance(surv ~ marker)$concordance')
        C_index = max(C_index, 1 - C_index)
        return C_index

    def _func_obj(self, X, Y, delta, coeffs_ext):
        """The global objective to be minimized by the QNEM algorithm
        (including penalization)
        """
        n_features = self.n_features
        if self.intercept:
            coeffs = coeffs_ext[:n_features + 1] - coeffs_ext[n_features + 1:]
            coeffs_ext = np.delete(coeffs_ext, [0, n_features + 1])
        else:
            coeffs = coeffs_ext[:n_features] - coeffs_ext[n_features:]
        self.coeffs = coeffs
        log_lik = self._log_lik(X, Y, delta)
        pen = self._func_pen(coeffs_ext)
        return -log_lik + pen

    def _func_sub_obj(self, X, q, coeffs_ext):
        """Computes the objective, namely the function to be minimized at
        each QNEM iteration using fmin_l_bfgs_b, for the incidence part 
        of the model. It computes
            mean( -q_i -x_i^Y beta - log(1 + exp(-x_i^Y beta)) 
            + penalization(beta)

        :param X: ndarray, shape=(n_samples, n_features)
            The features matrix
        :param q: ndarray, shape=(n_samples,)
            The soft-assignments obtained by the E-step
        :param coeffs_ext: ndarray, shape=(2*n_features,)
            The parameters of the mixture decompose 
            on positive and negative parts
        :return: float
            The value of the objective to be minimized at each QNEM step.
        """
        n_features = self.n_features
        if self.intercept:
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
        """Computes the gradient of the objective used in fmin_l_bfgs_b
        """
        n_features = self.n_features
        n_samples = self.n_samples
        if self.intercept:
            coeffs = coeffs_ext[:n_features + 1] - coeffs_ext[n_features + 1:]
            coeffs_0 = coeffs[0]
            coeffs = coeffs[1:]
        else:
            coeffs_0 = 0
            coeffs = coeffs_ext[:n_features] - coeffs_ext[n_features:]

        grad_pen = self._grad_pen(coeffs)
        u = coeffs_0 + X.dot(coeffs)

        if self.intercept:
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
        """
        verbose = self.verbose
        max_iter = self.max_iter
        print_every = self.print_every
        tol = self.tol
        warm_start = self.warm_start
        model = self.model
        intercept = self.intercept

        n_samples, n_features = X.shape
        self.n_samples = n_samples
        self.n_features = n_features
        self._start_solve()

        # Initialize coeffs to 0. which makes pi all equal to 0.5
        if intercept:
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
            pi = self.predict_proba(X, intercept, coeffs)

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
                maxiter=50,
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
        self.coeffs = -coeffs

    @staticmethod
    def predict_proba(X, intercept, coeffs):
        """Computes the probabilities of being on the low-risk group
        """
        if intercept:
            coeffs_0 = coeffs[0]
            coeffs = coeffs[1:]
        else:
            coeffs_0 = 0
        u = coeffs_0 + X.dot(coeffs)
        return QNEM.logistic_grad(u)

    def score(self, X, Y, delta, metric):
        """Computes the score with the trained parameters on the given data,
        either log-likelihood or C-index
        """
        if metric == 'log_lik':
            return self._log_lik(X, Y, delta)

        if metric == 'C-index':
            return self._c_index(Y, delta, self.predict_proba(X, self.intercept,
                                                              self.coeffs))

    def cross_validate(self, X, Y, delta, n_folds=3, eta=0.1,
                       adaptative_grid=True, grid_size=50,
                       grid_elastic_net=np.array([0]), shuffle=True,
                       verbose=True, metric='log_lik'):
        """Apply n_folds cross-validation using the given data, to select the
        best penalization parameter
        """
        from sklearn.cross_validation import KFold
        n_samples = Y.shape[0]
        cv = KFold(n_samples, n_folds=n_folds, shuffle=shuffle)
        self.grid_elastic_net = grid_elastic_net
        self.adaptative_grid = adaptative_grid
        self.grid_size = grid_size
        tol = self.tol
        warm_start = self.warm_start
        model = self.model

        if adaptative_grid:
            # from KKT condition:
            gamma_max = 1. / np.log(10.) * np.log(
                1. / (1. - eta) * (.5 / n_samples) \
                * np.absolute(X).sum(axis=0).max())
            grid_elastic_net = np.logspace(gamma_max - 4, gamma_max, grid_size)

        learners = [
            QNEM(verbose=False, tol=tol, eta=eta, warm_start=warm_start,
                 model=model, intercept=self.intercept)
            for _ in range(n_folds)
            ]

        n_grid_elastic_net = grid_elastic_net.shape[0]
        scores = np.empty((n_grid_elastic_net, n_folds))
        if verbose is not None:
            verbose = self.verbose
        for idx_elasticNet, l_elastic_net in enumerate(grid_elastic_net):
            if verbose:
                print("Testing l_elastic_net=%.2e" % l_elastic_net, "on fold"),
            for n_fold, (idx_train, idx_test) in enumerate(cv):
                if verbose:
                    print(n_fold),
                X_train = X[idx_train]
                X_test = X[idx_test]
                Y_train = Y[idx_train]
                Y_test = Y[idx_test]
                delta_train = delta[idx_train]
                delta_test = delta[idx_test]
                learner = learners[n_fold]
                learner.l_elastic_net = l_elastic_net
                learner.fit(X_train, Y_train, delta_train)
                scores[idx_elasticNet, n_fold] = learner.score(
                    X_test, Y_test, delta_test, metric)
            if verbose:
                print(": avg_score=%.2e" % scores[idx_elasticNet, :].mean())

        self.scores = scores
        avg_scores = scores.mean(axis=1)
        self.avg_scores = avg_scores

        idx_best = np.unravel_index(avg_scores.argmax(), avg_scores.shape)[0]
        l_elastic_net_best = grid_elastic_net[idx_best]
        idx = [i for i, j in enumerate(
            list(avg_scores >= avg_scores.max() - np.std(avg_scores))
        )
               if j]
        if len(idx) > 0 and max(idx) != len(avg_scores) - 1:
            idx_chosen = max(idx)
        else:
            idx_chosen = idx_best

        l_elastic_net_chosen = grid_elastic_net[idx_chosen]
        self.grid_elastic_net = grid_elastic_net
        self.l_elastic_net_best = l_elastic_net_best
        self.l_elastic_net_chosen = l_elastic_net_chosen