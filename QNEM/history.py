# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

from .base import BaseClass
import numpy as np
from collections import defaultdict

def n_iter_func(n_iter, **kwargs):
    return n_iter

def obj_func(obj, **kwargs):
    return obj

def rel_obj_func(rel_obj, **kwargs):
    return rel_obj

def spars_func(coeffs, **kwargs):
    eps = np.finfo(coeffs.dtype).eps
    return np.sum(np.abs(coeffs) > eps, axis=None)

class History(BaseClass):
    """A class to manage the history along iterations of a solver.
    """

    def __init__(self, **kwargs):
        BaseClass.__init__(self, **kwargs)
        self.set_params(minimum_col_width=8,
                        print_order=["n_iter", "obj", "step", "rel_obj"])
        self.set_params(**kwargs)

        # Instantiate values of the history
        self.clear()

        # History function to compute history values based on parameters
        # used in a solver
        history_func = {}
        history_func["n_iter"] = n_iter_func
        history_func["obj"] = obj_func
        history_func["rel_obj"] = rel_obj_func
        self.history_func = history_func

        # Default print style of history values. Default is %.2e
        print_style = defaultdict(lambda: "%g")
        print_style["n_iter"] = "%d"
        print_style["n_epoch"] = "%d"
        print_style["n_inner_prod"] = "%d"
        print_style["spars"] = "%d"
        print_style["rank"] = "%d"
        self.set_params(print_style=print_style)

    def clear(self):
        """Reset history values"""
        self.values = defaultdict(list)

    def update(self, **kwargs):
        """Update the history along the iterations.

        For each keyword argument, we apply the history function corresponding
        to this keyword, and use its results in the history
        """
        n_iter = kwargs["n_iter"]
        self.n_iter = n_iter
        history_func = self.history_func
        history = self.values
        for key, func in history_func.items():
            history[key].append(func(**kwargs))

    def set_print_order(self, *args):
        """Allows to set the print order of the solver's history
        """
        self.set_params(print_order=list(*args))
        self.clear()
        return self

    def update_history_func(self, **kwargs):
        self.history_func.update(**kwargs)
        self.clear()
        return self

    def update_print_style(self, **kwargs):
        self.print_style.update(**kwargs)
        return self

    def print_history(self):
        """Verbose the current line of history
        """
        values = self.values
        n_iter = self.n_iter
        print_order = self.params["print_order"]
        # If this is the first iteration, plot the history's column names
        if n_iter == 0:
            min_width = self.params["minimum_col_width"]
            line = ' | '.join([name.center(min_width) for name in
                               print_order if name in values])
            names = [name.center(min_width) for name in print_order]
            self.col_widths = map(len, names)
            print line

        col_widths = self.col_widths
        print_style = self.params["print_style"]
        line = ' | '.join([(print_style[name] % values[name][-1])
                        .rjust(col_widths[i])
                        for i, name in enumerate(print_order) \
                            if name in values])
        print line

    def _as_dict(self):
        dd = BaseClass._as_dict(self)
        dd.update(**self.values)
        return dd

    def get_values(self):
        return self.values

    def _add_end_history(self):
        end_history = {}
        self["end_history"] = end_history
        for key, hist in self["history"].items():
            end_history[key] = hist[-1]
