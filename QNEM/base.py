# -*- coding: utf-8 -*-
# Author: Simon Bussy <simon.bussy@gmail.com>

from datetime import datetime
import json
import numpy as np
from time import time

class Dict(object):
    """
    A class that encompasses a dict.
    Only used to get both attribute and keyword access to fields.
    """
    def __init__(self, **kwargs):
        # This might seem weird, but must be done this way since we
        # overload __setattr__ below
        object.__setattr__(self, "_dict", {})
        self._update(**kwargs)

    def _update(self, **kwargs):
        self._dict.update(**kwargs)
        for key, val in kwargs.items():
            object.__setattr__(self, key, val)

    def __setattr__(self, key, val):
        self._update(**{key: val})

    def __setitem__(self, key, val):
        self._update(**{key: val})

    def __getitem__(self, key):
        return self._dict[key]

class BaseClass:
    """The base class of many objects
    """

    def __init__(self, **kwargs):
        self.params = Dict()
        class_name = self.__class__.__name__
        self._info = Dict()
        # Next line is weird because class is a reserved keyword
        self._set_info(**{"class": class_name})
        self.set_params(**kwargs)

    @staticmethod
    def _get_now():
        return str(datetime.now()).replace(" ", "_").replace(":", "-")

    def _as_dict(self):
        dd = {}
        # First parameters
        dd.update(self.get_params())
        # Then info, so that it replace possible similar keys in params
        dd.update(self.get_info())
        return dd

    def __str__(self):
        return json.dumps(self._as_dict(), sort_keys=True, indent=2)

    def set_params(self, **kwargs):
        """Set parameters"""
        self.params._update(**kwargs)
        return self

    def get_params(self, kw=None):
        """
        Parameters
        ----------
        kw: str, default=None
            The name of the parameter to be returned. If None, the dict
            of all parameters is returned

        Returns
        -------
        output: object
            either the required parameter of the dict of all parameters
        """
        if kw is None:
            return self.params._dict
        else:
            return self.params[kw]

    def _set_info(self, **kwargs):
        self._info._update(**kwargs)
        return self

    def get_info(self, kw=None):
        """
        Parameters
        ----------
        kw: str, default=None
            The name of the information to be returned. If None, the dict
            of all parameters is returned

        Returns
        -------
        output: object
            either the required parameter of the dict of all information
        """
        if kw is None:
            return self._info._dict
        else:
            return self._info[kw]
