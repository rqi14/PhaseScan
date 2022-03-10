# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 12:21:19 2020

@author: qirun
"""
import numpy as np
from modules.proc_utils import map_array


class LuminescenceCorrector:
    def __init__(self, calib_img: np.ndarray):
        """
        Luminescence correct object

        Parameters
        ----------
        calib_img : np.ndarray
            Calibration image. Image of a homogeneous sample.

        Returns
        -------
        None.

        """
        self._correct_mat = calib_img / np.mean(calib_img)
        self._correct_mat[
            self._correct_mat == 0] = 1e30  # turn off pixels which have 0 values because they are probably out of boundary
        self._w, self._h = calib_img.shape

    @property
    def w(self):
        return self._w

    @property
    def h(self):
        return self._h

    def correct_img(self, img: np.ndarray):
        """
        Correct image

        Parameters
        ----------
        img : np.ndarray.
            Image to be corrected.

        Returns
        -------
        np.ndarray
            Corrected image.

        """
        return (img / self._correct_mat).astype(img.dtype)
