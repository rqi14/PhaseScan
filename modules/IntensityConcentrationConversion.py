# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 10:22:59 2020

!!! NOTE: In this script the unit of intensity is per unit volume (length unit is the same as cvt pixel length and
device height). The calculation scripts output intensity per PIXEL volume and it is completely DIFFERENT!!!

This script calculates calibration LUT using capillary images.
Two types of data structure supported.
Type 1: separated capillary images
The images should be named as [Channel name]_[Concentration].[Format], e.g. 488_0.1.tif
If the background images were taken for individual capillary image, they should be named as follows.
The background images should be named as [Channel name]_[Concentration]_bkg.[Format], .e.g 488_0.1_bkg.tif
If background images for individual capillary do no exist, it automatically uses the background image specified in the
config file.
In this mode, the ROI must be selected for each of the images.
If you want to ignore some images, simply move them out of the folder, or rename them in a way that will not be detected
Type 2: combined PDMS images (PDMS device with multiple channels)
The images should be named as follows:
[Channel name]_[Concentration1]_[Concentration2]..._[Concentrationn]_some-extra-info_h[Height of device].[Format]
Note: h[Height of device] is optional for this type of images. If this does not exist, it will automatically use the
height specified in the config file.
Note2: A known issue in name matching is that after the last concentration there must be an underscore. For example,
488_5_0.tif won't work and 488_5_0_.tif will. This is important when no height is specified and no extra info provided
In this mode, if there is some concentration that does not exist, simply press ESC or close the window to skip it.
Type 3: This is for images with quadrants.
The images should be named as follows:
qcvt#[channel_name_1]_[conc_1]_[conc_2]_h[device height]#[ch_2]_[conc_1]_[conc_2]_h[Height]-some-extra-info.[format]
Data of multiple channels can be put in one image.
Note the individual background is not supported in this mode
_h[device height] is optional. If not given, global height in config file will be used.

Final data structure
if ch_names[i] = 488
{488} means the index of 488 in ch_names
in 488 channel value_matrix[{488}] = [list of 488 concentration values]
int_matrix[{488}][{647}] = [list of the 647 intensity with corresponding 488 concentrations]

value_matrix [[1,2,3],[4,5,6,7,8]]
int_matrix [[[1,2,3],[4,5,6]], [[1,2,3,4,5],[2,3,4,5,6]]]
Note: only the outer [] means list, the inner ones are all np.ndarray as they meet the dimension requirements.

@author: qirun
"""
import copy
import os
import pathlib
import pickle
import re
from typing import Iterable

from enum import Enum, auto
from typing import List, Tuple, Callable, Union
import logging

import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

from modules.EvenLuminescence import LuminescenceCorrector
from modules.ParamLoading import ParamLoader
from modules.PolygonSelection import PolygonSelector
from modules.proc_utils import imreadmulti_mean
from abc import abstractmethod


# import warnings
# warnings.filterwarnings('error')


class IntensityConcentrationConverterBase:
    value_matrix = None
    int_matrix = None
    ch_names = None
    interp_matrix = None
    fix_zero = False  # mapping relation intercept fixed at zero
    linear = False  # linear assumption

    def __init__(self):
        self.value_matrix = None
        self.int_matrix = None
        self.ch_names = None
        self.interp_matrix = None

    def _get_default_init_values_gen(self):
        num_frac = 5
        mx = np.array([j.max() for j in self.value_matrix])
        mn = np.array([j.min() for j in self.value_matrix])
        diff = mx - mn
        return (mx - diff / num_frac * i for i in range(num_frac + 1))

    def cvt(self, int_values: np.ndarray, pixel_size, init_values=None):
        """
        Convert intensity values to concentrations
        :param int_values: intensity values (intensity per pixel)
        :param pixel_size: pixel size of the pixel in int_values
        :param init_values: array of initial values for fsolve. By default use maximum values in the cvt images
        :return: ndarray of root
        """
        if self.interp_matrix is None:
            self.force_gen_interp_matrix()
        sol_func = lambda iv: fsolve(lambda x: self._eqn_matrix(x, int_values / pixel_size ** 3, self.interp_matrix),
                                     iv, full_output=True)
        if init_values is not None:
            sol, _, flag, _ = sol_func(init_values)
            if flag == 1:
                return sol
        iv_gen = self._get_default_init_values_gen()
        for ivs in iv_gen:
            sol, _, flag, _ = sol_func(ivs)
            if flag == 1:
                return sol
        sol = [None for i in range(int_values.shape[0])]
        print(f'cvt: solution not found for intensity values {str(int_values)}')
        return sol

    @staticmethod
    def _eqn_matrix(x, y, interp_mat):
        rt = np.zeros(x.shape)  # x is concentration, y is intensity. Ax=b. Generate a b matrix here.
        for i in range(len(x)):  # j is dye and i is channel intensity
            for j in range(len(x)):
                rt[i] += interp_mat[j][i](x[j])  # add each contribution of component i
            rt[i] -= y[i]  # minux matrix b to make it zero for fsolve.
        return rt

    # def force_gen_interp_matrix(self):
    #     interp_matrix = [[None for j in range(len(self.ch_names))] for i in range(len(self.ch_names))]
    #
    #     self.interp_matrix = interp_matrix

    def force_gen_interp_matrix(self):
        interp_matrix = [[None for j in range(len(self.ch_names))] for i in range(len(self.ch_names))]
        if self.linear:
            for i in range(len(self.ch_names)):  # i is dye and j is channel
                for j in range(len(self.ch_names)):
                    if self.fix_zero:
                        # find zero in given data
                        zero_idx_array = np.where(self.value_matrix[i] == 0)[0]
                        if zero_idx_array.shape[0] == 0:
                            interception = None
                        else:
                            interception = self.int_matrix[i][j][zero_idx_array[0]]
                    else:
                        interception = None
                        # i is dye and j is channel
                    # noinspection PyTypeChecker
                    interp_matrix[i][j] = LinearFit(np.array(self.value_matrix[i]).astype(np.float),
                                                    np.array(self.int_matrix[i][j]), interception)

        else:
            for i in range(len(self.ch_names)):
                for j in range(len(self.ch_names)):
                    # i is dye and j is channel
                    # noinspection PyTypeChecker
                    interp_matrix[i][j] = interp1d(np.array(self.value_matrix[i]).astype(np.float),
                                                   np.array(self.int_matrix[i][j]), fill_value="extrapolate")
        self.interp_matrix = interp_matrix

    @staticmethod
    def _get_shaded_sum(signal: np.ndarray, polygon_points) -> Tuple:
        mask = np.zeros(signal.shape)
        cv2.fillPoly(mask, np.array([polygon_points]), 1)
        summed = np.sum(signal[mask != 0])
        pixel_count = np.sum(mask)
        return summed, pixel_count

    @abstractmethod
    def serialise(self, path, base_only):
        pass


class IntensityConcentrationConverter(IntensityConcentrationConverterBase):
    class Mode(Enum):
        AUTO = auto()
        SEPARATE = auto()
        COMBINED = auto()
        QUADRANT = auto()

    def __init__(self, root_dir, ch_names: List, ch_lumi_corr_objs: List, depth, pixel_size, *, bg_suffix='_bkg',
                 ch_bkg_paths=None, suppress_no_bkg_warning=False, mode=Mode.AUTO, delay_run=False, padding_zero=True,
                 linear=False, fix_zero=True, quadrant_split: Callable[[np.ndarray], Iterable[np.ndarray]] = None,
                 **kwargs):
        """
        Initialise an intensity to concentration converter. It supports two types of images, either each image contains
        one well with one concentration, or one image for each channel with multiple wells and different concentrations.
        The background can be specified in two ways. Either specifying the background for each channel by inputting a
        list of background image paths, or set a suffix so the program searches for background images in the same folder
        that have similar names with specified suffix, for example, 488_2.5.tif and 488_2.5_bkg.tif.
        :param root_dir: root dir containing folders with the names of channels
        :param ch_names: channel names
        :param ch_lumi_corr_objs: luminescence corrector objects for each channel
        :param depth: depth of the well
        :param pixel_size: pixel size of the image
        :param bg_suffix: background suffix for background search.
        :param ch_bkg_paths: paths of channel background. For quadrant mode this should be a string not list
        :param suppress_no_bkg_warning: whether the no bkg warning is suppressed
        :param mode: specifying whether it is one image for each channel and concentration (Mode.SEPARATE) or one image
                for each channel but multiple concentrations (Mode.COMBINED). The program can automatically determine
                if mode is set to auto (Mode.AUTO).
        :param delay_run: If False, run() has to be executed before using.
        :param padding_zero: padding_zero zero intensity for zero concentration to avoid negative extrapolation
        :param linear: Use linear fitting mode. If false, use inter- and extrapolation
        :param fix_zero: Fix interception at value = 0 if it exists in the dataset. Only works in linear fitting mode.
        :param quadrant_split: Only and must for quadrant mode. A callable that splits the images to a list of images
                corresponding to channels in ch_names only and in order.
        """
        super().__init__()
        self.root_dir = root_dir
        self.ch_names = ch_names
        self.ch_lumi_corr_objs = ch_lumi_corr_objs
        self.depth = depth
        self.pixel_size = pixel_size
        self.bg_suffix = bg_suffix
        self.suppress_no_bkg_warning = suppress_no_bkg_warning
        self.int_matrix = None  # type:Union[np.ndarray,None]
        self.value_matrix = None  # type:Union[np.ndarray,None]
        self.mode = mode
        self.ch_bkg_paths = ch_bkg_paths

        # quadrant mode compatibility
        self.ch_bkg = None  # type:Union[np.ndarray,None]
        if ch_bkg_paths is None:
            self.ch_bkgs = [None for p in ch_names]
        elif isinstance(ch_bkg_paths, str):  # if str is provided for background
            # if quadrant mode, set bkg path
            # if auto, leave it until mode is decided
            if self.mode == self.Mode.AUTO or self.mode == self.Mode.QUADRANT:
                self.ch_bkg = imreadmulti_mean(ch_bkg_paths)
                self.ch_bkgs = None
            else:  # mode is combined or separated, bkg needs to be expanded
                self.ch_bkgs = self._expand_bkg_to_bkgs(ch_bkg_paths)
        else:
            if len(ch_bkg_paths) != len(ch_names):
                raise IndexError('The length of ch_bkg_paths does not match that of ch_names')
            self.ch_bkgs = [imreadmulti_mean(p) if p is not None else None for p in ch_bkg_paths]

        self.padding_zero = padding_zero
        self.linear = linear
        self.fix_zero = fix_zero
        self.quadrant_split = quadrant_split
        if not delay_run:
            self.run()

    def _expand_bkg_to_bkgs(self, bkg_path):
        return [imreadmulti_mean(bkg_path) for p in self.ch_names]

    def _get_separete_re(self, channel):
        return re.compile(f'^{channel}_(\\d+\\.\\d+|\\d+)\\.\\D\\S{{1,3}}$')

    def _get_combined_re(self, channel, return_depth_re=True):
        combined_re = re.compile(f'^{channel}|_([\\d]+(?:\\.\\d+)?)(?![^_.])|_[\\s\\S].*\\.\\D\\S+')
        if return_depth_re:
            combined_depth_re = re.compile(f'^{channel}|_[\\d]+(?:\\.\\d+)?|\\.\\S+$|h(\\d+(?:\\.\\d+)?)')
            return combined_re, combined_depth_re
        else:
            return combined_re

    # regular expression for matching quadrant filenames
    _quadrant_fn_re = re.compile(r'qcvt#')  # match file name pattern
    _quadrant_section_re = re.compile(r'(#(?:[^_]+)(?:_h?\d+(?:\.\d+)?)+)')  # match info section
    _quadrant_channel_re = re.compile(r'#([^_]+)')  # match channel name in section
    _quadrant_concentration_re = re.compile(r'_(\d+(?:\.\d+)?)')  # match concentration in section
    _quadrant_depth_re = re.compile(r'_h(\d+(?:\.\d+)?)')  # match depth in section

    def _get_channel_re(self):
        """
        Return re object for extracting channel, and re object for extracting the suffix following channel
        :return:
        """
        return re.compile(r'^([^_]+)(_.*)')

    def _auto_mode(self, ch_idx):
        """
        Automate mode selection. If nothing
        :param ch_idx: signal index
        :return: mode enum
        """
        list_root = os.listdir(self.root_dir)
        # if any file named as 'qcvt#...' then go to quadrant mode
        for d in list_root:
            if self._quadrant_fn_re.match(d):
                return self.Mode.QUADRANT
        # from here it should be separated or combined mode
        channel = self.ch_names[ch_idx]
        ch_dir = os.path.join(self.root_dir, str(channel))
        if not (os.path.exists(ch_dir) and os.path.isdir(ch_dir)):
            raise FileNotFoundError('Cvt files or directories not found in the given path. '
                                    'Check the naming of files and directory structure.')
        list_dir = os.listdir(ch_dir)  # list files in ch_dir
        separate_re = self._get_separete_re(channel)
        combined_re = self._get_combined_re(channel, False)
        mode = None

        if self.ch_bkg is not None and self.ch_bkgs is None:  # this means only one bkg present
            self.ch_bkgs = self._expand_bkg_to_bkgs(self.ch_bkg_paths)
        for d in list_dir:
            combined_match = [p for p in combined_re.findall(d) if p]
            if len(combined_match) == 1:
                mode = self.Mode.SEPARATE
                break
            elif len(combined_match) > 1:
                mode = self.Mode.COMBINED
                break
        return mode

    def _get_mode_for_ch(self, ch_idx):
        if isinstance(self.mode, (list, np.ndarray)):
            ch_mode = self.mode[ch_idx]
        else:
            ch_mode = self.mode

        if ch_mode == self.Mode.AUTO:
            return self._auto_mode(ch_idx)
        else:
            return ch_mode

    def _padding_zero(self):
        """
        Insert 0,0 to the dataset
        """
        for ch_idx in range(len(self.ch_names)):
            if np.any(self.value_matrix[ch_idx] == 0):
                continue
            self.value_matrix[ch_idx] = np.insert(self.value_matrix[ch_idx], 0, 0)
            self.int_matrix[ch_idx] = np.insert(self.int_matrix[ch_idx], 0, 0, axis=1)

    def run(self):
        int_matrix = []
        value_matrix = []
        # iterate through channel folders
        for ch_idx in range(len(self.ch_names)):  # ch_idx is the current processing folder index
            ch_mode = self._get_mode_for_ch(ch_idx)
            if ch_mode == self.Mode.SEPARATE:
                ch_ints_arr, ch_values_arr = self._process_ch_dir_separate(ch_idx)
            elif ch_mode == self.Mode.COMBINED:
                ch_ints_arr, ch_values_arr = self._process_ch_dir_combined(ch_idx)
            elif ch_mode == self.Mode.QUADRANT:
                if self.quadrant_split is None:
                    raise ValueError('Quadrant mode is set for cvt but quadrant split callable is not defined')
                ch_ints_arr, ch_values_arr = self._process_ch_dir_quadrant(ch_idx)
            else:
                raise ValueError('Search mode not specified correctly. Check the file naming and structure in the cvt '
                                 'folder')
            """
            # Final data structure
            # if ch_names[i] = 488
            # {488} means the index of 488 in ch_names
            # in 488 channel value_matrix[{488}] = [list of 488 concentration values]
            # int_matrix[{488}][{647}] = [the 647 intensity with specified 488 concentrations]
            """
            int_matrix.append(ch_ints_arr)
            value_matrix.append(ch_values_arr)
        self.int_matrix = int_matrix
        self.value_matrix = value_matrix

        if self.padding_zero:
            self._padding_zero()  # padding zero where zero concentration does not exist.

        # generate interp matrix for future calculation
        self.force_gen_interp_matrix()

    def _process_ch_dir_quadrant(self, ch_idx):
        channel = self.ch_names[ch_idx]
        ch_dir = self.root_dir
        list_dir = os.listdir(ch_dir)
        ch_data_exist_flag = False
        # initialise
        ch_ints_list = []
        ch_values_list = []
        # list all files in the root directory
        for fn in list_dir:
            # Skip if not qcvt files or directory
            if self._quadrant_fn_re.match(fn) is None or os.path.isdir(os.path.join(self.root_dir, fn)):
                continue
            # check file extension, if not img format, skip to next file
            ext = os.path.splitext(fn)[1].lower()
            if not (ext == '.tif' or ext == '.png' or ext == '.png' or ext == '.bmp' or ext == '.gif'):
                continue
            # extract sections for each channel
            sections = self._quadrant_section_re.findall(fn)
            ch_sections = []

            for section in sections:
                ch_sec = self._quadrant_channel_re.findall(section)[0]
                # skip if data not belong to the current channel
                if not ch_sec == channel:  # skip to next section if this section not for this channel
                    continue
                ch_sections.append(section)
            if len(ch_sections) == 0:
                continue  # if image not include info for this channel, go to the next file
            # now it is sure this file name contains sections for this channel
            ch_data_exist_flag = True
            # load image
            ch_img = imreadmulti_mean(os.path.join(ch_dir, fn))
            ch_bkg = self.ch_bkg
            cropped = self.quadrant_split(cv2.subtract(ch_img, ch_bkg))
            signals = [self.ch_lumi_corr_objs[i].correct_img(cropped[i]) for i in range(len(self.ch_names))]
            ch_signal = signals[ch_idx]
            # initialise lists for calculation
            ch_poly_points_list_sf = []  # for single file only
            ch_ints_list_sf = []
            ch_values_list_sf = []
            for section in ch_sections:
                # current channel data found, set flag to true

                # now process the data by mannual selection
                ch_values_in_sec = [float(v) for v in self._quadrant_concentration_re.findall(section) if v]
                # match depth, if not found in the section, use global depth instead
                depth_match = [float(v) for v in self._quadrant_depth_re.findall(section) if v]
                if len(depth_match) == 0:
                    ch_depth = self.depth
                else:
                    ch_depth = depth_match[0]

                # start selection. loop through concentrations in section

                poly_gui = PolygonSelector('', ch_signal)
                for i in range(len(ch_values_in_sec)):
                    poly_gui.window_name = (
                        f'Select for {channel} - {ch_values_in_sec[i]} in {section} of {fn}')
                    poly_gui.run()
                    if poly_gui.is_completed:  # if the selection is completed then add to the result list
                        print(f'Completed for {channel} - {ch_values_in_sec[i]}')
                        poly_points = copy.deepcopy(poly_gui.points)  # retrieve points selected in the polygon selector
                        ch_poly_points_list_sf.append(poly_points)
                        # get shaded sum and number pixels shaded
                        int_pvs = []
                        for k in range(len(self.ch_names)):
                            shaded_sum, pixel_count = self._get_shaded_sum(signals[k], poly_points)
                            int_pv = shaded_sum / (
                                    pixel_count * self.pixel_size ** 2) / ch_depth  # intensity per unit volume
                            int_pvs.append(int_pv)
                        ch_ints_list_sf.append(int_pvs)  # Add to the channel intensity list
                        ch_values_list_sf.append(ch_values_in_sec[i])
                    else:
                        continue
                # Check if in the image there is any polygon selected
                if len(ch_values_list_sf) == 0:
                    raise ValueError('No polygon selected')
            ch_values_list.extend(ch_values_list_sf)
            ch_ints_list.extend(ch_ints_list_sf)

        if not ch_data_exist_flag:
            raise FileNotFoundError(f'Conversion data not found for channel {channel}')
        # all files containing info for this channel has been looped and selected. now sort and return
        # Sort values, intensities, poly points in the ascending order of values
        zipped_ch_vi = zip(ch_values_list, ch_ints_list)
        sorted_ch_vi = sorted(zipped_ch_vi)
        tuples_ch_vi = zip(*sorted_ch_vi)
        ch_values_list, ch_ints_list = [list(t) for t in tuples_ch_vi]
        ch_values = np.array(ch_values_list)
        # by definition the row number should be channel and cols are conc.
        # in this case the matrix needs to be transformed
        ch_ints = np.array(ch_ints_list).T
        return ch_ints, ch_values

    def _process_ch_dir_combined(self, ch_idx):
        channel = self.ch_names[ch_idx]
        ch_re, ch_depth_re = self._get_combined_re(channel)
        ch_dir = os.path.join(self.root_dir, str(channel))  # enter channel folder
        list_dir = os.listdir(ch_dir)  # list files in ch_dir
        # find the combined image in this ch_dir
        ch_path = None
        for d in list_dir:
            name_match = ch_re.findall(d)
            if len(name_match) > 1:
                if name_match[0] != "":  # if first match is not "", it means the channel name is not matched
                    continue
                ch_path = os.path.join(ch_dir, d)
                break
        if ch_path is None:
            raise FileNotFoundError(f'main cvt image for {channel} not found')
        ch_signal = self._retrieve_signal(ch_path, self.ch_lumi_corr_objs[ch_idx])

        # select the regions in the order specified in the filename but sort it later
        ch_basename = os.path.basename(ch_path)
        ch_values_in_fn = [float(v) for v in ch_re.findall(ch_basename) if v]
        depth_match = [float(v) for v in ch_depth_re.findall(ch_basename) if v]
        if len(depth_match) == 0:
            ch_depth = self.depth
        else:
            ch_depth = depth_match[0]
        print(f'\nSelecting {ch_path}. \nDepth is {ch_depth}')
        # start selection for each concentration
        ch_values_list = []  # concentration values in filenames of the image for which the poly select is completed
        ch_ints_list = []  # intensity for this channel only
        ch_poly_points_list = []
        poly_gui = PolygonSelector('', ch_signal)
        for i in range(len(ch_values_in_fn)):
            poly_gui.window_name = (f'Select for {channel} - concentration {ch_values_in_fn[i]} in {ch_basename}')
            poly_gui.run()
            if poly_gui.is_completed:  # if the selection is completed then add to the result list
                print(f'Completed for {channel} - {ch_values_in_fn[i]}')
                poly_points = copy.deepcopy(poly_gui.points)  # retrieve points selected in the polygon selector
                ch_poly_points_list.append(poly_points)
                # get shaded sum and number pixels shaded
                shaded_sum, pixel_count = self._get_shaded_sum(ch_signal, poly_points)
                int_pv = shaded_sum / (pixel_count * self.pixel_size ** 2) / ch_depth  # intensity per unit volume
                ch_ints_list.append(int_pv)  # Add to the channel intensity list
                ch_values_list.append(ch_values_in_fn[i])
            else:
                continue
        # Check if in the image there is any polygon selected
        if len(ch_values_list) == 0:
            raise ValueError('No polygon selected')
        # Sort values, intensities, poly points in the ascending order of values
        zipped_ch_vip = zip(ch_values_list, ch_ints_list, ch_poly_points_list)
        sorted_ch_vip = sorted(zipped_ch_vip)
        tuples_ch_vip = zip(*sorted_ch_vip)
        ch_values_list, ch_ints_list, ch_poly_points_list = [list(t) for t in tuples_ch_vip]
        # combine the data from main channel to the rest
        ch_values = np.array(ch_values_list)  # convert ch_values (concentrations) to np.ndarray
        ch_ints = np.zeros((len(self.ch_names), len(ch_values_list)))  # create matrix for intensity contribution
        ch_ints[ch_idx] = np.array(ch_ints_list)  # fill the data in the main channel

        # Then count the rest of the channels
        # First check if other channels exist, otherwise fill with zeros
        ch_itf_ext = os.path.splitext(ch_path)[1]
        # retrieve the string following channel name
        ch_suffix_re = self._get_channel_re()
        ch_suffix = ch_suffix_re.findall(ch_basename)[0][1]
        for ch_idx_itf in range(len(self.ch_names)):  # loop through other channels. itf = interference
            if ch_idx_itf == ch_idx:  # skip main channel
                continue
            ch_itf_basename = self.ch_names[ch_idx_itf] + ch_suffix  # Get basename of the proposed filename of itf
            ch_itf_path = os.path.join(ch_dir, ch_itf_basename)
            # check if other channel exists
            if os.path.exists(ch_itf_path):  # if exists, process it
                ch_itf_signal = self._retrieve_signal(ch_itf_path, self.ch_lumi_corr_objs[ch_idx_itf])
                for i in range(len(ch_values)):  # loop through all polygons selected
                    shaded_sum, pixel_count = self._get_shaded_sum(ch_itf_signal, ch_poly_points_list[i])
                    int_pv = shaded_sum / (pixel_count * self.pixel_size ** 2) / self.depth
                    ch_ints[ch_idx_itf, i] = int_pv
        return ch_ints, ch_values

    def _process_ch_dir_separate(self, ch_idx):
        channel = self.ch_names[ch_idx]
        # ch_re = self._get_separete_re(channel)  # regex to extract value in filename
        ch_re, ch_depth_re = self._get_combined_re(channel)  # use universal regular expression
        ch_dir = os.path.join(self.root_dir, str(channel))  # enter channel folder
        list_dir = os.listdir(ch_dir)  # list files in ch_dir
        ch_ints = [[] for i in range(len(self.ch_names))]
        # ch_values = [[] for i in range(len(self.ch_names))]
        # print(list_dir)

        # get a sorted array of values and filenames (sorted by values in ascending order)
        ch_values = []
        ch_filenames = []
        for i in range(len(list_dir)):
            ch_filename = list_dir[i]
            name_match = ch_re.findall(ch_filename)
            if len(name_match) > 1:
                if name_match[0] != "":  # if first match is not "", it means the channel name is not matched
                    continue
            ch_value = float(name_match[1])  # Value in the filename
            ch_values.append(ch_value)
            ch_filenames.append(ch_filename)

        # sort according to ch_values (concentration) ascending
        zipped_ch_vfn = zip(ch_values, ch_filenames)
        sorted_ch_vfn = sorted(zipped_ch_vfn)
        tuples_ch_vfn = zip(*sorted_ch_vfn)
        ch_values, ch_filenames = [list(t) for t in tuples_ch_vfn]

        for i in range(len(ch_values)):  # iterate through the present values
            # process the main channel (ch_idx)
            ch_value = ch_values[i]  # Value in the filename
            ch_filename = ch_filenames[i]
            depth_match = [float(v) for v in ch_depth_re.findall(ch_filename) if v]
            if len(depth_match) == 0:
                ch_depth = self.depth
            else:
                ch_depth = depth_match[0]
            ch_signal = self._retrieve_signal(os.path.join(ch_dir, ch_filename),
                                              self.ch_lumi_corr_objs[
                                                  ch_idx])  # remove bkg from img_diff, correct luminescence
            # Start selection
            is_completed = False  # Flag for successful selection of ROI
            poly_gui = PolygonSelector(f'Select ROI - {ch_filename}', ch_signal)
            print(f'\nSelecting {os.path.join(ch_dir, ch_filename)}. \nDepth is {ch_depth}')
            while not is_completed:
                poly_gui.run()
                is_completed = poly_gui.is_completed
            poly_points = poly_gui.points  # retrieve points selected in the polygon selector
            shaded_sum, pixel_count = self._get_shaded_sum(ch_signal,
                                                           poly_points)  # get shaded sum and number pixels shaded
            int_pv = shaded_sum / (pixel_count * self.pixel_size ** 2) / ch_depth  # intensity per unit volume
            ch_ints[ch_idx].append(int_pv)  # Add to the channel intensity list
            # process other channels
            ch_itf_ext = os.path.splitext(ch_filename)[1]  # itf = interference
            for ch_idx_itf in range(len(self.ch_names)):
                if ch_idx_itf == ch_idx:
                    continue
                channel_itf = self.ch_names[ch_idx_itf]  # name of the channel
                ch_itf_filename = self._gen_ch_filename(channel_itf, ch_value, ch_itf_ext)
                ch_itf_full_path = os.path.join(ch_dir, ch_itf_filename)
                if not os.path.exists(ch_itf_full_path):
                    logging.warning(f"{ch_itf_full_path} does not exist! Counting {channel_itf} interference as zero")
                    int_pv = 0
                else:
                    print(ch_idx_itf)
                    ch_itf_signal = self._retrieve_signal(ch_itf_full_path, self.ch_lumi_corr_objs[ch_idx_itf])
                    shaded_sum, pixel_count = self._get_shaded_sum(ch_itf_signal, poly_points)
                    int_pv = shaded_sum / (pixel_count * self.pixel_size ** 2) / self.depth
                ch_ints[ch_idx_itf].append(int_pv)
        return np.array(ch_ints), np.array(ch_values).astype(np.float)

    @staticmethod
    def _split_path(path):
        """
        Split the full path into dir, basename, filename and extension

        Parameters
        ----------
        path : TYPE
            path to be splitted.

        Returns
        -------
        directory : TYPE
            DESCRIPTION.
        basename : TYPE
            DESCRIPTION.
        filename : TYPE
            DESCRIPTION.
        extension : TYPE
            DESCRIPTION.

        """
        directory, basename = os.path.split(path)
        filename, extension = os.path.splitext(basename)
        return directory, basename, filename, extension

    @staticmethod
    def _get_for_a_single_file(path):
        img = imreadmulti_mean(path)
        polys = PolygonSelector('Select ROI', img)
        polys.run()

    def _gen_ch_filename(self, channel, ch_value, ext, is_bkg=False):
        if is_bkg:
            return f'{channel}_{ch_value}{self.bg_suffix}{ext}'
        else:
            return f'{channel}_{ch_value}{ext}'

    def _retrieve_signal(self, ch_full_path, ch_lc: LuminescenceCorrector):
        """
        Retrieve single data image stack and look for corresponding background if exists
        :param ch_full_path: full path of image
        :param ch_lc: corresponding luminescence corrector
        :return:
        """
        ch_dir, ch_basename, ch_filename, ch_ext = self._split_path(ch_full_path)
        ch_bkg_basename = f'{ch_filename}{self.bg_suffix}{ch_ext}'
        ch_bkg_path = os.path.join(ch_dir, ch_bkg_basename)
        ch_ch_re = self._get_channel_re()
        channel = ch_ch_re.findall(ch_basename)[0][0]
        ch_idx = [str(ch) for ch in self.ch_names].index(channel)
        ch_img = imreadmulti_mean(ch_full_path)
        if os.path.exists(ch_bkg_path):  # if bkg per img_diff exists
            ch_bkg = imreadmulti_mean(ch_bkg_path)
        else:  # if bkg per img_diff not exist
            ch_bkg = self.ch_bkgs[ch_idx]
        if ch_bkg is None:
            signal_raw = ch_img
            if not self.suppress_no_bkg_warning:
                logging.warning(f'Background image not exist for {ch_full_path}.')
        else:
            signal_raw = cv2.subtract(ch_img, ch_bkg)
        ch_signal = ch_lc.correct_img(signal_raw)
        return ch_signal

    def serialise(self, path, base_only=True):
        """
        Serialise the object and save it as pickle file
        :param path: Saving path
        :param base_only: If true, only base class is saved (without image data)
        """
        pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        with open(path, 'wb+') as handle:
            if base_only:
                p = IntensityConcentrationConverterBase()
                p.value_matrix = self.value_matrix
                p.int_matrix = self.int_matrix
                p.interp_matrix = self.interp_matrix
                p.ch_names = self.ch_names
                p.fix_zero = self.fix_zero
                p.padding_zero = self.padding_zero
                p.linear = self.linear
                pickle.dump(p, handle)
            else:
                pickle.dump(self, handle)

    # def force_gen_interp_matrix(self):
    #     if self.linear:
    #         interp_matrix = [[None for j in range(len(self.ch_names))] for i in range(len(self.ch_names))]
    #         for i in range(len(self.ch_names)):
    #             for j in range(len(self.ch_names)):
    #                 if self.fix_zero:
    #                     # find zero in given data
    #                     zero_idx_array = np.where(self.value_matrix[i]==0)[0]
    #                     if zero_idx_array.shape[0]==0:
    #                         interception = None
    #                     else:
    #                         interception = self.int_matrix[i][j][zero_idx_array[0]]
    #                 else:
    #                     interception = None
    #                 # noinspection PyTypeChecker
    #                 interp_matrix[i][j] = LinearFit(np.array(self.value_matrix[i]).astype(np.float),
    #                                                np.array(self.int_matrix[i][j]), interception)
    #         self.interp_matrix = interp_matrix
    #     else:
    #         super().force_gen_interp_matrix()


class LinearFit:
    def __init__(self, x, y, interception=None):
        self.x = x
        self.y = y
        self.m = None
        self.c = interception
        self._gen_fit()

    def _gen_fit(self):
        if self.c is None:
            A = np.vstack([self.x, np.ones(self.x.shape[0])]).T
            self.m, self.c = np.linalg.lstsq(A, self.y, rcond=None)[0]
        else:
            A = np.vstack(self.x)
            self.m = np.linalg.lstsq(A, self.y - self.c, rcond=None)[0]

    def __call__(self, x):
        return x * self.m + self.c


def cvt_df(df: pd.DataFrame, cvt_obj: IntensityConcentrationConverterBase, params: ParamLoader):
    df_new = df.copy()
    for index, row in df_new.iterrows():
        num_ch = len(cvt_obj.ch_names)
        ints = np.array([row[f'signal_{i}_pv'] for i in range(num_ch)])
        concs = cvt_obj.cvt(ints, params.img_pixel_size)
        for i in range(num_ch):
            df_new.loc[index, f'signal_{i}_pv_cvt'] = concs[i]
    return df_new


class LinearThroughOriginConverter():
    def __init__(self, slope_matrix, channels, buffer_intensities):
        self.slope_matrix = slope_matrix
        self.channels = channels
        self.buffer_intensities = [buffer_intensities[ch] for ch in channels]
        
    def convert(self, channels: List[str], intensities: np.ndarray):
        """Calculates concentrations from measured intensities.
        
        Parameters
        ----------
        channels: List[str]
            List of channel names corresponding to the columns in `intensities`. Must match
            the order of the channels in the `IntensityConcentrationConverterBase` passed to
            `LinearThroughOriginConverter.from_int_conc_converter`.
        intensities : np.ndarray
            Array of shape (stack_size, n_channels) containing the intensity per unit
            volume for each channel across the columns. The order of intensities must 
            match `self.channels`.
            
        Returns
        -------
        np.ndarray
            Array of shape (stack_size, n_channels) containing concentrations for dyes
            matching `self.channels`. Units correspond to the concentrations in the 
            `IntensityConcentrationConverterBase` passed to
            `LinearThroughOriginConverter.from_int_conc_converter`.
        """
        for channel_a, channel_b in zip(channels, self.channels):
            assert (channel_a == channel_b)

        intensities_norm = np.maximum(intensities - self.buffer_intensities, 0)
        ch_len = len(channels)
        slope_stack = np.broadcast_to(self.slope_matrix, (intensities_norm.shape[0], ch_len, ch_len))
        # Solve the system of linear equations `slope_stack`*c=i where we're looking for the concentration vector c and 
        # i is a vector of intensities. To speed things up we solve this for the intensities from all droplets.
        return np.linalg.solve(slope_stack, intensities_norm)


    @staticmethod
    def from_int_conc_converter(int_conc_converter: IntensityConcentrationConverterBase):
        """Creates a converter from an `IntensityConcentrationConverterBase` object.
        
        Parameters
        ----------
        int_conc_converter: IntensityConcentrationConverterBase
            
        Returns
        -------
        LinearThroughOriginConverter
            Converter object, call `convert` to obtain concentrations from intensities.
        """
        channels = int_conc_converter.ch_names
        n_channels = len(channels)
        slope_matrix = np.zeros((n_channels, n_channels))
        concentrations = int_conc_converter.value_matrix
        intensities = np.array(int_conc_converter.int_matrix)
        buffer_intensities = {}
        for (row, channel_wavelength) in enumerate(channels):
            # NOTE the original intensity values in cvt_obj contain buffer intensities for each
            # dye in each channel. With one buffer per channel those values should be the same
            # across dyes, here we take the average.
            buffer_intensity = intensities[:,row,0].mean()
            buffer_intensities[channel_wavelength] = buffer_intensity
            for (col, sample_wavelength) in enumerate(channels):
                conc = concentrations[col]
                intensity = intensities[col][row][1]
                # slope
                slope_matrix[row, col] = max(intensity - buffer_intensity, 0) / np.diff(conc)
        return LinearThroughOriginConverter(slope_matrix, channels, buffer_intensities)
        

def convert_linear(df: pd.DataFrame, cvt_obj: IntensityConcentrationConverterBase, params: ParamLoader):
    n_channels = len(cvt_obj.ch_names)
    converter = LinearThroughOriginConverter.from_int_conc_converter(cvt_obj)
    intensity_stack = df.loc[:, [f'signal_{i}_pv' for i in range(n_channels)]].values / (params.img_pixel_size ** 3)
    concentrations = converter.convert(cvt_obj.ch_names, intensity_stack)
    df_new = df.copy()
    df_new.loc[:, [f'signal_{i}_pv_cvt' for i in range(n_channels)]] = concentrations
    return df_new
