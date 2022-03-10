from typing import List

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from modules.proc_utils import img_preprocess
from collections.abc import Iterable
from typing import Union


class SphereAnalyser:
    def __init__(self, signals: List[np.ndarray], detect_idx: int, feature_idx: int, uniform_idx: int, min_size=15,
                 max_size=45, dark_thresholds=None, **kwargs):
        """
        An object to detect spheres in fluorescence images
        :param signals: A list of signals (background subtracted, pure effective signals)
        :param detect_idx: Index of signal used to detect circles
        :param feature_idx: Index of signal used to detect feature
        :param uniform_idx: Index of signal used to check uniformity
        """
        self.signals = signals
        self.detect_idx = detect_idx
        self.feature_idx = feature_idx
        self._check_signals()
        self.img_h, self.img_w = self.signals[0].shape
        self.img_enhance_params = None
        self.feature_params = None
        self.uniform_check_params = None
        self.reset_img_enhance_params()
        self.reset_feature_params()
        self.reset_uniform_check_params()
        self.df = None
        self.uniform_idx = uniform_idx
        self.min_size = min_size
        self.max_size = max_size
        self.img_detect = None

        # Load analyser parameters
        # load dark thresholds
        if isinstance(dark_thresholds, Iterable):
            self.dark_threshold = dark_thresholds
        elif dark_thresholds is None:
            self.dark_threshold = [0 for i in range(len(self.signals))]
        else:
            self.dark_threshold = [dark_thresholds for i in range(len(self.signals))]

    def _check_signals(self):
        if len(self.signals) == 0:
            raise IndexError('No signal provided')
        if self.detect_idx > len(self.signals) - 1 or self.detect_idx < 0:
            raise IndexError('Index for detection is invalid')
        if self.feature_idx > len(self.signals) - 1 or self.feature_idx < 0:
            raise IndexError('Index for feature is invalid')
        if len(self.signals) > 2:
            shape = self.signals[0].shape
            for i in range(1, len(self.signals)):
                if self.signals[i].shape != shape:
                    raise ValueError(f'Signal {i} does not have the same shape as signal 0')
                if len(self.signals[i].shape) != 2:
                    raise ValueError(f'Signal {i} is not grey-scale')

    def reset_img_enhance_params(self):
        """
        Set img_enhance_params to default values

        'logarithm': True,
        'thresh_high_percentile_1': 100,
        'thresh_low_percentile_1': 10,
        'thresh_low_percentile_2': 10}
        """
        img_enhance_params = {'logarithm': True,
                              'thresh_high_percentile_1': 100,
                              'thresh_low_percentile_1': 10,
                              'thresh_low_percentile_2': 0}
        self.img_enhance_params = img_enhance_params

    def reset_feature_params(self):
        """
        Set feature_params to default values

        # 'shrink_factor': 0.92,
        'cut_side': 0.05,
        'feature_thresh': 0.05
        """
        feature_params = {  # 'shrink_factor': 0.92,
            'cut_side': 0.05,
            'feature_thresh': 0.05}
        self.feature_params = feature_params

    def reset_uniform_check_params(self):
        """
        Set uniform_check_params to default values

        'shrink_factor': 0.92,
        'cut_side': 0.05,
        'feature_thresh': 0.05
        """
        uniform_check_params = {  # 'shrink_factor': 0.92,
            'cut_side': 0.05,
            'feature_thresh': 0.05}
        self.uniform_check_params = uniform_check_params

    def run(self):
        self.analyse_as_circles()
        # label
        self.label_circles()

    def analyse_as_circles(self):
        """
        Analyse the signals and export result to self.df_oi
        """
        # detect circles in the image
        self._find_rois()
        # Drop small and outside boundary
        self._drop_size(self.min_size, self.max_size)
        self._drop_outside_boundary()
        # Then quantify them
        self._quantify_all_circles()

    def label_circles(self):
        # label dark circles
        self.label_dark()
        # Check uniform and abandon non-uniform
        self.label_uneven()
        # Label them according to phase separation
        self.label_features()

    def _find_rois(self):
        signal_detect = self.signals[self.detect_idx]
        self.img_detect = self.enhance_img(signal_detect)
        # img_detect=signal_detect.copy()
        self.df = self._fit_circles(self.img_detect)

    def enhance_img(self, img: np.ndarray):
        """
        enhance image for binary process and detection
        :param img: image to be enhanced
        """
        return cv2.medianBlur(img_preprocess(img,
                                             self.img_enhance_params['thresh_low_percentile_1'],
                                             self.img_enhance_params['thresh_high_percentile_1'],
                                             self.img_enhance_params['thresh_low_percentile_2'],
                                             self.img_enhance_params['logarithm']),
                              5)

    def _quantify_all_circles(self):
        """
        Quantify all the circles, calculate the intensity sum and intensity per pixel volume (pv) assuming they are
        spheres. Modify self.df_oi so that signal_{signal_indices}_sum and _pv are added.
        """
        img_h, img_w = self.signals[0].shape
        for index, row in self.df.iterrows():
            mask = self._create_circle_mask(row['x'], row['y'], row['radius'], img_w, img_h).astype(np.bool)
            for signal_idx in range(len(self.signals)):
                col_name_sum = f'signal_{signal_idx}_sum'
                col_name_pv = f'signal_{signal_idx}_pv'
                signal_sum = np.sum(self.signals[signal_idx][mask])
                signal_pv = signal_sum / (4 / 3 * np.pi * np.power(row['radius'], 3))
                self.df.loc[index, col_name_sum] = signal_sum
                self.df.loc[index, col_name_pv] = signal_pv

    def label_features(self):
        """
        Label features
        """
        feature_df = self._detect_uneven_circles(self.feature_idx)
        feature_df.columns = ['feature']
        self.df = self.df.join(feature_df)

    def _detect_uneven_circles(self, signal_idx):
        """
        Detect uneven circles in self.df_oi
        :param signal_idx: signal index for detection
        :return: df_oi containing only uneven flags and index (same as self.df_oi)
        """
        col_name = 'uneven'
        uneven_df = pd.DataFrame(columns=[col_name])
        for index, row in self.df.iterrows():
            uneven_df.loc[index, col_name] = self._detect_uneven_circle(self.signals[signal_idx], row['x'], row['y'],
                                                                        row['radius'],
                                                                        cut_side_factor=self.uniform_check_params[
                                                                            'cut_side'],
                                                                        feature_threshold=self.uniform_check_params[
                                                                            'feature_thresh'])
        return uneven_df

    def label_uneven(self):
        """
        Drop all uneven droplets by detecting unevenness in sigals[uniform_idx]
        """
        uneven_df = self._detect_uneven_circles(self.uniform_idx)
        uneven_df.columns = ['uneven']
        self.df = self.df.join(uneven_df)

    def label_dark(self):
        self._label_dark(range(len(self.signals)), self.dark_threshold)

    def _label_dark(self, signal_indices: Union[float, Iterable], threshold: Union[float, Iterable]):
        dark_df = pd.DataFrame(columns=['dark'])
        if not isinstance(signal_indices, Iterable):
            signal_indices = [signal_indices]
        if not isinstance(threshold, Iterable):
            threshold = [threshold]
        for index, row in self.df.iterrows():
            pa = np.array(
                [row[f'signal_{signal_idx}_sum'] / row['radius'] ** 2 / np.pi for signal_idx in signal_indices])
            if np.any(pa < threshold):
                dark_df.loc[index, 'dark'] = True
            else:
                dark_df.loc[index, 'dark'] = False
        self.df = self.df.join(dark_df)

    def _drop_uneven(self):
        """
        Drop all uneven droplets by detecting unevenness in sigals[uniform_idx]
        """
        uneven_df = self._detect_uneven_circles(self.uniform_idx)
        uneven_df = uneven_df[uneven_df.uneven]
        for index, row in uneven_df.iterrows():
            self.df.drop(index, inplace=True)

    def _drop_dark_per_area(self, signal_idx, threshold):
        for index, row in self.df.iterrows():
            pa = row[f'signal_{signal_idx}_sum'] / row['radius'] ** 2
            if pa < threshold:
                self.df.drop(index, inplace=True)

    def _drop_size(self, radius_lower_threshold, radius_upper_threshold):
        for index, row in self.df.iterrows():
            if row.radius < radius_lower_threshold or row.radius > radius_upper_threshold:
                self.df.drop(index, inplace=True)

    def _drop_outside_boundary(self):
        img_h, img_w = self.signals[0].shape
        for index, row in self.df.iterrows():
            if row.x - row.radius < 0 \
                    or row.y - row.radius < 0 \
                    or row.x + row.radius > img_w \
                    or row.y + row.radius > img_h:
                self.df.drop(index, inplace=True)

    @staticmethod
    def _sphere_fit(x_val, radius, x_offset, propt_const):
        # print(f'radius {radius}, x_offset: {x_offset}, propt_const:: {propt_const}')
        return propt_const ** 2 * (np.power(radius, 2) - np.power(x_val - x_offset, 2))

    @staticmethod
    def _create_circle_mask(x, y, radius, img_w, img_h, color=1, bg=None):
        if bg is None:
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
        else:
            mask = np.copy(bg)
        cv2.circle(mask, (int(x), int(y)), int(radius), color, -1)
        return mask

    # @classmethod
    def _detect_uneven_circle(self, img: np.ndarray, x, y, radius, ax=0, cut_side_factor=0.05, feature_threshold=0.05):
        """
        :param img: img_diff to be detected
        :param x: bounding box x
        :param y: bounding box y
        :param radius: radius of circle
        :param ax: axis for projection. ax=0 means projection on x axis, =1 means on y axis
        :param cut_side_factor: ignore image edge
        :param feature_threshold: threshold for peak detection
        :return: True for feature detected. None means fitting not successful.
        """

        popt, pcov, fit_flag, x_data, y_data = self._detect_uneven_circle_part_fit_sphere(img, x, y, radius, ax)
        if not fit_flag:
            return None

        # radius = popt[0]
        y_model = self._detect_uneven_circle_part_get_model_data(x_data, y_data, popt)
        criteria_seq = self._detect_uneven_circle_part_seq_diff(y_data, y_model)
        criteria_seq = self._detect_uneven_circle_part_cut_side(criteria_seq, cut_side_factor)
        feature_flag = self._detect_uneven_circle_part_check_criteria(criteria_seq, feature_threshold)
        return feature_flag

    @classmethod
    def _detect_uneven_circle_part_fit_sphere(cls, img: np.ndarray, x, y, radius, ax=0):
        """
        Return the fitting result, fitting target as sphere in 3D space with depth.
        :param img: whole img_diff
        :param x: coordinate x of sphere in img_diff
        :param y: coordinate y of sphere in img_diff
        :param radius: radius of sphere
        :param ax: axis for summing up to 1D
        :return: popt, pcov, flag, x_data, y_data. Flag is True if fitting is successful and is False if unsuccessful
        """
        img_h, img_w = img.shape
        mask = cls._create_circle_mask(x, y, radius, img_w, img_h).astype(np.bool)
        shaded_img = img.copy()
        shaded_img[np.bitwise_not(mask)] = 0
        summed1d = np.trim_zeros(np.sum(shaded_img, axis=ax)) / np.trim_zeros(np.sum(mask, axis=ax))
        # Sphere fit
        y_data = summed1d
        x_data = np.array(range(summed1d.shape[0]))
        n = x_data.shape[0]  # the number of pixels on one axis

        if n <= 1:
            return None
        try:
            popt, pcov = curve_fit(cls._sphere_fit, x_data, np.power(y_data, 2),
                                   p0=[radius, summed1d[int(summed1d.shape[0] / 2)] / (2 * radius), radius],
                                   maxfev=10000)
            return popt, pcov, True, x_data, y_data
        except RuntimeError:
            return None, None, False, None, None

    @classmethod
    def _detect_uneven_circle_part_get_model_data(cls, x_data, y_data, popt):
        """
        Generate model data based on sphere fit results
        :param x_data: real x data
        :param y_data: real y data
        :param popt: popt (fitting)
        :return: model values of y
        """
        y2_temp = cls._sphere_fit(x_data, *popt)
        y2_temp[y2_temp < 0] = np.power(y_data[y2_temp < 0],
                                        2)  # when y2^2 (predicted) less than 0, use y data instead
        y2 = np.sqrt(y2_temp)
        return y2

    @classmethod
    def _detect_uneven_circle_part_seq_diff(cls, data_values, model_values):
        """
        A method to find the sequence difference in data and model values. Percentage diff with respect to model values.
        The difference must be positive (data must be greater than models)
        :param data_values: data values
        :param model_values: model values
        :return: a sequence representing the difference in the same length as both
        """
        diff = (data_values - model_values) / model_values
        diff[diff < 0] = 0  # only look for peaks
        return diff

    # @classmethod
    def _detect_uneven_circle_part_cut_side(self, seq, cut_side_factor):
        """
        Cut both sides of the sequence
        :param seq: sequence
        :param cut_side_factor: cut side factor specifying how much is cutted (0-1)
        :return: cutted sequence
        """
        cut_range = int(seq.shape[0] * cut_side_factor)
        if cut_range > 0:
            return seq[int(seq.shape[0] * cut_side_factor):-int(seq.shape[0] * cut_side_factor)]
        else:
            return seq

    @classmethod
    def _detect_uneven_circle_part_check_criteria(cls, seq, threshold):
        return np.max(seq) > threshold

    @staticmethod
    def _fit_circles(img: np.ndarray) -> pd.DataFrame:
        """
        Fit bright circles in the image
        :param img: signal image with bright circles to be fitted
        :return: data frame
        """
        # Threshold the image for morphological closing (to complete filled circles)
        # th is threshold value,
        _, img_binary = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        img_morph_closed = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel_close)
        dist_transform = cv2.distanceTransform(img_morph_closed, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        radius_estimate = 40
        dist_transform_with_border = cv2.copyMakeBorder(dist_transform, radius_estimate, radius_estimate,
                                                        radius_estimate, radius_estimate,
                                                        cv2.BORDER_ISOLATED, 0)
        shrink_radius = 10
        circle_template_binary = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                           (2 * (radius_estimate - shrink_radius) + 1,
                                                            2 * (radius_estimate - shrink_radius) + 1))
        circle_template_binary = cv2.copyMakeBorder(circle_template_binary, shrink_radius, shrink_radius, shrink_radius,
                                                    shrink_radius, cv2.BORDER_ISOLATED, 0)
        circle_template_dist_transform = cv2.distanceTransform(circle_template_binary, cv2.DIST_L2,
                                                               cv2.DIST_MASK_PRECISE)
        # template matching If input image is of size (WxH) and template image is of size (wxh), output image will have
        # a size of (W-w+1, H-h+1).
        # It returns a grayscale image, where each pixel denotes how much does the neighbourhood of that pixel match
        # with template.
        matching_pixel_count = cv2.matchTemplate(dist_transform_with_border, circle_template_dist_transform,
                                                 cv2.TM_CCOEFF_NORMED)
        # _, mx, _, _ = cv2.minMaxLoc(matching_pixel_count)
        # threshold the upper 25% matching pixels.
        _, peaks_mask = cv2.threshold(matching_pixel_count, 0.5, 255, cv2.THRESH_BINARY)
        peaks_mask_uint8 = cv2.convertScaleAbs(peaks_mask)
        contours, _ = cv2.findContours(peaks_mask_uint8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # result_df = pd.DataFrame(columns=['x', 'y', 'radius'])
        result_list = []
        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            _, mx, _, mxloc = cv2.minMaxLoc(dist_transform[y:y + h, x:x + w], peaks_mask_uint8[y:y + h, x:x + w])
            # cv2.circle(im_enhanced, (int(mxloc[0]+x), int(mxloc[1]+y)), int(mx*shrink_factor), (255, 0, 0), 2)
            # cv2.rectangle(im_enhanced, (x, y), (x+w, y+h), (0, 255, 255), 2)
            # cv2.drawContours(im_enhanced, contours, i, (0, 0, 255), 2)
            # result_df = result_df.append({'x': x + mxloc[0],
            #                               'y': y + mxloc[1],
            #                               'radius': mx
            #                               }, ignore_index=True)
            result_list.append({'x': x + mxloc[0],
                                'y': y + mxloc[1],
                                'radius': mx
                                })
        result_df = pd.DataFrame.from_records(result_list)
        return result_df
