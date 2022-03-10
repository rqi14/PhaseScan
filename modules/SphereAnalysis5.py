"""
Improved condensate labelling by bilateral filtering, improved convolution kernel and padding
+ enhancement mode 1: the conventional method adapted for the CoH microscope, 2: adapted for high signal-to-noise ratio
  camera
"""
import numbers
from typing import List
import numpy as np
from scipy.optimize import curve_fit
import cv2
from modules.SphereAnalysis3 import SphereAnalyser3
from modules.proc_utils import _parse_8bit, img_preprocess, _parse_2bgr
import matplotlib.pyplot as plt
import pandas as pd
from modules.SphereAnalysis4 import SphereAnalyser4

try:
    from modules.third_party.rolling_ball_modified import subtract_background_rolling_ball
except Exception as err:
    print(f"Failed to load modified rolling ball module. Use CPython instead. {err}")
    from cv2_rolling_ball import subtract_background_rolling_ball

class SphereAnalyser5(SphereAnalyser4):
    def __init__(self, signals: List[np.ndarray], detect_idx: int, feature_idx: int, uniform_idx: int, *,
                 cut_centre: float = 0, cut_side: float = 0, ignore_bright_pixels: int = 3,
                 uneven_threshold: float = 0.05, feature_threshold=5, feature_padding_pixels=5,
                 feature_padding_threshold=20, feature_smooth_sigma=11, feature_connected_pixels=2,
                 feature_min_cluster_num=1, enhance_mode=1, **kwargs):
        """
        Initiate a sphere analyser 5
        :param signals: Signal images, background subtracted, illumination corrected.
        :param detect_idx: Index of image used for detecting shapes
        :param feature_idx: Index of image used for detecting feature
        :param uniform_idx: Index of image used for checking uniformity
        :param cut_centre: Number of pixels or fraction of radius at centre to be ignored in uniform check
        :param cut_side: Number of pixels or fraction of radius at edge to be ignored in uniform check
        :param ignore_bright_pixels: Number of the brightest or darkest pixels to be ignored in uniform check
        :param uneven_threshold: Threshold for uniform check
        :param feature_threshold: Threshold for feature check
        :param feature_padding_pixels: Number of pixels (extending radius) for padding
        :param feature_padding_threshold: Brightness percentage threshold for padding. Pixels with brightness below this
        value will be ignored for padding
        :param kwargs: key word arguments for compatibilty
        """
        super().__init__(signals, detect_idx, feature_idx, uniform_idx, cut_centre, cut_side, ignore_bright_pixels,
                         uneven_threshold, **kwargs)
        self.feature_threshold = feature_threshold
        self.feature_padding_pixels = feature_padding_pixels
        self.feature_padding_threshold = feature_padding_threshold
        self.feature_kernel = np.array(([-1, -1, -1, -1, -1],
                                        [-1, -1, -1, -1, -1],
                                        [-1, -1, 24, -1, -1],
                                        [-1, -1, -1, -1, -1],
                                        [-1, -1, -1, -1, -1]), dtype="int")
        self.feature_smooth_sigma = feature_smooth_sigma
        self.feature_connected_pixels = feature_connected_pixels
        self.feature_min_cluster_num = feature_min_cluster_num
        self.enhance_mode = enhance_mode

    def label_features(self):
        """
        Label features and fill df
        """
        col_name = 'feature'
        feature_df = pd.DataFrame(columns=[col_name])
        img_smoothed = cv2.bilateralFilter(self.signals[self.feature_idx].astype(np.float32), -1,
                                           self.feature_smooth_sigma, self.feature_smooth_sigma).astype(
            self.signals[self.feature_idx].dtype)
        for index, row in self.df.iterrows():
            detect_result = self._detect_feature(img_smoothed, row['x'],
                                                 row['y'], row['radius'],
                                                 self.feature_threshold,
                                                 self.feature_padding_pixels,
                                                 self.feature_padding_threshold,
                                                 self.feature_kernel,
                                                 self.feature_connected_pixels)
            # detection result 0: feature label 1: num of features 2: mean feature size in pixels
            feature_df.loc[index, col_name], feature_df.loc[index, 'num_of_condensates'], feature_df.loc[
                index, 'average_condensate_size'], feature_df.loc[index, 'vol_of_condensates'] = (
                *detect_result, detect_result[1] * detect_result[2] ** 1.5 / np.power(row.radius, 2))
        self.df = feature_df.combine_first(self.df)

    def _detect_feature(self, img, x, y, radius, feature_threshold, padding_pixels, padding_threshold, kernel,
                        connected_pixels):
        """
        Atomic operation for labelling one droplet.
        :param img: image
        :param x: x coordinate of circle centre
        :param y: y coordinate of circle centre
        :param radius: radius of circule
        :param feature_threshold: threshold value for feature determination
        :param padding_pixels: number of pixels padded around the circle
        :param padding_threshold: threshold for removing dark pixels around the circle before padding
        :param kernel: kernel used for convolution
        :param connected_pixels: Number of connected pixels required for a feature
        :return: true for feature false for no feature; num of features, mean feature sizes in pixels
        """
        xr = round(x)
        yr = round(y)
        rr = round(radius)
        # Get one droplet of interest
        roi_mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.circle(roi_mask, (xr, yr), rr, 255, -1)
        # img8 = _parse_8bit(img)

        roi = img.copy()
        roi[roi_mask == 0] = 0
        roi_isolated = roi[
                       yr - rr:yr + rr + 1,
                       xr - rr:xr + rr + 1]

        # rescale for padding
        rescaled = cv2.resize(roi_isolated,
                              (roi_isolated.shape[1] + 2 * padding_pixels, roi_isolated.shape[0] + 2 * padding_pixels))
        # expand the original roi isolated by adding padding
        expanded = rescaled.copy()
        roi_isolated_mask = np.zeros(roi_isolated.shape, dtype=np.uint8)
        cv2.circle(roi_isolated_mask, (rr + 1, rr + 1), rr, 255, -1)
        roi_iso_trimmed = roi_isolated.copy()
        roi_iso_trimmed[roi_isolated_mask == 0] = 0
        med = np.median(roi_iso_trimmed[roi_iso_trimmed > 0])
        if roi_iso_trimmed.max() == 0:
            return False, 0, 0
        q = np.percentile(roi_iso_trimmed[roi_iso_trimmed > 0], padding_threshold)
        roi_isolated_mask[roi_iso_trimmed < q - (med - q)] = 0
        expanded_mask = cv2.copyMakeBorder(roi_isolated_mask, padding_pixels, padding_pixels,
                                           padding_pixels, padding_pixels, cv2.BORDER_CONSTANT, 0)
        expanded[expanded_mask > 0] = roi_isolated[roi_isolated_mask > 0]
        filtered = cv2.filter2D(expanded, -1, kernel)
        cropped = filtered.copy()
        cropped[expanded_mask == 0] = 0
        criteria_roi = cropped[expanded_mask > 0]
        cmc = np.median(criteria_roi)
        qc = np.percentile(criteria_roi, 75)
        thresh = qc + (qc - cmc) * feature_threshold if qc > 0 else feature_threshold * np.mean(criteria_roi)
        cropped[cropped <= thresh] = 0
        bn = np.zeros(cropped.shape, dtype=np.uint8)
        bn[cropped > 0] = 1
        num_label, ret = cv2.connectedComponents(bn)
        label_count = np.zeros(num_label)
        for i in range(num_label):
            label_count[i] = np.sum(ret == i + 1)
        label_count.sort()

        if label_count[-1] >= connected_pixels:
            # return label, number of condensates, average sizes of condensates
            return True, np.sum(label_count >= connected_pixels), np.mean(label_count[label_count >= connected_pixels])
        else:
            return False, 0, 0

    def enhance_img(self, img: np.ndarray):
        if self.enhance_mode == 1:
            return self.enhance_img_alter_1(img)
        elif self.enhance_mode == 2:
            return self.enhance_img_alter_2(img)
        elif self.enhance_mode == 3:
            return self.enhance_img_alter_3(img)
        else:
            raise ValueError("Enhance mode does not exist")

    def enhance_img_alter_1(self, img: np.ndarray):
        img_1 = img_preprocess(img, log=True, dtype=img.dtype)
        # print('rolling ball')
        img_2, background = subtract_background_rolling_ball(_parse_8bit(img_1), self.max_size, light_background=False,
                                                             use_paraboloid=False, do_presmooth=True)
        # img_2 = _parse_8bit(img_1)
        img_3 = img_preprocess(img_2, log=False, dtype=img_2.dtype)

        # img_4 = cv2.morphologyEx(img_3, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))
        img_5 = img_preprocess(img_3, log=False, dtype=img_3.dtype)
        img_6 = img_preprocess(img_5, 10, 90, log=False, dtype=img_5.dtype)
        return img_6
        # return _parse_8bit(img)

    def enhance_img_alter_2(self, img: np.ndarray):
        img_1 = img_preprocess(img, 10, 90, log=True, dtype=np.uint8)
        return img_1

    def enhance_img_alter_3(self, img:np.ndarray):
        img_1 = img_preprocess(img, 5, 99, log=True, dtype=np.uint8)
        return img_1