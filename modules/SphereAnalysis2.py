"""
+ Label features by blurring and subtraction
"""
import cv2
import numpy as np
import pandas as pd

from modules.SphereAnalysis import SphereAnalyser
from modules.proc_utils import _parse_8bit
from typing import List


class SphereAnalyser2(SphereAnalyser):
    def __init__(self, signals: List[np.ndarray], detect_idx: int, feature_idx: int, uniform_idx: int,
                 stddev_tolerance: float = 3, **kwargs):

        super().__init__(signals, detect_idx, feature_idx, uniform_idx, **kwargs)
        self.stddev_tolerance = stddev_tolerance

    def label_features(self):
        """
        Label features
        """

        col_name = 'feature'
        feature_df = pd.DataFrame(columns=[col_name])
        stddev_tolerance = self.stddev_tolerance
        blurred = cv2.blur(self.signals[self.feature_idx], (3, 3))
        diff = cv2.subtract(self.signals[self.feature_idx], blurred)
        # print(diff.max())
        # print(diff.min())
        _, stddevs = cv2.meanStdDev(diff)
        # print(stddevs[0, 0])
        for index, row in self.df.iterrows():
            feature_df.loc[index, col_name] = self._detect_uneven_circle_blur(diff, row['x'], row['y'],
                                                                              row['radius'], stddev_tolerance)

        # self.df = self.df.join(feature_df)
        self.df = feature_df.combine_first(self.df)
    def _detect_uneven_circle_blur(self, img_diff, x, y, radius, stddev_tolerance: float = 3):
        roi = _parse_8bit(img_diff)
        img_h, img_w = img_diff.shape
        _, stddevs = cv2.meanStdDev(roi)
        mask = self._create_mask_for_blur_enhancement(x, y, radius, img_w, img_h).astype(np.bool)
        roi[np.bitwise_not(mask)] = 0
        _, binary = cv2.threshold(roi, stddev_tolerance * stddevs[0, 0], 255, cv2.THRESH_BINARY)
        if np.any(binary > 0):
            return True
        else:
            return False

    def _create_mask_for_blur_enhancement(self, x, y, radius, img_w, img_h):
        return self._create_circle_mask(x, y, radius, img_w, img_h)
