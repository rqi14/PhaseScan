"""
Label features by blurring and subtraction
+ exclude size by radius standard deviation
"""
from typing import List

import numpy as np

from modules.SphereAnalysis2 import SphereAnalyser2


class SphereAnalyser3(SphereAnalyser2):
    def __init__(self, signals: List[np.ndarray], detect_idx: int, feature_idx: int, uniform_idx: int, min_size=15,
                 max_size=45, min_size_stddev=3, max_size_stddev =3, **kwargs):
        super().__init__(signals, detect_idx, feature_idx, uniform_idx, min_size=min_size, max_size=max_size, **kwargs)
        self.min_size_stddev = min_size_stddev
        self.max_size_stddev = max_size_stddev

    def _drop_size(self, radius_lower_threshold, radius_upper_threshold):
        """
        Drop size based on standard deviation
        :param radius_lower_threshold: Number of theta less than average
        :param radius_upper_threshold: Number of theta greater than average
        """

        for index, row in self.df.iterrows():
            if row.radius <= self.min_size or row.radius >= self.max_size:
                self.df.drop(index, inplace=True)
        if self.df.shape[0] == 0:
            return
        radius = self.df['radius']
        std = np.std(radius)
        mean = np.mean(radius)
        min_rad = mean - radius_lower_threshold * std * self.min_size_stddev
        max_rad = mean + radius_upper_threshold * std * self.max_size_stddev
        for index, row in self.df.iterrows():
            if row.radius < min_rad or row.radius > max_rad:
                self.df.drop(index, inplace=True)
