"""
Label features by blurring and subtraction
exclude size by radius standard deviation
+ Ignore centre of droplet
+ Moving stddev_tolerance to find optimal

"""
import numbers
from typing import List

import numpy as np
from scipy.optimize import curve_fit
import cv2
from modules.SphereAnalysis3 import SphereAnalyser3
import warnings

from modules.proc_utils import _parse_8bit, img_preprocess, _parse_2bgr

# warnings.filterwarnings('error')
import matplotlib.pyplot as plt
import pandas as pd
import platform
# if platform.architecture() == ("64bit", "WindowsPE"):
try:
    from modules.third_party.rolling_ball_modified import subtract_background_rolling_ball
except Exception as err:
    print(f"Failed to load modified rolling ball module. Use CPython instead. {err}")
    from cv2_rolling_ball import subtract_background_rolling_ball
# else:
#     from cv2_rolling_ball import subtract_background_rolling_ball


class SphereAnalyser4(SphereAnalyser3):
    def __init__(self, signals: List[np.ndarray], detect_idx: int, feature_idx: int, uniform_idx: int,
                 cut_centre: float = 0, cut_side: float = 0.05, ignore_bright_pixels: int = 3, uneven_threshold: float = 1, **kwargs):
        super().__init__(signals, detect_idx, feature_idx, uniform_idx, **kwargs)
        # if 'cut_centre' in kwargs:
        #     if not isinstance(kwargs['cut_centre'], numbers.Number):
        #         raise TypeError('cut_centre factor must be a float between 0 and 1')
        #     self.cut_centre = kwargs['cut_centre']
        # else:
        #     self.cut_centre = 0
        self.cut_centre = cut_centre
        self.cut_side = cut_side
        self.ignore_bright_pixels = ignore_bright_pixels
        self.uneven_threshold=uneven_threshold

    def _detect_uneven_circles(self, signal_idx):
        """
        Detect uneven circles in self.df_oi
        :param signal_idx: signal index for detection
        :return: df_oi containing only uneven flags and index (same as self.df_oi)
        """
        col_name = 'uneven'
        uneven_df = pd.DataFrame(columns=[col_name])
        for index, row in self.df.iterrows():
            # print(f'idx {index}')
            # if index == 325:
            #     print('break')
            uneven_df.loc[index, col_name] = self._detect_uneven_circle(self.signals[signal_idx], row['x'], row['y'],
                                                                        row['radius'],
                                                                        cut_side_factor=self.cut_side,
                                                                        feature_threshold=self.uneven_threshold)
        return uneven_df

    def _detect_uneven_circle(self, img: np.ndarray, x, y, radius, ax=0, cut_side_factor=0.05, feature_threshold=0.05):
        """
        detect one uneven circle for uniform check
        :param img: img_diff to be detected
        :param x: bounding box x
        :param y: bounding box y
        :param radius: radius of circle
        :param ax: axis for projection. ax=0 means projection on x axis, =1 means on y axis
        :param cut_side_factor: ignore image edge
        :param feature_threshold: threshold for peak detection
        :return: True for feature detected. None means fitting not successful.
        """
        feature_flag = self._fit_as_3d_sphere(img, x, y, radius, feature_threshold, cut_side_factor, self.cut_centre,
                                              self.ignore_bright_pixels)
        return feature_flag

    # ignore centre
    # @classmethod
    def _detect_uneven_circle_part_cut(self, seq: np.ndarray, cut_side_factor: float, calculated_centre_x):
        """
        Cut both sides of the sequence
        :param seq: sequence
        :param cut_side_factor: cut side factor specifying how much is cutted (0-1)
        :return: cutted sequence
        """
        cut_mask = np.ones(seq.shape, dtype=np.bool)  # default True meaning keep

        cut_range = int(seq.shape[0] * cut_side_factor)
        if cut_range > 0:
            cut_side_mask = np.ones(seq.shape, dtype=np.bool)  # default True meaning keep
            cut_side_mask[int(seq.shape[0] * cut_side_factor):-int(seq.shape[0] * cut_side_factor)] = False  # mask keep
            cut_mask[cut_side_mask] = False  # remove side

        # remove centre from criteria
        if 1 > self.cut_centre >= 0:  # ratio mode
            abs_offset = int(seq.shape[0] * self.cut_centre)
            lower_idx = calculated_centre_x - abs_offset
            upper_idx = calculated_centre_x + abs_offset
        elif self.cut_centre >= 1:
            lower_idx = calculated_centre_x - self.cut_centre
            upper_idx = calculated_centre_x + self.cut_centre
        else:
            return seq[cut_mask]

        lower_idx = int(lower_idx) if lower_idx >= 0 else 0
        upper_idx = int(upper_idx) if upper_idx < seq.shape[0] else seq.shape[0] - 1
        cut_mask[lower_idx:upper_idx] = False
        if np.all(cut_mask == False):
            raise ValueError('All value cut for uneven check')
        return seq[cut_mask]

    @staticmethod
    def _get_cut_centre_abs(cut_centre, radius):
        if 0 <= cut_centre < 1:
            cut_centre_abs = radius * cut_centre
        elif cut_centre >= 1:
            cut_centre_abs = cut_centre
        else:
            raise ValueError('cut_centre cannot be less than 0')
        return cut_centre_abs

    def _fit_as_3d_sphere(self, img: np.ndarray, x, y, radius, threshold, cut_side, cut_centre, ignore_bright_pixels):
        img = cv2.GaussianBlur(img, (0, 0), 3)

        cut_side_factor = cut_side if cut_side < 1 else cut_side / radius
        img_h, img_w = img.shape
        mask = self._create_circle_mask(x, y, radius * (1 - cut_side_factor), img_w, img_h)
        # cut centre
        cut_centre_abs = self._get_cut_centre_abs(cut_centre, radius)
        if cut_centre_abs > 0:
            mask = self._create_circle_mask(x, y, cut_centre_abs, img_w, img_h, color=0, bg=mask)
        mask = mask.astype(np.bool)

        shaded_img = img.copy()
        shaded_img[np.bitwise_not(mask)] = 0

        ys, xs = np.where(mask == True)
        zs = img[ys, xs] / 2  # divide by two to make the sphere centre at z=0 plane

        # r, xc, yc = cls._sphere_fit_3d_fix_z0(xs, ys, zs)
        popt, _, fit_flag = self._sphere_fit_3d_helper(xs, ys, zs, radius, x, y)
        if popt is None:
            return True
        xc, yc, r_fit, prop_const = popt
        # reconstruct data
        # distance to circle centre
        dc = np.sqrt(np.power(xs - xc, 2) + np.power(ys - yc, 2))
        z_model_temp = np.power(r_fit, 2) - np.power(dc, 2)
        z_model_temp[z_model_temp < 0] = np.power(zs[z_model_temp < 0] * prop_const, 2) * 1.0000001
        z_model = np.sqrt(z_model_temp) / prop_const

        # criteria = np.abs((zs - z_model) /z_model.mean())
        criteria = np.abs(
            np.sqrt(np.power(zs * prop_const - 0, 2) + np.power(ys - yc, 2) + np.power(xs - xc, 2)) - r_fit) / r_fit
        criteria[radius - dc < 0.05 * radius] *= 0.1

        debug_c = np.copy(criteria) > threshold
        # criteria_count -= np.sum(z_model[debug_c] * prop_const < radius * 0.05) * 0.8
        # criteria[np.logical_and(debug_c, z_model * prop_const < radius * 0.05)] /= 5
        criteria.sort()
        # self._debug_plot(r_fit, xc, yc, 0, xs,ys,zs/prop_const)
        # self._debug_plot(r_fit, xc, yc, 0, xs, ys, zs * popt[3], debug_c)
        if ignore_bright_pixels > 0:
            criteria = criteria[0:-ignore_bright_pixels]

        # if criteria.max() > self.uneven_threshold:
        #     print('break')
        check_result = criteria.max() > threshold

        # if check_result == False:
        #
        #     self._debug_plot(r_fit, xc, yc, 0, xs, ys, zs * popt[3],debug_c)
        #     print('debug true ')
        return check_result

    @staticmethod
    def _debug_plot(r, x0, y0, z0, xv, yv, zv, debug_c):
        from mpl_toolkits.mplot3d import Axes3D

        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = np.cos(u) * np.sin(v) * r
        y = np.sin(u) * np.sin(v) * r
        z = np.cos(v) * r
        x = x + x0
        y = y + y0
        z = z + z0

        #   3D plot of Sphere
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xv[debug_c==False], yv[debug_c==False], zv[debug_c==False], zdir='z', s=2, c='b', rasterized=True)
        ax.scatter(xv[debug_c], yv[debug_c], zv[debug_c], zdir='z', s=2, c='r', rasterized=True)
        ax.plot_wireframe(x, y, z, color="y")
        # ax.set_aspect('equal')
        # ax.set_xlim3d(-35, 35)
        # ax.set_ylim3d(-35, 35)
        # ax.set_zlim3d(-70, 0)
        ax.set_xlabel('$x$ (mm)', fontsize=16)
        ax.set_ylabel('\n$y$ (mm)', fontsize=16)
        zlabel = ax.set_zlabel('\n$z$ (mm)', fontsize=16)
        plt.show(block=False)
        # plt.savefig('steelBallFitted.pdf', format='pdf', dpi=300, bbox_extra_artists=[zlabel], bbox_inches='tight')

    @classmethod
    def _sphere_fit_3d_helper(cls, x_data, y_data, z_data, radius, x, y):
        xyz = np.column_stack((x_data, y_data, z_data))
        try:
            popt, pcov = curve_fit(cls._sphere_fit_3d_sp, xyz, np.zeros(xyz.shape[0]),
                                   p0=[x, y, radius, z_data.max() / radius],
                                   maxfev=10000)
            return popt, pcov, True
        except RuntimeError:
            return None, None, False

    @staticmethod
    def _sphere_fit_3d_sp(xyz, x0, y0, r, prop_const):
        z0 = 0
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2] * prop_const
        return np.power(x - x0, 2) + np.power(y - y0, 2) + np.power(z - z0, 2) - np.power(r, 2)

    def _create_mask_for_blur_enhancement(self, x, y, radius, img_w, img_h):
        if self.cut_side < 1:
            radius_after_cut = radius * (1 - self.cut_side)
        else:
            radius_after_cut = radius - self.cut_side
        mask = self._create_circle_mask(x, y, radius_after_cut, img_w, img_h)
        cut_centre_abs = self._get_cut_centre_abs(self.cut_centre, radius)  # add centre removal
        mask = self._create_circle_mask(x, y, cut_centre_abs, img_w, img_h, color=0, bg=mask)
        return mask

    def _detect_uneven_circle_blur(self, img_diff, x, y, radius, stddev_tolerance=3):
        roi = img_diff
        img_h, img_w = img_diff.shape
        mask = self._create_mask_for_blur_enhancement(x, y, radius, img_w, img_h).astype(np.bool)
        masked = roi[mask]
        mean, stddevs = cv2.meanStdDev(masked)
        criteria = np.sum(masked > mean + stddev_tolerance * stddevs[0, 0])

        debug_img0 = img_preprocess(img_diff.copy(),1,99, log=False)
        debug_img0[mask==False] = 0
        debug_img = debug_img0.copy()
        debug_img = _parse_2bgr(debug_img)
        debug_img[np.logical_and(mask, roi > mean + stddev_tolerance * stddevs[0, 0]), 0] = 0
        debug_img[np.logical_and(mask, roi > mean + stddev_tolerance * stddevs[0, 0]), 1] = 0
        debug_img[np.logical_and(mask,roi > mean + stddev_tolerance * stddevs[0, 0]),2] = 255
        if criteria > self.ignore_bright_pixels:
            return True
        else:
            return False

    def enhance_img(self, img: np.ndarray):
        img_1 = img_preprocess(img, log=True, dtype=img.dtype)
        # print('rolling ball')
        img_2, background = subtract_background_rolling_ball(_parse_8bit(img_1), self.max_size, light_background=False,
                                                             use_paraboloid=False, do_presmooth=True)
        # img_2 = _parse_8bit(img_1)
        img_3 = img_preprocess(img_2, log=False, dtype=img_2.dtype)

        # img_4 = cv2.morphologyEx(img_3, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))
        img_5 = img_preprocess(img_3, log=True, dtype=img_3.dtype)
        img_6 = img_preprocess(img_5, 10, 90, log=False, dtype=img_5.dtype)
        return img_6

    def label_features(self):
        """
        Label features
        """

        col_name = 'feature'
        feature_df = pd.DataFrame(columns=[col_name])
        stddev_tolerance = self.stddev_tolerance
        blurred = cv2.GaussianBlur(self.signals[self.feature_idx], (3,3),0)
        diff = cv2.subtract(self.signals[self.feature_idx], blurred)
        for index, row in self.df.iterrows():
            feature_df.loc[index, col_name] = self._detect_uneven_circle_blur(diff, row['x'], row['y'],
                                                                              row['radius'], stddev_tolerance)
        self.df = feature_df.combine_first(self.df)
