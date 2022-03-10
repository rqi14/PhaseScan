# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 11:58:40 2020

@author: qirun
"""

import os
from typing import List, Union

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from modules.ParamLoading import ParamLoader
from modules.utils import to_effective_df

import uuid
from scipy.stats import gaussian_kde
from modules.utils import get_output_path


def map_array(img: np.ndarray, target_dtype, dtype_scale=False, **kwargs) -> np.ndarray:
    """
    Map a matrix (img_diff) to given dtype. If min and max are specified, they will be used as boundaries of img_diff
    :param img:
    :param target_dtype:
    :param dtype_scale: Map the complete range of the old data type. default false: mapping only the range with values
    :param kwargs: Settings:
    :keyword min: Absolute minimum value in the original image that is to be mapped to minimum of new dtype
    :keyword max: Absolute minimum value in the original image that is to be mapped to maximum of new dtype
    :return: ndarray after mapping
    """
    if 'min' in kwargs:
        r_min = kwargs['min']
    else:
        r_min = np.min(img)
    if 'max' in kwargs:
        r_max = kwargs['max']
    else:
        r_max = np.max(img)
    if r_min == r_max:
        dtype_scale = True
    if dtype_scale:
        if np.issubdtype(img.dtype, np.integer):
            info_func = np.iinfo
        else:
            info_func = np.finfo
        r_max = info_func(img.dtype).max
        r_min = info_func(img.dtype).min

    old_range = r_max - r_min
    new_max = np.iinfo(target_dtype).max
    new_min = np.iinfo(target_dtype).min
    new_range = new_max - new_min
    map_factor = new_range / old_range

    mapped = np.clip((img - r_min) * map_factor + new_min, new_min, new_max)

    return np.array(mapped, target_dtype)


def img_preprocess(img: np.ndarray, low_percentile1=0, high_precentile1=100, low_percentile2=0, log=True,
                   dtype=np.uint8) -> np.ndarray:
    """
    Process the image to reduce the heternegeity in brightness
    :param img: Raw image ndarray
    :param low_percentile1: Lower bound percentile in step 1 from 0 to 100
    :param high_precentile1: Upper bound percentile in step 1 from 0 to 100
    :param low_percentile2: lower bound perentile in step 2 from 0 to 100
    :param log: If true, calculate logarithm of image
    :param dtype: target data type
    :return: enhanced image
    """
    img2 = img.copy()
    img2[img2 == 0] = 1
    img2 = map_array(img2, dtype, min=np.percentile(img2, low_percentile1),
                     max=np.percentile(img2, high_precentile1))
    img2[img2 == 0] += 1
    if log:
        img2 = np.log(img2)
    img2 = map_array(img2, dtype, min=np.percentile(img2, low_percentile2))
    return img2


def save_circle_masks(img: np.ndarray, df: pd.DataFrame, directory: str, prefix: str, suffix: str, shrink_factor=1,
                      img_format='.jpg'):
    """
    Save circle masked region for each fitted circle
    :param img: img_diff
    :param df: data frame of img_diff detection result
    :param directory: directory for saving
    :param prefix: prefix before index
    :param suffix: suffix after index
    :param shrink_factor: shrink factor, not used
    :param img_format: image format including dot
    """
    for index, row in df.iterrows():
        circle_mask_prep = np.zeros(img.shape, dtype=np.uint8)
        cv2.circle(circle_mask_prep, (int(row['x']), int(row['y'])),
                   int(row['radius']), (1), -1)
        circle_mask = circle_mask_prep == True

        shaded_img = np.zeros(img.shape, dtype=img.dtype)
        shaded_img[circle_mask] = img[circle_mask]
        shaded_img_uint8 = map_array(shaded_img, np.uint8)
        cv2.imwrite(os.path.join(directory, f"{prefix}{index}{suffix}{img_format}"), shaded_img_uint8)


def plot_detection_result(detected_df: pd.DataFrame, annotate=True, x_name='signal_0_pv', y_name='signal_1_pv',
                          color_name='feature', window_name='plot_detection_result',
                          x_label="channel 0", y_label="channel 1"):
    handle = plt.figure(num=window_name)
    plt.clf()
    plt.cla()
    df_true = detected_df[detected_df[color_name] == True]
    df_false = detected_df[detected_df[color_name] == False]
    scatter_data_true = df_true[[x_name, y_name]].values
    scatter_data_false = df_false[[x_name, y_name]].values
    xt, yt = scatter_data_true.T
    xf, yf = scatter_data_false.T
    plt.scatter(xt, yt, s=2, c="red")
    plt.scatter(xf, yf, s=2, c="blue")
    plt.legend(['Phase separation', 'Mixed'])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if annotate:
        for index, row in detected_df.iterrows():
            plt.annotate(f'{index}', (row[x_name], row[y_name]))
    return handle


def plot_detection_result_3d(detected_df: pd.DataFrame, annotate=True, x_name='signal_0_pv', y_name='signal_1_pv',
                             z_name='signal_2_pv', color_name='feature', window_name='plot_detection_result',
                             x_label="channel 0", y_label="channel 1", z_label="channel 2"):
    fig = plt.figure(num=window_name)
    plt.clf()
    plt.cla()
    ax = fig.add_subplot(111, projection='3d')
    df_true = detected_df[detected_df[color_name] == True]
    df_false = detected_df[detected_df[color_name] == False]
    scatter_data_true = df_true[[x_name, y_name, z_name]].values
    scatter_data_false = df_false[[x_name, y_name, z_name]].values
    xt, yt, zt = scatter_data_true.T
    xf, yf, zf = scatter_data_false.T
    ax.scatter(xt, yt, zt, s=2)
    ax.scatter(xf, yf, zf, s=2)
    ax.legend(['Phase separation', 'Mixed'])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    if annotate:
        for index, row in detected_df.iterrows():
            ax.text(row[x_name], row[y_name], row[z_name], f'{index}', zdir=None)
    return fig


def plot_df_helper(df: pd.DataFrame, params: ParamLoader, annotate=True, window_name=None,
                   label_suffix="signal per pixel volume", name_suffix=None):
    df_plot = to_effective_df(df) 
    if len(params.ch_plot_idx) == 2:
        handle = plot_detection_result(
            df_plot,
            x_name=f'signal_{params.ch_plot_idx[0]}_pv{name_suffix or ""}',
            y_name=f'signal_{params.ch_plot_idx[1]}_pv{name_suffix or ""}',
            x_label=f'{params.channels[params.ch_plot_idx[0]]} {label_suffix}',
            y_label=f'{params.channels[params.ch_plot_idx[1]]} {label_suffix}',
            annotate=annotate, window_name=window_name)
        return handle
    elif len(params.ch_plot_idx) == 3:
        handle = plot_detection_result_3d(
            df_plot,
            x_name=f'signal_{params.ch_plot_idx[0]}_pv{name_suffix or ""}',
            y_name=f'signal_{params.ch_plot_idx[1]}_pv{name_suffix or ""}',
            z_name=f'signal_{params.ch_plot_idx[2]}_pv{name_suffix or ""}',
            x_label=f'{params.channels[params.ch_plot_idx[0]]} {label_suffix}',
            y_label=f'{params.channels[params.ch_plot_idx[1]]} {label_suffix}',
            z_label=f'{params.channels[params.ch_plot_idx[2]]} {label_suffix}',
            annotate=annotate, window_name=window_name)
        return handle
    else:
        return None


def read_combine_detected(excel_path, sort=True):
    """
    Combine True_sheet and False_sheet in an excel file (generated by 
    phasediagram_combi_kit script) and return a new dataframe

    Parameters
    ----------
    excel_path : string
        path of excel file
    sort : boolean, optional
        Not used. The default is True.

    Returns
    -------
    combined_df : pd.DataFrame
        Combined Dataframe.

    """
    true_df = pd.read_excel(excel_path, sheet_name="True_sheet", index_col=0).dropna()
    false_df = pd.read_excel(excel_path, sheet_name="False_sheet", index_col=0).dropna()
    combined_df = true_df.append(false_df)
    if sort:
        combined_df.sort_index(inplace=True)
    return combined_df


def imreadmulti_mean(img_path: str, imread_flags: int = -1) -> np.ndarray:
    """
    Read image stack and return the average

    Parameters
    ----------
    img_path : str
        Path of image stack.
    imread_flags : int, optional
        cv2 imread flags. The default is -1.

    Returns
    -------
    mean_images : np.ndarray
        mean of image stack.

    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f'not found: {img_path}')

    images = cv2.imreadmulti(img_path, flags=imread_flags)[1]
    mean_images = np.mean(images, axis=0).astype(images[0].dtype)
    return mean_images


def _parse_2bgr(img: np.ndarray) -> np.ndarray:
    """
    Convert grey scale to BGR. If not grey scale return a copy of original image.
    Note this function does not convert BGRA images
    :param img:
    :return:
    """
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        return np.copy(img)


def _parse_8bit(img: np.ndarray) -> np.ndarray:
    """
    Try to convert an image to uint8. If the image is uint8, return an copy of the original image
    :param img: Image
    :return: Converted image
    """
    if img.dtype != np.uint8:
        conversion_factor = np.iinfo(img.dtype).max / np.iinfo(np.uint8).max
        return (img / conversion_factor).astype(np.uint8)
        # return map_array(img_diff, np.uint8)
    else:
        return np.copy(img)


def find_slice_idx(sq: np.ndarray, idx: Union[np.ndarray, List]) -> np.ndarray:
    invert_flag = False
    # sq = np.array(range(6))
    # idx = np.array([1, 5, 4])
    sequence = sq.copy()  # move the first point specified in idx as origin
    if not np.all(np.diff(idx) > 0):  # If not clockwise, go anticlockwise by flip the shape and idx
        invert_flag = True
        idx = sq.shape[0] - idx
        idx[idx == sq.shape[0]] = 0
    sequence -= idx[0]
    sequence[sequence < 0] += sequence.shape[0]
    mask = np.logical_and(sequence >= 0, sequence <= np.max(sequence[idx]))
    seq_mask = np.vstack((sequence, mask, sq)).T
    seq_mask_sorted = seq_mask[np.argsort(seq_mask[:, 0])]
    selected = seq_mask_sorted[seq_mask_sorted[:, 1] == True, 2]
    if invert_flag:
        selected = sq.shape[0] - selected
        selected[selected == sq.shape[0]] = 0
    return selected


def plot_density_2d(df, params: ParamLoader, fn="density_2D.png", xlim=None, ylim=None):
    col_suffix = "_cvt" if params.train_on_cvt else ""
    col_names = [f"signal_{i}_pv" + col_suffix for i in params.ch_plot_idx]
    density_handle = plt.figure(num='Density plot ' + str(uuid.uuid4()))
    df_true = df[df.feature == True]
    xy_true = df_true[col_names].to_numpy()
    df_false = df[df.feature == False]
    xy_false = df_false[col_names].to_numpy()
    plt.scatter(xy_true[:, 0], xy_true[:, 1], cmap=plt.get_cmap('Reds'), c=find_density(xy_true), s=2,
                label='Phase separation')
    plt.colorbar()
    plt.scatter(xy_false[:, 0], xy_false[:, 1], cmap=plt.get_cmap('Blues'), c=find_density(xy_false), s=2,
                label='Mixed')
    plt.colorbar()
    # plt.scatter(boundary_x, boundary_y, s=2, c='k')
    plt.legend()
    leg = plt.gca().get_legend()
    leg.legendHandles[0].set_color('red')
    leg.legendHandles[1].set_color('blue')
    # leg.legendHandles[2].set_color('k')
    plt.xlabel(f'{params.channels[0]} Concentration')
    plt.ylabel(f'{params.channels[1]} Concentration')
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.show(block=False)
    density_handle.savefig(os.path.join(get_output_path(params), fn))
    return density_handle


# Extra plot with density
def find_density(xy):
    xy_h = xy.T  # two horizontal rows
    return gaussian_kde(xy_h)(xy_h)
