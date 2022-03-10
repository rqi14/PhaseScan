import numpy as np

import sys

import numpy as np
import pandas as pd
import cv2
from modules.ParamLoading import ParamLoader
from modules.utils import get_output_df_path, argv_proc
# from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.widgets
from modules.proc_utils import _parse_8bit
import json


# def main(argv=None):
#     ag = argv_proc(argv, sys.argv)
#     params = ParamLoader(ag[0])


def crop_quadrant(img_src, raw_index, quadrant_align_info):
    if raw_index == quadrant_align_info.ref_idx_qpos:
        return _crop(img_src, quadrant_align_info.qpos_denorm(img_src.shape)[raw_index])
    elif raw_index in quadrant_align_info.warp_idx_qpos.keys():
        im = _crop(img_src, quadrant_align_info.qpos_denorm(img_src.shape)[raw_index])
        return cv2.warpAffine(im, quadrant_align_info.warp_idx_qpos[raw_index],
                              quadrant_align_info.dsize(img_src.shape))
    else:
        raise IndexError(f"Index {raw_index} does not exist in quadrant align info")
    # return warped


def get_cropped_img_set(full_img: np.ndarray, params: ParamLoader):
    """
    Get cropped img set (only for valid channels that are not None specified in 'channels')
    :param full_img: full image with four quadrant
    :param params: params object
    :return: list of cropped images
    """
    chs_raw = params.channels_raw
    img_h, img_w = full_img.shape
    q_rects = params.quadrant_align_info
    # q_rects = [((rect[0][0] * img_w, rect[0][1] * img_h), (rect[1][0] * img_w, rect[1][1] * img_h), rect[2]) for
    #            rect in q_rects_norm]
    ch_idx_of_interest = [i for i in range(len(chs_raw)) if chs_raw[i] is not None]
    return [crop_quadrant(full_img, r, params.quadrant_align_info) for r in ch_idx_of_interest]


# def get_cropped_img_set(full_img: np.ndarray, channels_raw, )

def _crop(img, rectangle):
    """
    Crop the image according to defined points
    :param img: image to be cropped
    :param rectangle: coordinates of two points defining a rectangle, [x1,y1,x2,y2]
    :return:
    """
    q_pos = rectangle
    return img[q_pos[1]:q_pos[3], q_pos[0]:q_pos[2]]

# def index_to_raw_index(idx, params):
#     params.channels
