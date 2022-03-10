import os
from collections import Iterable
from typing import Union, Dict, Tuple, List

import cv2

from modules.EvenLuminescence import LuminescenceCorrector
from modules.ParamLoading import ParamLoader
from modules.proc_utils import imreadmulti_mean
from modules.quadrant_utils import get_cropped_img_set, crop_quadrant

import numpy as np

class DataLoader:
    def __init__(self, param_source: Union[str, ParamLoader], missing_key_warning=True, missing_key_error=False,
                 optional_list=None):
        """
        Initiate a DataLoader for loading signals and Illumination images.
        :param param_source:
        :param missing_key_warning:
        :param missing_key_error:
        :param optional_list:
        """
        if type(param_source) is str:
            self.params = ParamLoader(param_source, missing_key_warning, missing_key_error,
                                      optional_list)  # type: ParamLoader
        elif type(param_source) is ParamLoader:
            self.params = param_source  # type: ParamLoader
        self._lumi_corrs = None
        self._bkgs = None
        self._bkg = None
        self._signals = None

    # def load(self, params: ParamLoader):
    #     return DataLoader

    def get_lumi_corrs(self):
        if self._lumi_corrs is None:
            if self.params.quadrant_mode:
                lumi_corr_imgs = [imreadmulti_mean(p) for p in self.params.lumi_corr_img_paths]
                lumi_corr_bkg = imreadmulti_mean(self.params.lumi_corr_bkg_path)
                lumi_corr_signals_full = [cv2.subtract(im, lumi_corr_bkg) for im in lumi_corr_imgs]
                # lumi_corr_signals = [crop_quadrant(lumi_corr_signals_full[i], self.params.quadrant_align_info[i]) for i in
                #                      range(len(self.params.channels))]
                lumi_corr_signals = [
                    crop_quadrant(lumi_corr_signals_full[i], self.get_ch_idx_raw(self.params.channels[i]),
                                  self.params._quadrant_align_info) for i in range(len(self.params.channels))]

                self._lumi_corrs = [LuminescenceCorrector(lumi_corr_signals[i]) for i in
                                    range(len(self.params.channels))]
            else:
                self._lumi_corrs = [
                    LuminescenceCorrector(cv2.subtract(imreadmulti_mean(self.params.lumi_corr_img_paths[i]),
                                                       imreadmulti_mean(
                                                           self.params.lumi_corr_bkg_paths[i])))
                    for i in range(len(self.params.channels))]

        return self._lumi_corrs

    def get_bkgs(self):
        if self._bkgs is None:
            if self.params.quadrant_mode:
                self._bkgs = self._load_quadrant_images(self.params.bkg_path)
            else:
                self._bkgs = [imreadmulti_mean(self.params.bkg_paths[k]) for k in range(len(self.params.bkg_paths))]
        return self._bkgs

    def get_bkg(self):
        """
        Only for quadrant mode. Get the full background image
        :return: uncropped background image
        """
        if not self.params.quadrant_mode:
            raise AttributeError('This function is only valid in quadrant mode')
        self._bkg = imreadmulti_mean(self.params.bkg_path)
        return self._bkg

    def iter_signals(self) -> Tuple:
        if self.params.quadrant_mode:
            for i in range(0, 100):
                valid_filename = f'{i}{self.params.img_format}'
                img_path = os.path.join(self.params.img_dir, valid_filename)
                if not os.path.exists(img_path):
                    continue
                # load signals
                data_imgs = self._process_quadrant_data_images(img_path)
                for k in range(len(data_imgs)):
                    img_num = f'{i}-{k}'
                    cropped_data_images_single_frame = data_imgs[k]
                    signals = [self.get_lumi_corrs()[j].correct_img(cropped_data_images_single_frame[j]) for j in
                               range(len(self.params.channels))]
                    yield img_num, signals
        else:
            for i in range(0, 100):
                filenames = [f'{k}_{i}{self.params.img_format}' for k in self.params.channels]
                file_exist_flag = True
                for fn in filenames:
                    if not os.path.exists(os.path.join(self.params.img_dir, fn)):
                        file_exist_flag = False
                        break
                if not file_exist_flag:
                    continue

                # load signals
                signals = [self.get_lumi_corrs()[k].correct_img(cv2.subtract(imreadmulti_mean(
                    os.path.join(self.params.img_dir, f'{self.params.channels[k]}_{i}{self.params.img_format}')),
                    self.get_bkgs()[k]))
                    for k in range(len(self.params.channels))]
                yield i, signals

    def get_signals(self, load_callback=None) -> Dict:
        """
        Get signals. Delay loaded
        :param load_callback: Will pass image number under processing to load_callback
        :return: signals dictionary, img_num: [signals]
        """
        if self._signals is None:
            # loop through all files
            signals_dict = {k: v for k, v in self.iter_signals()}
            if signals_dict == {}:
                raise ValueError('No data image matching naming convention. Check if file exists and named correctly.')
            self._signals = signals_dict
        return self._signals

    def _load_quadrant_images(self, img_paths):
        ch_idx_in_raw = self.get_ch_idx_raw()  # valid channels' indices in the raw index list
        if isinstance(img_paths, list):
            # this is for the case where the images are separate for channels
            # crop corresponding images
            return [crop_quadrant(imreadmulti_mean(img_paths[i]), i, self.params.quadrant_align_info) for i in
                    range(len(ch_idx_in_raw))]
        else:
            # this is for the case where one image includes all info for all channels
            assert isinstance(img_paths, str)
            img_of_interest = imreadmulti_mean(img_paths)
            return [crop_quadrant(img_of_interest, i, self.params.quadrant_align_info) for i in
                    range(len(ch_idx_in_raw))]

    def get_ch_idx_raw(self, ch_names=None):
        """
        Get channel index in quadrant definition. if ch_nams not specified. return a full list of all available channels
        :param ch_names: ch name or list of ch names
        :return: idx or list of indices
        """
        if ch_names is None:
            return [self.params.channels_raw.index(c) for c in self.params.channels]
        elif isinstance(ch_names, Iterable) and not isinstance(ch_names, str):
            return [self.params.channels_raw.index(c) for c in ch_names]
        else:
            return self.params.channels_raw.index(ch_names)

    def _process_quadrant_data_images(self, img_path: str) -> Union[None, List[List[np.ndarray]]]:
        """

        Parameters
        ----------
        img_path image path

        Returns list of signals of each channel, for each image frame in a list
        -------

        """
        imgs_raw = cv2.imreadmulti(img_path, flags=-1)[1]
        bkg = self.get_bkg()
        signals_raw = [cv2.subtract(img_raw, bkg) for img_raw in imgs_raw]
        # imgs = [crop_quadrant(imgs_raw, rect) for rect in self.params.quadrant_align_info[self.get_ch_idx_raw()]]
        imgs = [get_cropped_img_set(signal_raw, self.params) for signal_raw in signals_raw]
        return imgs
