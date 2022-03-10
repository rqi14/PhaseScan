"""
Load json parameters
"""
import json
import logging
import os
from typing import List
# from jsoncomment import JsonComment
import copy
import commentjson
import numpy as np


class ParamLoader:
    def __init__(self, json_path, missing_key_warning=True, missing_key_error=False, optional_list=None):
        if not isinstance(json_path, str):
            self.__dict__ = json_path.__dict__.copy()
            return
        with open(json_path) as handle:
            # parser = JsonComment(json)
            parser = commentjson
            data = parser.load(handle)
            self.json_data = data

        # config options
        self.missing_key_warning = missing_key_warning
        self.missing_key_error = missing_key_error
        self.optional_list = optional_list if optional_list is not None else []

        # Check quadrant mode
        self.quadrant_mode = self.json_read_bool('quadrant_mode', False)

        # quadrant mode compaibility
        if self.quadrant_mode:
            info_dict = self._json_parse_read("quadrant_align_info", None)
            self._quadrant_align_info = self.QuadrantAlignInfo(info_dict)
            self.img_depth = self.json_read_num("img_depth")
        else:
            self._add_optional("quadrant_align_info")
            self._add_optional("img_depth")
            # self._add_optional("img_depth")

        # hidden functionality for compatibility
        self._add_optional('output_path_override')
        self.output_path_override = self.json_read_path('output_path_override')
        self._add_optional('stddev_scan')
        self._add_optional('train_on_cvt')
        self._add_optional('drug_plug_kwargs')

        # add relative path support
        self.json_path = json_path

        # Params
        self.use_csv = self.json_read_bool("use_csv")
        self.img_format = self.json_read_str('img_format')
        self.img_pixel_size = self.json_read_num('img_pixel_size')
        self.cvt_pixel_size = self.json_read_num('cvt_pixel_size')
        self.cvt_depth = self.json_read_num('cvt_depth')

        # Directories
        self.img_dir = self.json_read_path('img_dir')
        self.lumi_dir = self.json_read_path('lumi_dir')
        self.bkg_dir = self.json_read_path('bkg_dir')
        self.lumi_bkg_dir = self.json_read_path('lumi_bkg_dir')
        self.int_conc_cvt_dir = self.json_read_path('int_conc_cvt_dir')

        # Channels and bkg paths
        self.channels_raw = self.json_read_list('channels')  # quadrant mode compatibility
        self.channels = [i for i in self.channels_raw if i is not None]
        self.ch_detect_idx = self.json_find_channel_idx_by_name('ch_detect')
        self.ch_feature_idx = self.json_find_channel_idx_by_name('ch_feature')
        self.ch_uniform_idx = self.json_find_channel_idx_by_name('ch_uniform')
        # for compatibility
        if self.quadrant_mode:
            _bkg_paths = self.json_read_str('bkg_names')
            _lumi_corr_bkg_paths = self.json_read_str('lumi_corr_bkg_names')
            assert isinstance(_bkg_paths, str)
            assert isinstance(_lumi_corr_bkg_paths, str)
            self._bkg_paths = os.path.join(self.bkg_dir, _bkg_paths)
            self._lumi_corr_bkg_paths = os.path.join(self.lumi_bkg_dir, _lumi_corr_bkg_paths)
        else:
            self._bkg_paths = [os.path.join(self.bkg_dir, p) for p in self.json_read_dict_list_ch_mapped('bkg_names')]
            self._lumi_corr_bkg_paths = [os.path.join(self.lumi_bkg_dir, p) for p in
                                         self.json_read_dict_list_ch_mapped('lumi_corr_bkg_names')]  # property
        self.lumi_corr_img_paths = [os.path.join(self.lumi_dir, p) for p in
                                    self.json_read_dict_list_ch_mapped('lumi_corr_img_names')]

        # Processing options
        self.verbose_plot = self._json_parse_read('verbose_plot', True)
        self.plot_text_scale = self._json_parse_read('plot_text_scale', 1)
        self.plot_line_thickness = self._json_parse_read('plot_line_thickness', 1)
        self.stddev_scan = self.json_read_str('stddev_scan')
        self.analyser = self.json_read_str('analyser')
        self.parallel = self.json_read_str('parallel')
        # cvt map
        self.cvt_map_list = self.json_read_str('cvt_map')  # list but use str
        # plot
        if 'ch_plot' not in self.json_data:
            self.ch_plot_idx = [0, 1]  # for compatibility
        else:
            self.ch_plot_idx = self.json_find_ch_list_idx_by_name('ch_plot')

        # Training
        # Training channels
        self.ch_train_idx = self.json_find_ch_list_idx_by_name('ch_train', [0, 1])
        # train on cvt data otherwise use raw pv data
        self.train_on_cvt = self.json_read_bool('train_on_cvt', True)

        # support kwargs
        self.kwargs = self._json_parse_read('kwargs', {})
        self.cvt_kwargs = self._json_parse_read('cvt_kwargs', {})

        self.drug_plug_kwargs = self.process_kwargs_entry(self._json_parse_read('drug_plug_kwargs', {}))

        # compatibility
        if self.quadrant_mode:
            # this image depth should be number of pixels
            self.kwargs['img_depth'] = self.img_depth / self.img_pixel_size

    # for quadrant compatibility
    @property
    def lumi_corr_bkg_paths(self):
        if self.quadrant_mode:
            raise NameError(
                'Accessing path list in quadrant mode. In quadrant mode there is only one bkg path please use '
                'lumi_corr_img_path instead')
        else:
            return self._lumi_corr_bkg_paths

    @lumi_corr_bkg_paths.setter
    def lumi_corr_bkg_paths(self, value):
        self._lumi_corr_bkg_paths = value

    @property
    def lumi_corr_bkg_path(self):
        if not self.quadrant_mode:
            raise NameError(
                'Accessing path in non-quadrant mode. Use lumi_corr_bkg_paths instead')
        else:
            return self._lumi_corr_bkg_paths

    @lumi_corr_bkg_path.setter
    def lumi_corr_bkg_path(self, value):
        self._lumi_corr_bkg_paths = value

    @property
    def bkg_paths(self):
        if self.quadrant_mode:
            raise AttributeError(
                'Accessing path list in quadrant mode. In quadrant mode there is only one bkg path please use '
                'bkg_path instead')
        else:
            return self._bkg_paths

    @bkg_paths.setter
    def bkg_paths(self, value):
        self._bkg_paths = value

    @property
    def bkg_path(self):
        if not self.quadrant_mode:
            raise NameError(
                'Accessing path in non-quadrant mode. Use bkg_path instead')
        else:
            return self._bkg_paths

    @bkg_path.setter
    def bkg_path(self, value):
        self._bkg_paths = value

    @property
    def quadrant_align_info(self):
        if not self.quadrant_mode:
            raise AttributeError('Only available at quadrant mode')
        return copy.deepcopy(self._quadrant_align_info)

    @quadrant_align_info.setter
    def quadrant_align_info(self, value):
        self._quadrant_align_info = value

    def _add_optional(self, name):
        if name not in self.optional_list:
            self.optional_list.append(name)

    def get_json_cwd(self):
        return os.path.dirname(self.json_path)

    def json_read_str(self, key):
        return self._json_parse_read(key, None)

    def json_read_path(self, key):
        p = self._json_parse_read(key, None)
        if p is not None and not os.path.isabs(p):
            p = os.path.join(self.get_json_cwd(), p)
        return p

    def json_read_list(self, key):
        return self._json_parse_read(key, [])

    def json_read_num(self, key):
        value = self._json_parse_read(key, None)
        return float(value) if value is not None else None

    def _json_parse_read(self, key, default):
        if key in self.json_data:
            return self.json_data[key]
        else:
            if hasattr(self, 'optional_list') and self.optional_list is not None and key in self.optional_list:
                return default
            if hasattr(self, 'missing_key_warning') and self.missing_key_warning:
                logging.warning(f'Key {key} missing in the json file')
            if hasattr(self, 'missing_key_error') and self.missing_key_error:
                raise KeyError(f'Key {key} missing in the json file')
            return default

    def json_find_channel_idx_by_name(self, key, default=None):
        """
        Find channel index by its name
        :param key:
        :param default:
        :return:
        """
        name = self._json_parse_read(key, None)
        if name is None:
            return default
        else:
            return self.find_ch_idx_in_channels(name)

    def json_find_ch_list_idx_by_name(self, key, default=None):
        """
        Convert a channel list with mixed names and index to a list of channel index
        :param key:
        :param default:
        :return:
        """
        ch_list = self._json_parse_read(key, None)
        if ch_list is None:
            return default
        assert isinstance(ch_list, List)
        idx_list = [None for j in range(len(ch_list))]
        for i in range(len(ch_list)):
            idx_list[i] = self.find_ch_idx_in_channels(ch_list[i])
        return idx_list

    def find_ch_idx_in_channels(self, name_or_idx):
        if type(name_or_idx) is int:
            return name_or_idx
        else:
            assert type(name_or_idx) is str
            return self.channels.index(name_or_idx)

    def json_read_bool(self, key, default=None):
        bl = self._json_parse_read(key, None)
        if bl is None:
            return default
        else:
            assert type(bl) is bool
            return bl

    def json_read_dict_list_ch_mapped(self, key):
        if isinstance(self.json_data[key], dict):
            return list(map(self.json_data[key].get, self.channels))
        elif isinstance(self.json_data[key], list):
            return self.json_read_list(key)
        else:
            assert isinstance(self.json_data[key], str)
            # NOT NECESSARY ANYMORE
            # for compatibility, if quadrant mode then return single path and then split the image
            # if not quadrant mode, return a list of identical path
            # if self.quadrant_mode:
            #     return self.json_data[key]
            # else:
            return [self.json_data[key] for ch in self.channels]

    def dict_list_ch_mixed_read(self, dict_or_list_or_single_value, default=None):
        if isinstance(dict_or_list_or_single_value, dict):
            return list(map(lambda x: dict_or_list_or_single_value.get(x, default), self.channels))
        elif isinstance(dict_or_list_or_single_value, list):
            return dict_or_list_or_single_value
        else:  # single value repeat for all channels
            return [dict_or_list_or_single_value for ch in self.channels]

    def process_kwargs_entry(self, kwargs_dict: dict):
        """
        Process kawrgs and convert special items. #chmap -> map the channels and set to index list
        :param kwargs_dict:
        """
        new_kwargs_dict = kwargs_dict.copy()
        for k, v in kwargs_dict.items():
            key_split = k.split('#')
            if len(key_split) > 1:
                kd_split = key_split[-1].split('@')
                if kd_split[0] == 'chmap':
                    new_kwargs_dict[key_split[0]] = self.dict_list_ch_mixed_read(v, default=kd_split[1] if len(
                        kd_split) > 1 else None)
        return new_kwargs_dict

    class QuadrantAlignInfo:
        def __init__(self, quadrant_align_info_dict: dict):
            self.qpos = quadrant_align_info_dict["qpos"]
            self.ref_idx_qpos = quadrant_align_info_dict["ref"]
            self.warp_idx_qpos = {int(k): np.array(v) for k, v in quadrant_align_info_dict["warp"].items()}

        def qpos_denorm(self, img_shape):
            qpos_h, qpos_w = img_shape
            qpos_denorm = [[int(q[0] * qpos_w), int(q[1] * qpos_h), int(q[2] * qpos_w), int(q[3] * qpos_h)] for q in
                           self.qpos]
            return qpos_denorm

        def dsize(self, img_shape):
            """
            columns rows
            :param img_shape:
            :return:
            """
            qpos_denrom = self.qpos_denorm(img_shape)
            qpos_ref = qpos_denrom[self.ref_idx_qpos]
            return (int(qpos_ref[2]) - int(qpos_ref[0]), int(qpos_ref[3]) - int(qpos_ref[1]))
