import copy
import itertools
import os
from typing import Callable, Type, Union, Any, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib

from modules.DataLoading import DataLoader
from modules.ParamLoading import ParamLoader
from modules.SphereAnalysis import SphereAnalyser
from modules.SphereAnalysis3 import SphereAnalyser3
from modules.SphereAnalysis5 import SphereAnalyser5
# from modules.CylinderAnalysis import CylinderAnalyser
from modules.proc_utils import img_preprocess, plot_df_helper
from modules.proc_utils import save_circle_masks
from modules.utils import get_output_path, get_output_df_path, to_effective_df, write_df, write_output_meta

import multiprocessing
import functools


class SignalFileProcessingHelperBase:
    def __init__(self, param_source: Union[ParamLoader, str],
                 analyser: Type[SphereAnalyser] = SphereAnalyser3,
                 output_dir_method: Callable[[ParamLoader], str] = get_output_path,
                 output_df_path_method: Callable[[ParamLoader], str] = get_output_df_path,
                 **kwargs):
        self.data_set = DataLoader(param_source)
        self.analyser_cls = analyser
        self.output_dir_method = output_dir_method
        self.output_df_path_method = output_df_path_method
        self.verbose = self.data_set.params.verbose_plot
        self.kwargs = {**kwargs, **self.data_set.params.kwargs}
        self.plot_line_thickness = self.data_set.params.plot_line_thickness
        self.plot_text_scale = self.data_set.params.plot_text_scale

    @staticmethod
    def get_sub_df_basename():
        return 'sub_df.xlsx'

    def _plot_labelled_internal(self, sphere_analysis, analyser_num, img_num_output_dir_method, write_img_method,
                                params):
        pathlib.Path(img_num_output_dir_method(analyser_num)).mkdir(parents=True, exist_ok=True)
        print(f'Plotting labelled imgs for {analyser_num}')
        if sphere_analysis.df.shape[0] == 0:
            print(f'No droplet detected for image {analyser_num}')
            return
        # plot detection result
        # imgs_labelled = [
        #     cv2.cvtColor(
        #         sphere_analysis.img_detect if params.ch_detect_idx == i else sphere_analysis.enhance_img(
        #             sphere_analysis.signals[i]), cv2.COLOR_GRAY2BGR) for i in
        #     range(len(params.channels))]

        imgs_labelled = [
            cv2.cvtColor(
                img_preprocess(sphere_analysis.signals[i], log=True), cv2.COLOR_GRAY2BGR) for i in
            range(len(params.channels))]

        color = None
        for index, row in sphere_analysis.df.iterrows():
            if row.dark:
                color = (255, 0, 0)  # blue
            elif row.uneven:
                color = (255, 255, 0)  # aqua
            elif row.feature:
                color = (0, 0, 255)  # red
            elif not row.feature:
                # color = (255, 128, 128)  # light blue
                color = (93, 245, 66)  # Green
            for j in range(len(params.channels)):
                cv2.circle(imgs_labelled[j], (int(row.x), int(row.y)), int(row.radius), color,
                           params.plot_line_thickness)

        # Put index into the plot
        imgs_labelled_text = copy.deepcopy(imgs_labelled)
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # font_scale
        font_scale = self.plot_text_scale
        # white color in BGR
        color = (255, 255, 0)
        # Line thickness of 1 px
        thickness = self.plot_line_thickness
        for index, row in sphere_analysis.df.iterrows():
            text = str(row.name)
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            for j in range(len(params.channels)):
                cv2.putText(imgs_labelled_text[j], str(row.name),
                            (int(row.x - text_size[0] / 2), int(row.y + text_size[1] / 2)), font, font_scale, color,
                            thickness, cv2.LINE_AA)

        for j in range(len(params.channels)):
            channel = params.channels[j]
            write_img_method(f'img_labelled_ch_{channel}', analyser_num, imgs_labelled[j])
            write_img_method(f'img_labelled_text_ch_{channel}', analyser_num, imgs_labelled_text[j])

        df_plot = to_effective_df(sphere_analysis.df)
        dr_handle = plot_df_helper(df_plot, self.data_set.params, annotate=True)
        if dr_handle is not None:
            dr_handle.savefig(os.path.join(img_num_output_dir_method(analyser_num), 'Detection_result.png'))
            plt.close(dr_handle)
        else:
            print('ch_plot is not in 2D or 3D format, skip plotting detection result (combined)')

    def _plot_masked_circles_internal(self, img_num, sphere_analyser):
        for q in range(len(sphere_analyser.signals)):
            signal_enhanced = img_preprocess(sphere_analyser.signals[q], 10, 100, 10, True)
            if np.max(signal_enhanced) == 0:
                raise ValueError('Signal max is zero')
            pathlib.Path(self.get_img_num_output_dir(img_num)).mkdir(parents=True, exist_ok=True)
            save_circle_masks(signal_enhanced, sphere_analyser.df, self.get_img_num_output_dir(img_num),
                              f"{self.data_set.params.channels[q]}_{img_num}_", "")

    def get_img_num_output_dir(self, img_num: Union[str, int, float]):
        return os.path.join(self.output_dir_method(self.data_set.params), str(img_num))

    def write_img(self, file_base_name: str, img_num: Union[str, int, float], img: np.ndarray) -> None:
        cv2.imwrite(os.path.join(self.get_img_num_output_dir(img_num), f"{file_base_name}.jpg"), img)

    @staticmethod
    def get_sub_df_basename():
        return 'sub_df.xlsx'

    def save_sub_df(self, img_num, df):
        pathlib.Path(self.get_img_num_output_dir(img_num)).mkdir(parents=True, exist_ok=True)
        pd_writer = pd.ExcelWriter(
            os.path.join(self.get_img_num_output_dir(img_num), self.get_sub_df_basename()))
        try:
            df[df['feature'] == True].to_excel(pd_writer, sheet_name="True_sheet")
            df[df['feature'] == False].to_excel(pd_writer, sheet_name="False_sheet")
            pd_writer.save()
        except:
            return


class SignalFileProcessingHelper(SignalFileProcessingHelperBase):
    def __init__(self, param_source: Union[ParamLoader, str], analyser: Type[SphereAnalyser] = SphereAnalyser3,
                 output_dir_method: Callable[[ParamLoader], str] = get_output_path,
                 output_df_path_method: Callable[[ParamLoader], str] = get_output_df_path,
                 **kwargs):
        super().__init__(param_source, analyser, output_dir_method, output_df_path_method, **kwargs)
        self.analyser_list = []
        self.analyser_num = []

    def run(self):
        print('Initialising analysers')
        self.init_analysers()
        self.execute_foreach(self._run_internal)
        for i in self.get_iter_collection():
            # print(f'Processing {self.analyser_num[i]}')
            # self.detect_shape(i)
            # self.label(i)
            # self.plot_labelled(i)
            if self.verbose:
                print(f'Plotting masked regions for {self.analyser_num[i]}. Verbose enabled')
                self.plot_masked_circles(i)
            self.save_sub_dfs(i)
        self.combine_data()
        self.plot_labelled()

    @staticmethod
    def _detect_shape_internal(sphere_analysis, a_num):
        print(f'Detecting {a_num}')
        # print('eee')
        sphere_analysis.analyse_as_circles()

    @classmethod
    def _run_internal(cls, sphere_analysis, a_num):
        cls._detect_shape_internal(sphere_analysis, a_num)
        cls._label_internal(sphere_analysis, a_num)
        # cls._plot_labelled_internal(sphere_analysis, a_num)

    def get_iter_collection(self, idx: Union[int, Iterable] = None):
        if idx is None:
            return range(len(self.analyser_list))
        it = idx
        if not isinstance(idx, Iterable):
            if not isinstance(idx, int):
                raise TypeError(f'Index should be int or list of int. Got {type(idx)}')
            it = [it]
        return it

    def init_analysers(self):
        params = self.data_set.params
        for i, signals in self.data_set.get_signals().items():
            sphere_analysis = self.analyser_cls(signals, params.ch_detect_idx, params.ch_feature_idx,
                                                params.ch_uniform_idx,
                                                **self.kwargs)  # type:SphereAnalyser
            self.analyser_list.append(sphere_analysis)
            self.analyser_num.append(i)

    def detect_shape(self, idx: Union[int, Iterable] = None):
        # make sure output directory exists
        pathlib.Path(self.output_dir_method(self.data_set.params)).mkdir(parents=True, exist_ok=True)
        self.execute_foreach(self._detect_shape_internal, idx)

    def label(self, idx: Union[int, Iterable] = None):
        self.execute_foreach(self._label_internal, idx)

    @staticmethod
    def _label_internal(sphere_analysis, a_num):
        print(f'Labelling {a_num}')
        sphere_analysis.label_circles()

    def execute_foreach(self, func: Callable[[SphereAnalyser, int, Any], None], idx: Union[int, Iterable] = None, *args,
                        **kwargs):
        idx_it = self.get_iter_collection(idx)
        if self.data_set.params.parallel:
            # if False:
            try:
                cpu_count = multiprocessing.cpu_count()
                with multiprocessing.Pool(cpu_count) as p:
                    idx_it = self.get_iter_collection(idx)
                    it = [(func, self.analyser_list[i], self.analyser_num[i], args, kwargs) for i in idx_it]
                    al = p.starmap(self._execute_foreach_atom, it)
                    # p.starmap(self._execute_foreach_atom, idx_it)
                    for k in range(len(idx_it)):
                        self.analyser_list[idx_it[k]] = al[k]
                return
            except AttributeError as err:
                print(f'Cannot run in parallel mode. Running in sequence. {err}')
                for i in idx_it:
                    func(self.analyser_list[i], self.analyser_num[i], *args, **kwargs)
        for i in idx_it:
            func(self.analyser_list[i], self.analyser_num[i], *args, **kwargs)

    @staticmethod
    def _execute_foreach_atom(func: Callable[[SphereAnalyser, int, Any], None], analyser, analyser_num, args, kwargs):
        func(analyser, analyser_num, *args, **kwargs)
        return analyser

    def plot_labelled(self, idx: Union[int, Iterable] = None):
        # self.execute_foreach(self._plot_labelled_internal, idx, img_num_output_dir_method=self.get_img_num_output_dir,
        #                      write_img_method=self.write_img, params=self.data_set.params)
        for i in self.get_iter_collection():
            self._plot_labelled_internal(self.analyser_list[i], self.analyser_num[i], self.get_img_num_output_dir,
                                         self.write_img, self.data_set.params)

    @staticmethod
    def plot_labelled_one(sphere_analysis: SphereAnalyser):
        imgs_labelled = [
            cv2.cvtColor(img_preprocess(sphere_analysis.signals[i], 10, 100, 10, True), cv2.COLOR_GRAY2BGR) for i in
            range(len(sphere_analysis.signals))]

        # color = None
        for index, row in sphere_analysis.df.iterrows():
            if 'dark' in row and row.dark:
                color = (255, 0, 0)  # blue
            elif 'uneven' in row and row.uneven:
                color = (255, 255, 0)  # aqua
            elif 'feature' in row and row.feature:
                color = (0, 255, 255)  # yellow
            elif 'feature' in row and not row.feature:
                color = (0, 0, 255)  # red
            else:
                color = (0, 255, 0)
            for j in range(len(sphere_analysis.signals)):
                cv2.circle(imgs_labelled[j], (int(row.x), int(row.y)), int(row.radius), color, 1)

        # Put index into the plot
        imgs_labelled_text = copy.deepcopy(imgs_labelled)
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # font_scale
        font_scale = 1
        # white color in BGR
        color = (255, 255, 255)
        # Line thickness of 1 px
        thickness = 1
        for index, row in sphere_analysis.df.iterrows():
            text = str(row.name)
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            for j in range(len(sphere_analysis.signals)):
                cv2.putText(imgs_labelled_text[j], str(row.name),
                            (int(row.x - text_size[0] / 2), int(row.y + text_size[1] / 2)),
                            font,
                            font_scale, color, thickness, cv2.LINE_AA)
        return imgs_labelled, imgs_labelled_text

    def plot_masked_circles(self, idx: Union[int, Iterable] = None):
        for i in self.get_iter_collection(idx):
            img_num = self.analyser_num[i]
            sphere_analysis = self.analyser_list[i]
            self._plot_masked_circles_internal(img_num, sphere_analysis)

    def save_sub_dfs(self, idx: Union[int, Iterable] = None):
        # Save DataFrame for each image set
        for i in self.get_iter_collection(idx):
            self.save_sub_df(self.analyser_num[i], self.analyser_list[i].df)

    def load_sub_dfs(self, idx: Union[int, Iterable] = None):
        for i in self.get_iter_collection(idx):
            sphere_analysis = self.analyser_list[i]
            t_df = pd.read_excel(
                os.path.join(self.get_img_num_output_dir(self.analyser_num[i]), self.get_sub_df_basename()),
                index_col=0, sheet_name="True_sheet")
            f_df = pd.read_excel(
                os.path.join(self.get_img_num_output_dir(self.analyser_num[i]), self.get_sub_df_basename()),
                index_col=0, sheet_name="False_sheet")
            df = t_df.append(f_df)
            sphere_analysis.df = df

    def combine_data(self):
        parent_dir = self.output_dir_method(self.data_set.params)
        true_df = pd.DataFrame()
        false_df = pd.DataFrame()
        sub_df_basename = self.get_sub_df_basename()
        dfs_list = []
        for subdir, dirs, files in os.walk(parent_dir):
            if subdir[len(parent_dir):].count(os.sep) > 1:
                continue
            for file in files:
                if not file == sub_df_basename:
                    continue
                new_true_df = pd.read_excel(os.path.join(subdir, file), sheet_name="True_sheet").dropna()
                new_false_df = pd.read_excel(os.path.join(subdir, file), sheet_name="False_sheet").dropna()
                new_true_df['img_num'] = os.path.basename(subdir)
                new_false_df['img_num'] = os.path.basename(subdir)
                new_true_df.rename(columns={'Unnamed: 0': 'local index'}, inplace=True)
                new_false_df.rename(columns={'Unnamed: 0': 'local index'}, inplace=True)
                # true_df = true_df.append(new_true_df, ignore_index=True)
                # false_df = false_df.append(new_false_df, ignore_index=True)
                dfs_list.append(new_true_df)
                dfs_list.append(new_false_df)
        # combined_df = true_df.append(false_df, ignore_index=True)
        combined_df = pd.concat(dfs_list, ignore_index=True)
        # combined_df.to_excel(self.output_df_path_method(self.data_set.params))
        write_df(combined_df, self.output_df_path_method(self.data_set.params))
        handle = plot_df_helper(combined_df, self.data_set.params, annotate=False)
        if handle is not None:
            handle.savefig(os.path.join(parent_dir, r'plot_detection_result.png'))
        else:
            print('ch_plot is not in 2D or 3D format, skip plotting detection result (combined)')


class IterativeSignalFileProcessingHelper(SignalFileProcessingHelperBase):
    def __init__(self, param_source: Union[ParamLoader, str], *args, **kwargs):
        super().__init__(param_source, *args, **kwargs)
        self.result_dict = None

    def run(self):
        self.run_analysers()
        for k, v in self.result_dict.items():
            self.save_sub_df(k, v)
        df_combined = self.combine_dfs()
        # df_combined.to_excel(self.output_df_path_method(self.data_set.params))
        write_df(df_combined, self.output_df_path_method(self.data_set.params))
        write_output_meta(self.data_set.params, {'img_keys': list(self.result_dict.keys())})

    def combine_dfs(self):
        dfs = []
        for k, v in self.result_dict.items():
            v["img_num"] = k
            v.reset_index(inplace=True)
            v.rename(columns={'index': 'local index'}, inplace=True)
            dfs.append(v)
        df_combined = pd.concat(dfs, ignore_index=True)
        return df_combined

    def run_analysers(self):
        if self.data_set.params.parallel:
            try:
                cpu_count = multiprocessing.cpu_count()
                result = []
                with multiprocessing.Pool(cpu_count) as p:
                    data_gen = self.data_set.iter_signals()
                    chunksize = cpu_count * 10
                    func_partial = functools.partial(self.run_one_paired, params=self.data_set.params,
                                                     kwargs_as_dict=self.kwargs)
                    while True:
                        print(f'Reading data chunk: size {chunksize}')
                        chunk = itertools.islice(data_gen, chunksize)
                        result_chunk = list(p.imap(func_partial, chunk))
                        if len(result_chunk) > 0:
                            result.extend(result_chunk)
                        else:
                            break
                self.result_dict = dict(result)
                return
            except AttributeError as err:
                print(f'Cannot run in parallel mode. Running in sequence. {err}')
                pass

        df_dict = {}
        for img_num, signals in self.data_set.iter_signals():
            _, df_dict[img_num] = self.run_one(img_num, signals, self.data_set.params, **self.kwargs)
        self.result_dict = df_dict

    def run_one_paired(self, img_num_signals_tuple, params, kwargs_as_dict):
        return self.run_one(*img_num_signals_tuple, params, **kwargs_as_dict)

    def run_one(self, img_num, signals, params, *args, **kwargs):
        analyser = self.analyser_cls(signals, params.ch_detect_idx, params.ch_feature_idx,
                                     params.ch_uniform_idx, *args, **kwargs)  # type:SphereAnalyser
        print(f"Running {img_num}")
        analyser.run()
        self._plot_labelled_internal(analyser, img_num, self.get_img_num_output_dir, self.write_img,
                                     self.data_set.params)
        if self.verbose:
            self._plot_masked_circles_internal(img_num, analyser)
        return img_num, analyser.df
