import os
import pickle
import sys

import numpy as np
import pandas as pd

from modules.IntensityConcentrationConversion import convert_linear, IntensityConcentrationConverterBase, cvt_df
from modules.ParamLoading import ParamLoader
from modules.proc_utils import plot_detection_result, plot_df_helper
from modules.utils import get_output_path, get_output_df_path, get_cvt_obj_path, argv_proc, write_df, read_df


def main(argv=None):
    ag = argv_proc(argv, sys.argv)
    params = ParamLoader(ag[0])
    output_df_path = get_output_df_path(params)
    # df = pd.read_excel(output_df_path, index_col=0)
    df = read_df(output_df_path)
    with open(get_cvt_obj_path(params), 'rb') as handle:
        cvt_obj = pickle.load(handle)  # type:IntensityConcentrationConverterBase
    # df_new = convert_linear(df, cvt_obj, params)
    df_new = cvt_df(df, cvt_obj, params)
    # map concentration
    if params.cvt_map_list is not None:
        if len(params.cvt_map_list) != len(params.channels):
            raise ValueError('cvt_map not in the same dimension as channels')
        if type(params.cvt_map_list) is dict:
            # fill result cvt columns
            # for cvt_idx in range(len(params.cvt_map_list[0]['component'])):
            #     df_new[f'signal_{cvt_idx}_pv_cvt'] = 0
            addition = np.zeros((df.shape[0], len(params.cvt_map_list['component'])))
            comp_high = np.array(params.cvt_map_list['component'])
            dye_high = np.array(params.cvt_map_list['dye'])
            background_high = np.array(params.cvt_map_list['background'])
            scale_factor = comp_high / dye_high
            for cvt_idx in range(len(params.cvt_map_list['component'])):  # number of component
                for ch_idx in range(len(params.channels)):
                    addition[:,cvt_idx] += df_new[f'signal_{ch_idx}_pv_cvt'] * scale_factor[ch_idx, cvt_idx]
                addition[:,cvt_idx] += (1 - np.sum(np.array([df_new[f'signal_{ch_idx}_pv_cvt'] for ch_idx in range(len(params.channels))]).T/dye_high, axis=1)) * background_high[cvt_idx]

            for cvt_idx in range(len(params.cvt_map_list['component'])):
                df_new[f'signal_{cvt_idx}_pv_cvt'] = addition[:, cvt_idx]

        else:  # if it is a list
            for ch_idx in range(len(params.channels)):
                # 0 is old 1 is new
                pair = params.cvt_map_list[ch_idx]
                if pair is None:
                    continue
                df_new[f'signal_{ch_idx}_pv_cvt'] = df_new[f'signal_{ch_idx}_pv_cvt'] / pair[0] * pair[1]

    # df_new.to_excel(output_df_path)
    write_df(df_new, output_df_path)
    df_plot = df_new[np.logical_and(df_new.uneven == False, df_new.dark == False)]

    # handle_pv = plot_df_helper(df_plot, params, annotate=False, label_suffix="per volume (user defined)", name_suffix="_cvt")
    # handle_pv.savefig(os.path.join(get_output_path(params), r'plot_detection_result_cvt.png'))

    handle = plot_df_helper(df_plot, params, annotate=False, label_suffix="concentration", name_suffix="_cvt")
    handle.savefig(os.path.join(get_output_path(params), r'plot_detection_result_cvt.png'))
    # handle.savefig()
if __name__=="__main__":
    main()
