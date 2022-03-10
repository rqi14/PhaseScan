import json
import os
import pandas as pd
import numpy as np
from modules.ParamLoading import ParamLoader


def get_output_df_basename(params: ParamLoader):
    if params.use_csv:
        return r'combined_df.csv'
    else:
        return r'combined_df.xlsx'


def get_output_dir_name():
    return r'Output_'


def get_output_path(params, allow_override=True):
    if allow_override and hasattr(params, 'output_path_override') and params.output_path_override is not None:
        return params.output_path_override
    return os.path.join(params.img_dir, get_output_dir_name())


def get_output_df_path(params, allow_override=True):
    if allow_override and hasattr(params, 'output_path_override') and params.output_path_override is not None:
        return params.output_df_path_override
    return os.path.join(get_output_path(params), get_output_df_basename(params))


def get_cvt_obj_path(params):
    return os.path.join(get_output_path(params), 'cvt.pickle')


def argv_proc(argv, sysargv):
    if argv is None:
        if len(sysargv) == 1:
            raise ValueError('No config file specified.')
        ag = sysargv[1:]
    else:
        ag = argv
    return ag


def read_df(path):
    """
    Adaptive reading of excel or csv files
    :param path:
    :return:
    """
    path_ext = os.path.splitext(path)[1]
    if path_ext == ".xlsx" or path_ext == ".xls":
        df = pd.read_excel(path, index_col=0)
    elif path_ext == ".csv":
        df = pd.read_csv(path, index_col=0)
    else:
        raise FileNotFoundError(
            f"Excel or csv file not found in the given path. Check file type and extension. extension is {path_ext}")
    return df


def read_effective_df(path):
    df = read_df(path)
    df = to_effective_df(df)
    return df


def to_effective_df(df):
    return df[np.logical_and(df.uneven == False, df.dark == False)]


def write_df(df: pd.DataFrame, path: str):
    path_ext = os.path.splitext(path)[1]
    if path_ext == ".xlsx" or path_ext == ".xls":
        df.to_excel(path)
    elif path_ext == ".csv":
        df.to_csv(path)


def get_output_meta_basename():
    return 'meta.json'


def get_output_meta_path(params, allow_override=True):
    return os.path.join(get_output_path(params, allow_override), get_output_meta_basename())


def write_output_meta(params: ParamLoader, meta_dict, allow_override=True):
    with open(get_output_meta_path(params, allow_override), 'w+') as outfile:
        json.dump(meta_dict, outfile)


def load_output_meta(params, allow_override=True):
    with open(get_output_meta_path(params, allow_override), 'r') as outfile:
        json.load(outfile)
