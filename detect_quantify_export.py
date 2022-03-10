import os
import sys

from modules.FileProcessingTools import SignalFileProcessingHelper, IterativeSignalFileProcessingHelper
from modules.ParamLoading import ParamLoader
from modules.SphereAnalysis import SphereAnalyser
from modules.SphereAnalysis2 import SphereAnalyser2
from modules.SphereAnalysis3 import SphereAnalyser3
from modules.SphereAnalysis4 import SphereAnalyser4
from modules.SphereAnalysis5 import SphereAnalyser5
# from modules.CylinderAnalysis import CylinderAnalyser
# from modules.SphereAnalysisA import SphereAnalyserA
from modules.utils import argv_proc, get_output_path, get_output_df_path, read_effective_df, get_output_df_basename
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib



def main(argv=None):
    ag = argv_proc(argv, sys.argv)
    params = ParamLoader(ag[0])
    if hasattr(params, 'stddev_scan') and params.stddev_scan is not None and 'scan_enabled' in params.stddev_scan and \
            params.stddev_scan['scan_enabled']:
        stddev_scan(params)
    else:
        hp = SignalFileProcessingHelper(params, verbose=params.verbose_plot,
                                                 **{"analyser": globals()[
                                                     params.analyser]} if params.analyser is not None else {})
        hp.run()


def stddev_scan(params: ParamLoader):
    # initialise a number of processes with different stddev settings.
    # stddev_tolerance must be set explicitly in this mode
    print('Scan mode enabled because setting found in config: "stddev_scan": {"scan_enabled" : true}')
    stddev_min = params.stddev_scan['stddev_min']
    stddev_max = params.stddev_scan['stddev_max']
    stddev_scan_range = np.linspace(stddev_min, stddev_max, params.stddev_scan['scan_num'])
    # params_stddev_list = [copy.deepcopy(params) for i in stddev_scan_range]

    # for i in range(len(stddev_scan_range)):
    #     stddev = stddev_scan_range[i]
    #     print(f'Scanning stddev_tolerance = {stddev}')
    #     params_stddev = params_stddev_list[i]
    #     params_stddev.kwargs['stddev_tolerance'] = stddev
    #     params_stddev.output_path_override = _output_path_method_override_stddev(params_stddev)
    #     hp = SignalFileProcessingHelper(params_stddev, verbose=params_stddev.verbose_plot,
    #                                     **{"analyser": globals()[params_stddev.analyser]}
    #                                     if params_stddev.analyser is not None else {})
    #     hp.run()

    params = copy.deepcopy(params)
    hp = IterativeSignalFileProcessingHelper(params, verbose=params.verbose_plot,
                                             **{"analyser": globals()[
                                                 params.analyser]} if params.analyser is not None else {})
    hp.init_analysers()
    hp.detect_shape()
    print('Label uneven')
    hp.execute_foreach(_label_uneven)
    print('Label dark')
    hp.execute_foreach(_label_dark)

    # scan through stddev tolerance range
    true_list = np.zeros(stddev_scan_range.shape)
    false_list = np.zeros(stddev_scan_range.shape)
    for i in range(len(stddev_scan_range)):
        stddev = stddev_scan_range[i]
        print(f'Scanning stddev_tolerance = {stddev}')

        hp.output_dir_method = lambda prms: os.path.join(get_output_path(prms, allow_override=False),
                                                         f'stddev_{stddev}')
        hp.output_df_path_method = lambda prms: os.path.join(hp.output_dir_method(prms), get_output_df_basename(params))
        hp.execute_foreach(_process_stddev_change, std_dev=stddev)
        hp.save_sub_dfs()
        hp.combine_data()
        hp.plot_labelled()
        if hp.verbose:
            hp.plot_masked_circles()
        df = read_effective_df(hp.output_df_path_method(params))
        true_list[i]=df[df.feature==True].shape[0]
        false_list[i]=df[df.feature==False].shape[0]
        plt.close('all')

    plt.figure('stddev scan')
    plt.plot(stddev_scan_range, true_list, label="number of true features")
    plt.plot(stddev_scan_range, false_list, label="number of false features")
    plt.legend()
    plt.show(block=False)
    plt.savefig(os.path.join(get_output_path(params), 'stddev_scan.png'))

    # print('Initialising analysers')
    # self.init_analysers()
    # for i in self.get_iter_collection():
    #     print(f'Processing {self.analyser_num[i]}')
    #     self.detect_shape(i)
    #     self.label(i)
    #     self.plot_labelled(i)
    #     if self.verbose:
    #         print(f'Plotting masked regions for {self.analyser_num[i]}. Verbose enabled')
    #         self.plot_masked_circles(i)
    #     self.save_sub_dfs(i)
    # self.combine_data()

    # count feature true and false numbers in each
    # true_list = np.zeros(stddev_scan_range.shape)
    # false_list = np.zeros(stddev_scan_range.shape)
    # for i in range(len(stddev_scan_range)):
    #     params_stddev = params_stddev_list[i]
    #     df = read_effective_df(get_output_df_path(params_stddev))
    #     true_list[i]=df[df.feature].shape[0]
    #     false_list[i]=df[np.bitwise_not(df.feature)].shape[0]
    #
    # plt.figure('stddev scan')
    # plt.plot(stddev_scan_range, true_list, label="number of true features")
    # plt.plot(stddev_scan_range, false_list, label="number of false features")
    # plt.legend()
    # plt.show()


def _process_stddev_change(analyser: SphereAnalyser, a_num, std_dev: float):
    analyser.stddev_tolerance = std_dev
    analyser.label_features()

def _label_uneven(analyser, a_num):
    analyser.label_uneven()

def _label_dark(analyser, a_num):
    analyser.label_dark()

def _output_path_method_override_stddev(params):
    stddev = params.kwargs['stddev_tolerance']
    return os.path.join(get_output_path(params, allow_override=False),
                        f'stddev_{stddev}')


if __name__ == "__main__":
    main()
