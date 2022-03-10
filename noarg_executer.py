from enum import Enum, unique, auto

import cvt_int_conc
import detect_quantify_export
import fitting_svm
import gen_int_conc_converter

# import warnings
# warnings.simplefilter('error')

@unique
class ExecuteMode(Enum):
    PROCESS = auto()  # detect quantify and export
    FIT_DOUBLE = auto()  # fit boundary using feature true and false both
    GEN_CVT = auto()
    CVT = auto()
    ALL = auto()

if __name__ == '__main__':
    # --------- CHANGE HERE ----------
    # mode = ExecuteMode.PROCESS
    # mode = ExecuteMode.GEN_CVT
    # mode = ExecuteMode.CVT
    mode = ExecuteMode.ALL
    # mode = ExecuteMode.FIT_DOUBLE
    # mode = ExecuteMode.MANUAL
    config_paths = [
        r"H:\temp\210115_FUS_PEG_3reps_forsubmission\Minimum dataset\sa_params.jsonc"
    ]
    # --------------------------------
    for config_path in config_paths:
        if mode == ExecuteMode.PROCESS:
            detect_quantify_export.main([config_path])
        elif mode == ExecuteMode.FIT_DOUBLE:
            fitting_svm.main([config_path])
        elif mode == ExecuteMode.GEN_CVT:
            gen_int_conc_converter.main([config_path])
        elif mode == ExecuteMode.CVT:
            cvt_int_conc.main([config_path])
        elif mode == ExecuteMode.ALL:
            gen_int_conc_converter.main([config_path])
            detect_quantify_export.main([config_path])
            cvt_int_conc.main([config_path])
            fitting_svm.main([config_path])
        else:
            print('Nothing executed')
