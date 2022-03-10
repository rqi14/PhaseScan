import os
import sys

import matplotlib.pyplot as plt

from modules.DataLoading import DataLoader
from modules.IntensityConcentrationConversion import IntensityConcentrationConverter
from modules.quadrant_utils import get_cropped_img_set
from modules.utils import get_output_path, get_cvt_obj_path, argv_proc

"""
This script calculates calibration LUT using capillary images.
Two types of data structure supported.
Type 1: separated capillary images
The images should be named as [Channel name]_[Concentration].[Format], e.g. 488_0.1.tif
If the background images were taken for individual capillary image, they should be named as follows.
The background images should be named as [Channel name]_[Concentration]_bkg.[Format], .e.g 488_0.1_bkg.tif
If background images for individual capillary do no exist, it automatically uses the background image specified in the
config file.
In this mode, the ROI must be selected for each of the images.
If you want to ignore some images, simply move them out of the folder, or rename them in a way that will not be detected
Type 2: combined PDMS images (PDMS device with multiple channels)
The images should be named as follows:
[Channel name]_[Concentration1]_[Concentration2]..._[Concentrationn]_some-extra-info_h[Height of device].[Format]
Note: h[Height of device] is optional for this type of images. If this does not exist, it will automatically use the
height specified in the config file.
In this mode, if there is some concentration that does not exist, simply press ESC or close the window to skip it.

Final data structure in the pickle
if ch_names[i] = 488
{488} means the index of 488 in ch_names
in 488 channel value_matrix[{488}] = [list of 488 concentration values]
int_matrix[{488}][{647}] = [list of the 647 intensity with corresponding 488 concentrations]

value_matrix [[1,2,3],[4,5,6,7,8]]
int_matrix [[[1,2,3],[4,5,6]], [[1,2,3,4,5],[2,3,4,5,6]]]
"""


def main(argv = None):
    ag = argv_proc(argv, sys.argv)
    data_set = DataLoader(ag[0])
    print(f'use config file {ag[0]}')
    params = data_set.params
    cvt_options = params.cvt_kwargs
    if params.quadrant_mode:
        cvt_options["quadrant_split"] = lambda x: get_cropped_img_set(x, params)
    cvt = IntensityConcentrationConverter(params.int_conc_cvt_dir,
                                          params.channels,
                                          data_set.get_lumi_corrs(),
                                          params.cvt_depth,
                                          params.cvt_pixel_size,
                                          ch_bkg_paths=params.bkg_path if params.quadrant_mode else params.bkg_paths,
                                          **params.cvt_kwargs)
    cvt_pickle_path = get_cvt_obj_path(params)
    cvt.serialise(cvt_pickle_path)
    print(f'cvt saved to {cvt_pickle_path}')

    # plot concentration (value) matrix vs intensity matrix
    img_output_dir = get_output_path(params)
    for ch_dye in range(len(params.channels)):
        for ch_img in range(len(params.channels)):
            handle = plt.figure(num=f"{ch_dye} dye in {ch_img} image")
            plt.plot(cvt.value_matrix[ch_dye], cvt.int_matrix[ch_dye][ch_img], marker="o", label="data")
            plt.xlabel('Dye concentration')
            plt.ylabel('Intensity per unit volume')
            if cvt.linear:
                plt.plot(cvt.value_matrix[ch_dye], cvt.interp_matrix[ch_dye][ch_img](cvt.value_matrix[ch_dye]), linestyle="--", label="fit")
            plt.legend()
            save_name = f"fluorophore_in_{params.channels[ch_dye]}_imaged_in_{params.channels[ch_img]}_channel.png"
            handle.savefig(os.path.join(img_output_dir, save_name))


if __name__ == "__main__":
    main()

