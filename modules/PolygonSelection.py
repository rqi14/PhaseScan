# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 04:45:07 2020

@author: qirun
"""
import cv2
import numpy as np
import tkinter
from modules.proc_utils import img_preprocess, _parse_2bgr, _parse_8bit


class CapillaryThresh:
    def __init__(self, window_name, img, shade_color=(0, 255, 0), shade_transparency=0.2):
        self._window_name = window_name
        self._img = img
        self.shade_color = shade_color
        self.shade_transparency = shade_transparency


class PolygonSelector:
    __KEY_ESC = 27
    __KEY_X = 120
    __KEY_CAPITAL_X = 88
    __KEY_C = 99
    __KEY_CAPITAL_C = 67
    __KEY_ENTER = 13
    __KEY_R = 114
    __KEY_CAPITAL_R = 82
    __KEY_L = 108
    __KEY_CAPITAL_L = 76

    def __init__(self, window_name, img, refresh_rate=20, line_color=(0, 255, 0), shade_color=(0, 255, 0),
                 shade_transparency=0.2):
        """
        Initialise a polygon selector. self.is_completed shows if the selection is made successfully. The result can be
        retrieved from self._points_internal. The end status can be checked using is_completed. If false then it is not completed

        Parameters
        ----------
        window_name : str
            Window name to be displayed.
        img : np.ndarray
            Image upon which the selection will be made.
        refresh_rate : int, optional
            Refresh rate of the window. The default is 20.
        line_color : Tuple, optional
            Line color of the selected polygon region. The default is (0,255,0).
        shade_color : Tuple, optional
            Shade color of the selected polygon region. The default is (0,255,0).
        shade_transparency : float, optional
            Shading transparency. The default is 0.2.

        Returns
        -------
        None.

        """
        root = tkinter.Tk()
        self._display_width = root.winfo_screenwidth()
        self._display_height = root.winfo_screenheight()
        root.destroy()
        self._scale_factor = 1
        # self._scale_down_offset = 50  # pixels

        self.window_name = str(window_name)  # Name for our window
        self._img = img.copy()
        self._init_vars()
        self._img_disp = _parse_2bgr(_parse_8bit(img))

        h, w = self._img.shape
        hsf = h / self._display_height
        wsf = w / self._display_width
        hwsf_min = min(hsf, wsf)
        if hwsf_min > 1:
            # calculate int scale factor
            self._scale_factor = int(2 ** np.ceil(np.log2(hwsf_min)))
            print(f'Scale factor {self._scale_factor}')
            self._img_disp = cv2.resize(self._img_disp, ( int(w / self._scale_factor), int(h / self._scale_factor)))
        self._interval = int(1000 / refresh_rate)
        self.line_color = line_color
        self.shade_color = shade_color
        self.shade_transparency = shade_transparency

    def _init_vars(self):
        self._terminated = False
        self.is_completed = False
        self.cursor_pos = (0, 0)
        self._points_internal = []
        self._disp_log = False

    def _switch_disp(self):
        img_source = self._img

        if not self._disp_log:
            self._disp_log = True
            self._img_disp = _parse_2bgr(img_preprocess(img_source, 0, 100, 0, True))
        else:
            self._disp_log = False
            self._img_disp = _parse_2bgr(_parse_8bit(img_source))
        if self._scale_factor != 1:
            h, w = self._img.shape
            self._img_disp = cv2.resize(self._img_disp, (int(w / self._scale_factor), int(h / self._scale_factor)))

    def _on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.is_completed:  # Nothing more to do
            return
        if event == cv2.EVENT_MOUSEMOVE:
            self.cursor_pos = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self._points_internal.append((x, y))
            print(f"\rPoint {len(self._points_internal)} added {(self._points_internal[-1])}", end="")
            # sys.stdout.flush()
            # time.sleep(0.5)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._terminated = True
            if len(self._points_internal) >= 3:
                print(f"\rSelection successful, {len(self._points_internal)} _points_internal selected.", end="")
                self.is_completed = True
            else:
                print(f"\rSelection not completed, only {len(self._points_internal)} _points_internal selected.",
                      end="")

    def run(self):
        if self._terminated:  # if already terminated (window closed or esc), reinitialise variables
            self._init_vars()

        """
        Start the polygon selection process

        Returns
        -------
        None.

        """
        doc = """Please select an ROI by clicking corners of the desired polygon.
Right click to finish.
X to remove the last selected point.
C to clear all selected point.
Right click or ESC to finish selection.
L to view the enhanced image (log processed).
After the selection, you will see a review window, where the ROI is shaded on the original image in green.
R to reselect the polygon.
Enter or ESC to save the selection.
        """
        print(doc)

        self._execute()

    def _execute(self):
        self._run_select_window()
        if self.is_completed:
            review_flag = self._run_result_window()  # if accept
            if not review_flag:  # if not accept (re-select)
                self._init_vars()
                self._execute()
        else:
            # If select is not completed, close the window
            cv2.destroyWindow(self.window_name)

    def _run_select_window(self):
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_NORMAL)
        cv2.imshow(self.window_name, self._img)
        # cv2.resizeWindow(self.window_name, (300, 300))
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

        while not self._terminated:
            # Check if window is closed:
            if not cv2.getWindowProperty(self.window_name, 0) >= 0:
                self._terminated = True
                continue

            plot = self._img_disp.copy()
            if len(self._points_internal) > 0:
                cv2.polylines(plot, np.array([self._points_internal]), False, self.line_color, 2)
                cv2.line(plot, self._points_internal[-1], self.cursor_pos, self.line_color)
            cv2.imshow(self.window_name, plot)
            key_code = cv2.waitKey(self._interval)

            if key_code == self.__KEY_ESC:  #
                self._terminated = True
            elif key_code == self.__KEY_C or key_code == self.__KEY_CAPITAL_C:
                self._points_internal = []
            elif key_code == self.__KEY_X or key_code == self.__KEY_CAPITAL_X:
                if len(self._points_internal) > 0:
                    self._points_internal.pop()
            elif key_code == self.__KEY_L or key_code == self.__KEY_CAPITAL_L:
                self._switch_disp()

    def _draw_result(self):
        polygon_layer = self._img_disp.copy()
        img_layer = self._img_disp.copy()
        cv2.fillPoly(polygon_layer, np.array([self._points_internal]), self.shade_color)
        shaded_img = cv2.addWeighted(polygon_layer, self.shade_transparency, img_layer, 1 - self.shade_transparency, 0)
        return shaded_img

    def _run_result_window(self):
        """
        Display result and allow re-selection

        Returns bool
        -------
        True if accepted, False if reselect.

        """

        shaded_img = self._draw_result()
        review_terminated = False
        while not review_terminated:
            if not cv2.getWindowProperty(self.window_name, 0) >= 0:
                review_terminated = True
                return True
            cv2.imshow(self.window_name, shaded_img)
            key_code = cv2.waitKey()
            if key_code == self.__KEY_ESC or key_code == self.__KEY_ENTER:
                review_terminated = True
                cv2.destroyWindow(self.window_name)
                return True
            elif key_code == self.__KEY_R or key_code == self.__KEY_CAPITAL_C:
                print('\rreselect                                     ', end="")
                review_terminated = True
                return False
            elif key_code == self.__KEY_L or key_code == self.__KEY_CAPITAL_L:
                self._switch_disp()
                shaded_img = self._draw_result()

    @property
    def points(self):
        return [tuple(xyv * self._scale_factor for xyv in point) for point in self._points_internal]

# if __name__ == "__main__":
#     img_original = imreadmulti_mean(
#         r'C:\Users\qirun\Documents\DataAnalysis\200715\phasescan_wells\exp2\fluor_cal\488\488_0.5.tif')
#     # img_uint8=map_array(img_original,np.uint8)
#     polys = PolygonSelector('Select calib region', img_original)
#     polys.run()
