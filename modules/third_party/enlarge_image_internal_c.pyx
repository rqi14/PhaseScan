#roll_ball_c.pyx
#!python
#cython: language_level=3
import numpy as np
cimport numpy as np


cimport cython

def enlarge_image_internal_cython(small_img, float_img, width, height, x_s_indices, y_s_indices, x_weights, y_weights, s_width):
    return _enlarge_image_internal_c(small_img, float_img, width, height, np.array(x_s_indices), np.array(y_s_indices), np.array(x_weights), np.array(y_weights), s_width)

@cython.cdivision
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray _enlarge_image_internal_c(double[:] small_img, np.ndarray float_img_, long width, long height, long[:] x_s_indices, long[:] y_s_indices, double[:] x_weights, double[:] y_weights, long s_width):
        cdef np.ndarray line0_d, line1_d
        cdef int x,y, s_y_ptr, y_s_line0, p
        cdef double[:] line0, line1
        cdef bint swap = False
        cdef np.ndarray float_img_d
        cdef double[:] float_img
        cdef double weight
        line0_d = np.zeros(width, dtype=np.float64)
        line1_d = np.zeros(width, dtype=np.float64)
        line0=line0_d
        line1=line1_d
        float_img_d = np.copy(float_img_)
        float_img=float_img_d

        for x in range(0, width):
            line1[x] = small_img[x_s_indices[x]] * x_weights[x] + \
                       small_img[x_s_indices[x] + 1] * (1.0 - x_weights[x])
        y_s_line0 = -1
        for y in range(0, height):
            if y_s_line0 < y_s_indices[y]:
                line0, line1 = line1, line0
                y_s_line0 += 1
                s_y_ptr = int((y_s_indices[y] + 1) * s_width)
                for x in range(0, width):
                    line1[x] = small_img[s_y_ptr + x_s_indices[x]] * x_weights[x] + \
                               small_img[s_y_ptr + x_s_indices[x] + 1] * (1.0 - x_weights[x])
            weight = y_weights[y]
            p = y * width
            for x in range(0, width):
                float_img[p] = line0[x] * weight + line1[x] * (1.0 - weight)

                p += 1
        return float_img_d