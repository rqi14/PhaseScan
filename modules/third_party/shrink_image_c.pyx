#roll_ball_c.pyx
#!python
#cython: language_level=3
import numpy as np
cimport numpy as np

cimport cython

def shrink_image_cython(img, shrink_factor, height, width):
    return _shrink_image_c(img,shrink_factor,height, width)

@cython.cdivision
@cython.boundscheck(False)
@cython.wraparound(False)
cdef _shrink_image_c(np.ndarray img, int shrink_factor, int height, int width):
    # height, width = self.height, self.width
    cdef int s_height, s_width
    cdef int[:] float_img
    cdef np.ndarray img_copy_d, small_img_d
    cdef int x_mask_min, y_mas_min
    cdef int x,y
    cdef double[:,:] small_img
    cdef double[:,:] img_copy
    cdef double min_value
    cdef double [:,:] tmp

    s_height, s_width = int(height / shrink_factor), int(width / shrink_factor)

    img_copy_d = img.reshape((height, width))
    img_copy = img_copy_d
    small_img_d = np.ones((s_height, s_width), np.float64)
    small_img = small_img_d

    for y in range(0, s_height):
        for x in range(0, s_width):
            x_mask_min = shrink_factor * x
            y_mask_min = shrink_factor * y
            tmp = img_copy[y_mask_min:y_mask_min + shrink_factor,
                        x_mask_min:x_mask_min + shrink_factor]
            min_value = np.min(tmp)
            small_img[y, x] = min_value
    return small_img_d.reshape(s_height * s_width)

# This is already working
# cdef _shrink_image_c(np.ndarray img, int shrink_factor, int height, int width):
#     # height, width = self.height, self.width
#     cdef int s_height, s_width
#     cdef int[:] float_img
#     cdef np.ndarray img_copy_d, small_img_d
#     cdef int x_mask_min, y_mas_min, min_value
#     cdef int x,y
#     cdef double[:] small_img
#     cdef double[:] img_copy
#
#     s_height, s_width = int(height / shrink_factor), int(width / shrink_factor)
#
#     img_copy_d = img.reshape((height, width)).copy()
#     img_copy = img_copy_d
#     small_img_d = np.ones((s_height, s_width), np.float64)
#     small_img = small_img_d
#
#     for y in range(0, s_height):
#         for x in range(0, s_width):
#             x_mask_min = shrink_factor * x
#             y_mask_min = shrink_factor * y
#             min_value = np.min(img_copy.base[y_mask_min:y_mask_min + shrink_factor,
#                         x_mask_min:x_mask_min + shrink_factor])
#             small_img.base[y, x] = min_value
#     return small_img_d.reshape(s_height * s_width)
