#roll_ball_c.pyx
#!python
#cython: language_level=3
import numpy as np
cimport numpy as np


cimport cython

def roll_ball_cython(float_img, height, width, ball_data, ball_width):
    z_ball = np.array(ball_data)
    return roll_ball_c(float_img, z_ball, height, width, ball_width)

@cython.cdivision
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray roll_ball_c (np.ndarray fimg, double[:] z_ball, int height, int width, int ball_width):
    cdef int radius
    # cdef np.ndarray[np.float64_t, ndim=1, mode='c'] cache
    cdef double[:] cache
    cdef int next_line_to_write, next_line_to_read, src, dest, y0, y_ball, y_ball0, y_end, x0, x_ball0, x_end, cache_pointer, bp
    # cdef z_reduced
    cdef int y, x, yp, p
    cdef double z, z_min, z_reduced
    cdef double[:] float_img
    float_img_data = np.copy(fimg)
    float_img = float_img_data
    radius = int(ball_width / 2)
    cache = np.zeros((width * ball_width), dtype=np.float64)

    for y in range(-radius, height + radius):
        next_line_to_write = (y + radius) % ball_width
        next_line_to_read = y + radius
        if next_line_to_read < height:
            src = next_line_to_read * width
            dest = next_line_to_write * width
            cache[dest:dest+width] = float_img[src:src+width]
            float_img[src:src+width] = -2.3e308

        y0 = max((0, y - radius))
        y_ball0 = y0 - y + radius
        y_end = y + radius
        if y_end >= height:
            y_end = height - 1
        for x in range(-radius, width+radius):
            z = 2.3e308
            x0 = max((0, x - radius))
            x_ball0 = x0 - x + radius
            x_end = x + radius
            if x_end >= width:
                x_end = width - 1

            y_ball = y_ball0
            for yp in range(y0, y_end + 1):
                cache_pointer = (yp % ball_width) * width + x0
                bp = x_ball0 + y_ball * ball_width
                for xp in range(x0, x_end + 1):
                    z_reduced = cache[cache_pointer] - z_ball[bp]
                    if z > z_reduced:
                        z = z_reduced
                    cache_pointer += 1
                    bp += 1
                y_ball += 1

            y_ball = y_ball0
            for yp in range(y0, y_end + 1):
                p = x0 + yp * width
                bp = x_ball0 + y_ball * ball_width
                for xp in range(x0, x_end + 1):
                    z_min = z + z_ball[bp]
                    if float_img[p] < z_min:
                        float_img[p] = z_min
                    p += 1
                    bp += 1
                y_ball += 1
    return float_img_data


