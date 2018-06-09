"""
--------------------------------------------------------------------------------------------------------------------

This code comes with a MIT license.

Copyright (c) 2018 Yoann Berenguer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

Please acknowledge and give reference if using the source code for your project

--------------------------------------------------------------------------------------------------------------------

"""

import pygame
import numpy
from numpy import putmask
import multiprocessing
from multiprocessing import Process, Queue
import hashlib
import colorsys
import time
import os
import math

__author__ = "Yoann Berenguer"
__copyright__ = "Copyright 2007."
__credits__ = ["Yoann Berenguer"]
__license__ = "MIT License"
__version__ = "1.0.0"
__maintainer__ = "Yoann Berenguer"
__email__ = "yoyoberenguer@hotmail.com"
__status__ = "Demo"


class Gaussian3x3(Process):
    """ 2D Gaussian blur filter.
        Example with 3 x 3 convolution kernel
        Separable Horizontal / Vertical, thanks 2D Gaussian filter kernel is separable as it
        can be expressed as the outer product of two vectors.
        The filter can be split into two passes, horizontal and vertical, each with O(n) complexity per
        pixel (where n is the kernel size)
        Each pixel in the image gets multiplied by the Gaussian kernel.
        This is done by placing the center pixel of the kernel on the image pixel
        and multiplying the values in the original image with the pixels in the
        kernel that overlap. The values resulting from these multiplications are
        added up and that result is used for the value at the destination pixe
    """

    def __init__(self, listener_name_, data_, out_, event_):
        super(Process, self).__init__()
        self.out_ = out_
        self.data = data_
        self.listener_name = listener_name_
        print('Gaussian3x3 %s started ' % (listener_name_))
        self.event = event_
        self.stop = False
        # kernel 3x3 separable
        self.kernel_v = numpy.array(([0.25, 0.5, 0.25]))  # vertical vector
        self.kernel_h = numpy.array(([0.25, 0.5, 0.25]))  # horizontal vector

    # Horizontal
    @staticmethod
    def convolution_h(shape: tuple, surface_: pygame.Surface, source_array_: numpy.ndarray, kernel_: numpy.ndarray):

        amount = pygame.math.Vector3(0, 0, 0)
        for y in range(0, shape[1]):

            for x in range(0, shape[0]):

                accumulator = pygame.math.Vector3(0, 0, 0) + amount
                try:
                    for kernel_offset in range(0, 2):
                        xx = x + kernel_offset

                        # if xx < shape.shape[0]:
                        v_color = pygame.math.Vector3(*surface_.get_at((xx, y))[:3])
                        amount = v_color * kernel_[kernel_offset + 1]
                        accumulator += amount

                    source_array_[x][y] = accumulator
                except IndexError:
                    pass
        return source_array_

    # Vertical
    @staticmethod
    def convolution_v(shape: tuple, surface_: pygame.Surface, source_array_: numpy.ndarray, kernel_: numpy.ndarray):

            amount = pygame.math.Vector3(0, 0, 0)
            for y in range(0, shape[1]):

                for x in range(0, shape[0]):

                    accumulator = pygame.math.Vector3(0, 0, 0) + amount
                    try:
                        for kernel_offset in range(0, 2):
                                yy = y + kernel_offset

                                # if yy < shape.shape[1]:

                                v_color = pygame.math.Vector3(*surface_.get_at((x, yy))[:3])

                                amount = v_color * kernel_[kernel_offset + 1]
                                accumulator += amount

                        source_array_[x][y] = accumulator
                    except IndexError:
                        pass

            return source_array_

    def run(self):
        while not self.event.is_set():

            if self.data[self.listener_name] is not None:
                rgb_array = self.data[self.listener_name]
                surface_ = pygame.surfarray.make_surface(rgb_array)
                N = 10  # increase N for a stronger blur
                for r in range(N):

                    source_array_ = numpy.zeros([rgb_array.shape[0],
                                                 rgb_array.shape[1], 3], dtype=numpy.uint8)

                    surface_ = pygame.surfarray.make_surface(
                        self.convolution_h(rgb_array.shape, surface_, source_array_, self.kernel_h))

                    source_array_ = self.convolution_v(
                        rgb_array.shape, surface_, source_array_, self.kernel_v)

                # Send the data throughout the QUEUE
                self.out_.put({self.listener_name: source_array_})
                # Delete the job from the list (self.data).
                # This will place the current process in idle
                self.data[self.listener_name] = None
            else:
                # Minimize the CPU utilization while the process
                # is listening.
                time.sleep(0.001)
        print('Gaussian3x3 %s is dead.' % self.listener_name)


class Sharpen(Process):

    def __init__(self, listener_name_, data_, out_, event_):
        super(Process, self).__init__()
        self.out_ = out_
        self.data = data_
        self.listener_name = listener_name_
        print('Sharpen %s started with pid %s ' % (listener_name_, os.getpid()))
        self.event = event_
        self.stop = False
        self.sharpen_kernel = \
            numpy.array(([0, -1, 0],
                         [-1, 28, -1],
                         [0, -1, 0])).astype(dtype=numpy.float)
        self.sharpen_kernel *= 1/24
        self.kernel_half = 1

    def run(self):
        while not self.event.is_set():
            # if data are present in the list
            if self.data[self.listener_name] is not None:
                rgb_array = self.data[self.listener_name]
                surface = pygame.surfarray.make_surface(rgb_array)

                source_array_ = numpy.zeros([rgb_array.shape[0],
                                             rgb_array.shape[1], 3], dtype=numpy.uint8)

                color = pygame.Color(0, 0, 0, 0)

                for y in range(0, rgb_array.shape[1]):

                    for x in range(0, rgb_array.shape[0]):

                        r, g, b = 0, 0, 0

                        for kernel_offset_y in range(-self.kernel_half, self.kernel_half + 1):

                            for kernel_offset_x in range(-self.kernel_half, self.kernel_half + 1):

                                xx = x + kernel_offset_x
                                yy = y + kernel_offset_y

                                if (0 < xx < rgb_array.shape[0]) and (0 < yy < rgb_array.shape[1]):
                                    color = surface.get_at((xx, yy))

                                k = self.sharpen_kernel[kernel_offset_y + self.kernel_half,
                                                                      kernel_offset_x + self.kernel_half]
                                r += color[0] * k
                                g += color[1] * k
                                b += color[2] * k

                        source_array_[x][y] = (r, g, b)

                # Send the data throughout the QUEUE
                self.out_.put({self.listener_name: source_array_})
                # Delete the job from the list (self.data).
                # This will place the current process in idle
                self.data[self.listener_name] = None

            # Minimize the CPU utilization while the process
            # is listening.
            else:
                time.sleep(0.001)
        print('Gaussian_Blur5x5 %s is dead.' % self.listener_name)


class GaussianBlur3x3(Process):
    """ 2D Gaussian blur filter with 3x3 kernel.
        Each pixel in the image gets multiplied by the Gaussian kernel. """
    def __init__(self, listener_name_, data_, out_, event_):
        super(Process, self).__init__()
        self.out_ = out_
        self.data = data_
        self.listener_name = listener_name_
        print('GaussianBlur3x3 %s started with pid %s ' % (listener_name_, os.getpid()))
        self.event = event_
        self.stop = False
        self.x = 0
        self.y = 0

        self.kernel_gaussian_blur3x3 = \
            numpy.array(([[1, 2, 1],
                          [2, 4, 2],
                          [1, 2, 1]])).astype(dtype=numpy.float)

        self.kernel_gaussian_blur3x3 *= 1 / 16
        print(self.kernel_gaussian_blur3x3)
        self.kernel_half = 1

    def run(self):
        while not self.event.is_set():
            # if data is present in the list
            if self.data[self.listener_name] is not None:
                rgb_array = self.data[self.listener_name]
                surface = pygame.surfarray.make_surface(rgb_array)

                source_array_ = numpy.zeros([rgb_array.shape[0],
                                             rgb_array.shape[1], 3], dtype=numpy.uint8)

                color = pygame.Color(0, 0, 0, 0)

                for y in range(0, rgb_array.shape[1]):

                    for x in range(0, rgb_array.shape[0]):

                        r, g, b = 0, 0, 0

                        for kernel_offset_y in range(-self.kernel_half, self.kernel_half + 1):

                            for kernel_offset_x in range(-self.kernel_half, self.kernel_half + 1):

                                xx = x + kernel_offset_x
                                yy = y + kernel_offset_y

                                if (0 < xx < rgb_array.shape[0]) and (0 < yy < rgb_array.shape[1]):
                                    color = surface.get_at((xx, yy))

                                k = self.kernel_gaussian_blur3x3[kernel_offset_y + self.kernel_half,
                                                                               kernel_offset_x + self.kernel_half]
                                r += color[0] * k
                                g += color[1] * k
                                b += color[2] * k

                        source_array_[x][y] = (r, g, b)

                # Send the data throughout the QUEUE
                self.out_.put({self.listener_name: source_array_})
                # Delete the job from the list (self.data).
                # This will place the current process in idle
                self.data[self.listener_name] = None

            # Minimize the CPU utilization while the process
            # is listening.
            else:
                time.sleep(0.001)
        print('GaussianBlur3x3 %s is dead.' % self.listener_name)


class GaussianBlur5x5(Process):
    """ 2D Gaussian blur filter with 5x5 kernel.
        Each pixel in the image gets multiplied by the Gaussian kernel. """
    def __init__(self, listener_name_, data_, out_, event_):
        super(Process, self).__init__()
        self.out_ = out_
        self.data = data_
        self.listener_name = listener_name_
        print('Gaussian_Blur5x5 %s started with pid %s ' % (listener_name_, os.getpid()))
        self.event = event_
        self.stop = False
        # Gaussian blur 5x5  1/256
        self.kernel_gaussian_blur5x5 = \
            numpy.array(([1, 4, 6, 4, 1],
                         [4, 16, 24, 16, 4],
                         [6, 24, 36, 24, 6],
                         [4, 16, 24, 16, 4],
                         [1, 4, 6, 4, 1])).astype(dtype=numpy.float)
        self.kernel_gaussian_blur5x5 *= 1 / 256
        print(self.kernel_gaussian_blur5x5)
        self.kernel_half = 2

    def run(self):
        while not self.event.is_set():
            # if data is present in the list
            if self.data[self.listener_name] is not None:
                rgb_array = self.data[self.listener_name]
                surface = pygame.surfarray.make_surface(rgb_array)

                source_array_ = numpy.zeros([rgb_array.shape[0],
                                             rgb_array.shape[1], 3], dtype=numpy.uint8)

                color = pygame.Color(0, 0, 0, 0)

                for y in range(0, rgb_array.shape[1]):

                    for x in range(0, rgb_array.shape[0]):

                        r, g, b = 0, 0, 0

                        for kernel_offset_y in range(-self.kernel_half, self.kernel_half + 1):

                            for kernel_offset_x in range(-self.kernel_half, self.kernel_half + 1):

                                xx = x + kernel_offset_x
                                yy = y + kernel_offset_y

                                if (0 < xx < rgb_array.shape[0]) and (0 < yy < rgb_array.shape[1]):
                                    color = surface.get_at((xx, yy))

                                k = self.kernel_gaussian_blur5x5[kernel_offset_y + self.kernel_half,
                                                                               kernel_offset_x + self.kernel_half]
                                r += color[0] * k
                                g += color[1] * k
                                b += color[2] * k

                        source_array_[x][y] = (r, g, b)

                # Send the data throughout the QUEUE
                self.out_.put({self.listener_name: source_array_})
                # Delete the job from the list (self.data).
                # This will place the current process in idle
                self.data[self.listener_name] = None

            # Minimize the CPU utilization while the process
            # is listening.
            else:
                time.sleep(0.001)
        print('Gaussian_Blur5x5 %s is dead.' % self.listener_name)


class BoxBlur3x3(Process):

    def __init__(self, listener_name_, data_, out_, event_):
        super(Process, self).__init__()
        self.out_ = out_
        self.data = data_
        self.listener_name = listener_name_
        print('Box_blur3x3 %s started with pid %s ' % (listener_name_, os.getpid()))
        self.event = event_
        self.stop = False
        self.x = 0
        self.y = 0
        self.kernel_size = 6
        self.kernel_half = 1

    def run(self):
        while not self.event.is_set():
            # if data is present in the list
            if self.data[self.listener_name] is not None:
                rgb_array = self.data[self.listener_name]
                surface = pygame.surfarray.make_surface(rgb_array)

                source_array_ = numpy.zeros([rgb_array.shape[0],
                                             rgb_array.shape[1], 3])

                for x in range(0, rgb_array.shape[0]):

                    for y in range(0, rgb_array.shape[1]):
                        source_array_[x][y % rgb_array.shape[1]] = pygame.transform.average_color(surface,
                                                                    (x - self.kernel_half,
                                                                     y - self.kernel_half,
                                                                     self.kernel_size,
                                                                     self.kernel_size))[:3]

                # Send the data throughout the QUEUE
                self.out_.put({self.listener_name: source_array_})
                # Delete the job from the list (self.data).
                # This will place the current process in idle
                self.data[self.listener_name] = None
                # print('Blur_Kernel_5x5 %s complete ' % self.listener_name)

            # Minimize the CPU utilization while the process
            # is listening.
            else:
                time.sleep(0.001)
        print('Box_blur3x3 %s is dead.' % self.listener_name)


class SplitSurface:

    def __init__(self, process_: int, array_, queue, check_: bool = False):

        assert isinstance(process_, int), \
            'Expecting an int for argument process_, got %s ' % type(process_)
        assert isinstance(array_, numpy.ndarray), \
            'Expecting numpy.ndarray for argument array_, got %s ' % type(array_)
        assert isinstance(check_, bool), \
            'Expecting bool for argument check_, got %s ' % type(check_)

        self.process = process_  # Process number
        self.shape = array_.shape  # array shape
        self.col, self.row, self.c = tuple(self.shape)  # Columns, Rows, colors
        self.pixels = array_.size / self.c  # Pixels [w x h]
        self.size = array_.size  # Array size [w x h x colors]
        self.array = array_  # surface (numpy.array)
        self.queue = queue
        self.split_array = []
        # self.split()
        self.split_non_equal()

        # Checking hashes (input array & output)
        # Below works only if the array is rebuild
        # if check_:
        #    self.hash = hashlib.md5()
        #    self.hash.update(array_.copy('C'))
        #    self.hash_ = hashlib.md5()
        #    self.hash_.update(self.split_array.copy('C'))
        #    assert self.hash.hexdigest() == self.hash_.hexdigest(), \
        #        '\n[-] Secure hashes does not match.'

    def split_equal(self):
        # Split array into multiple sub-arrays of equal size.
        self.split_array = numpy.split(self.array, self.process)
        self.queue.put(self.split_array)

    def split_non_equal(self):
        # Split an array into multiple sub-arrays of equal or near-equal size.
        #  Does not raise an exception if an equal division cannot be made.
        split_ = numpy.array_split(self.array, self.process, 1)
        # self.split_array = numpy.vstack((split_[i] for i in range(self.process)))
        self.queue.put(split_)

    def split(self):

        split_array = []

        # chunk size (d_column, d_row) for a given number of process.
        d_column, d_row = self.col // self.process, self.row
        # cut chunks of data from the surface (numpy 3D array).
        print(d_column, d_row)
        split_size = 0
        # Summing the chunks and calculate the remainder if any.
        for i in range(self.process):
            split_array.append(self.array[0:d_row, i * d_column:i * d_column + d_column])
            split_size += split_array[i]

        remainder = int((self.pixels - self.process * d_column * d_row))

        # Remainder not null --> Adding the remainder to the last chunk
        if remainder != 0:
            split_array[self.process - 1] = self.array[0:d_row,
                                            (self.process - 1) * d_column:(self.process - 1) * d_column + d_column + (
                                                    remainder // d_row)]

        self.queue.put(split_array)

        # rebuild complete array
        # self.split_array = numpy.hstack((split_array[i] for i in range(self.process)))
        # self.queue.put(self.split_array)


if __name__ == '__main__':
    # Map size
    SIZE = (500, 400)
    SCREENRECT = pygame.Rect((0, 0), SIZE)
    pygame.init()
    SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.RESIZABLE, 32)
    TEXTURE1 = pygame.image.load("Assets\\Graphics\\seychelles.jpg").convert()
    TEXTURE1 = pygame.transform.smoothscale(TEXTURE1, (SIZE[0], SIZE[1]//2))

    TIMINGS = []

    array = pygame.surfarray.pixels3d(TEXTURE1.copy())

    PROCESS = 6 # multiprocessing.cpu_count()

    QUEUE_OUT = multiprocessing.Queue()
    QUEUE_IN = multiprocessing.Queue()
    EVENT = multiprocessing.Event()

    MANAGER = multiprocessing.Manager()

    DATA = MANAGER.dict()

    SplitSurface(PROCESS, array, QUEUE_IN)
    new_array = QUEUE_IN.get()

    i = 0
    for array in new_array:
        DATA[i] = array
        i += 1

    t1 = time.time()
    for i in range(PROCESS):
        # BoxBlur3x3(i, DATA, QUEUE_OUT, EVENT).start()
        GaussianBlur5x5(i, DATA, QUEUE_OUT, EVENT).start()
        # Sharpen(i, DATA, QUEUE_OUT, EVENT).start()
        # Gaussian3x3(i, DATA, QUEUE_OUT, EVENT).start()
        # GaussianBlur3x3(i, DATA, QUEUE_OUT, EVENT).start()

    FRAME = 0
    clock = pygame.time.Clock()
    STOP_GAME = False
    PAUSE = False

    while not STOP_GAME:

        pygame.event.pump()

        while PAUSE:
            event = pygame.event.wait()
            keys = pygame.key.get_pressed()
            if keys[pygame.K_PAUSE]:
                PAUSE = False
                pygame.event.clear()
                keys = None
            break

        for event in pygame.event.get():

            keys = pygame.key.get_pressed()

            if event.type == pygame.QUIT or keys[pygame.K_ESCAPE]:
                print('Quitting')
                STOP_GAME = True

            elif event.type == pygame.MOUSEMOTION:
                MOUSE_POS = event.pos

            elif keys[pygame.K_PAUSE]:
                PAUSE = True
                print('Paused')

        t1 = time.time()

        i = 0
        for array in new_array:
            DATA[i] = array
            i += 1

        temp = {}
        for i in range(PROCESS):
            for key, value in QUEUE_OUT.get().items():
                temp[str(key)] = value
        sort = []
        for key, value in sorted(temp.items(), key=lambda item: (int(item[0]), item[1])):
            sort.append(value)

        split_array = numpy.hstack(sort[r] for r in range(PROCESS))

        # print('\n[+] Surface reconstruction')
        surface = pygame.surfarray.make_surface(split_array)

        print('\n[+] time : ', time.time() - t1)
        TIMINGS.append(time.time() - t1)

        SCREEN.fill((0, 0, 0, 0))
        SCREEN.blit(TEXTURE1, (0, 0))
        SCREEN.blit(surface, (0, SIZE[1]//2))

        pygame.display.flip()
        TIME_PASSED_SECONDS = clock.tick(120)
        FRAME += 1

    EVENT.set()

    sum_ = 0
    for r in TIMINGS:
        sum_ += r
    avg = sum_ / len(TIMINGS)

    time.sleep(2)
    print(TIMINGS)
    print('\nAverage : ', avg)
    pygame.quit()
