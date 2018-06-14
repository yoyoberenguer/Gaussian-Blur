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
import multiprocessing
from multiprocessing import Process, Queue, freeze_support
import hashlib
import time
import os

__author__ = "Yoann Berenguer"
__copyright__ = "Copyright 2007."
__credits__ = ["Yoann Berenguer"]
__license__ = "MIT License"
__version__ = "1.0.0"
__maintainer__ = "Yoann Berenguer"
__email__ = "yoyoberenguer@hotmail.com"
__status__ = "Demo"


class GaussianBoxBlur11x11(Process):
    """ 2D Gaussian blur filter.
        Example with 11 x 11 convolution kernel
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
        self.out_ = out_    # Queue for sending data
        self.data = data_   # Queue containing current data to process
        self.listener_name = listener_name_  # Process name
        print('GaussianBoxBlur %s started ' % listener_name_)
        self.event = event_  # multiprocess event to use for cancelling aborting threads from upper level
        self.stop = False   # same but local variable
        # kernel 5x5 separable
        self.kernel_v = numpy.array(([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))  # vertical vector
        self.kernel_h = numpy.array(([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))  # horizontal vector
        self.kernel_half = 5
        self.kernel_length = len(self.kernel_h)

        # Horizontal
    def convolution_h(self):

            for y in range(self.kernel_half, self.shape[1] - self.kernel_half):
                lock = False
                r, g, b = (0, 0, 0)
                for x in range(self.kernel_half, self.shape[0] - self.kernel_half):

                    if not lock:
                        for kernel_offset in range(-self.kernel_half, self.kernel_half + 1):
                            try:

                                xx = x + kernel_offset
                                color = self.surface.get_at((xx, y))
                                r += color[0]
                                g += color[1]
                                b += color[2]
                            except IndexError:
                                r += 128
                                g += 128
                                b += 128
                            finally:
                                lock = True
                    else:
                            c1 = self.surface.get_at((x + kernel_offset, y))
                            c2 = self.surface.get_at((x - kernel_offset - 1, y))
                            r += c1[0] - c2[0]
                            g += c1[1] - c2[1]
                            b += c1[2] - c2[2]

                    self.source_array[x - self.kernel_half][y - self.kernel_half] = (r/self.kernel_length,
                                                                                    g/self.kernel_length,
                                                                                    b/self.kernel_length)
            return self.source_array

        # Vertical
    def convolution_v(self):

            for x in range(self.kernel_half, self.shape[0] - self.kernel_half):
                lock = False
                r, g, b = 0, 0, 0

                for y in range(self.kernel_half, self.shape[1] - self.kernel_half):

                    if not lock:
                        for kernel_offset in range(-self.kernel_half, self.kernel_half + 1):
                            try:
                                yy = y + kernel_offset
                                color = self.surface.get_at((x, yy))
                                r += color[0]
                                g += color[1]
                                b += color[2]
                            except IndexError:
                                r += 128
                                g += 128
                                b += 128
                            finally:
                                lock = True
                    else:
                            c1 = self.surface.get_at((x, y + kernel_offset))
                            c2 = self.surface.get_at((x, y - kernel_offset - 1))
                            r += c1[0] - c2[0]
                            g += c1[1] - c2[1]
                            b += c1[2] - c2[2]

                    self.source_array[x - self.kernel_half][y - self.kernel_half] = (r / self.kernel_length,
                                                                                         g / self.kernel_length,
                                                                                         b / self.kernel_length)
            return self.source_array

    def run(self):
        while not self.event.is_set():

            if self.data[self.listener_name] is not None:
                rgb_array = self.data[self.listener_name]

                # Surface with extra padding
                self.surface = pygame.surfarray.make_surface(rgb_array)

                # Shape of the array with extra padding
                self.shape = rgb_array.shape

                self.source_array = numpy.zeros((self.shape[0] - self.kernel_half * 2,
                                                 self.shape[1] - self.kernel_half * 2, 3))
                # vertical_convo = self.convolution_v()
                # self.source_array = numpy.zeros((self.shape[0]-8, self.shape[1]-8, 3))
                # self.surface = pygame.surfarray.make_surface(vertical_convo)
                source_array_ = self.convolution_h()

                # print(self.listener_name, rgb_array.shape, source_array_.shape)
                # Send the data throughout the QUEUE
                self.out_.put({self.listener_name: source_array_})
                # Delete the job from the list (self.data).
                # This will place the current process in idle
                self.data[self.listener_name] = None
            else:
                # Minimize the CPU utilization while the process
                # is listening.
                time.sleep(0.001)
        print('GaussianBoxBlur11x11 %s is dead.' % self.listener_name)



class SplitSurface:

    def __init__(self, process_: int, array_, queue, check_: bool = False, padding_=None):

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
        self.padding = padding_
        self.split()
        # self.split_non_equal()

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
        d_column, d_row = self.col, self.row // self.process
        # cut chunks of data from the surface (numpy 3D array).

        split_size = 0
        # Summing the chunks and calculate the remainder if any.
        for i_ in range(self.process):
            padding_rlow = 0
            padding_rhigh = self.padding
            if i_ != 0:
                # adding extra padding on the top edge (extra pixels).
                # The quantity is determine by the kernel size, e.g for a 9 x 9 kernel
                # size, the extra padding would be 4 pixels to add on the top edge and bottom edge etc.
                # The padding can also be extended to the left/right side
                padding_rlow = self.padding  # padding for the top edge (row only)
                padding_rhigh = self.padding  # padding for the bottom edge ( row only)

            # split the array in almost equal sizes and adding extra padding on the edges.
            split_array.append(numpy.array(self.array[0:d_column + 5, d_row * i_
                                                         - padding_rlow: d_row * i_ + d_row + padding_rhigh, :]))
            # split_size += split_array[i]

        #remainder = int((self.pixels - self.process * d_column * d_row))

        # Remainder not null --> Adding the remainder to the last chunk
        #if remainder != 0:
        #    split_array[self.process - 1] = self.array[0:d_row,
        #                                    (self.process - 1) * d_column:(self.process - 1) * d_column + d_column + (
        #                                            remainder // d_row)]
        self.queue.put(split_array)


if __name__ == '__main__':
    freeze_support()
    # Map size
    SIZE = (800, 600)
    SCREENRECT = pygame.Rect((0, 0), SIZE)
    pygame.init()
    SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.RESIZABLE, 32)
    TEXTURE1 = pygame.image.load("Assets\\Graphics\\seychelles.jpg").convert()
    TEXTURE1 = pygame.transform.smoothscale(TEXTURE1, (SIZE[0], SIZE[1] >> 1))
    padding = 5
    PADDING = pygame.transform.smoothscale(TEXTURE1, (SIZE[0] + padding * 2, (SIZE[1] >> 1) + padding * 2))

    TIMINGS = []

    array = pygame.surfarray.pixels3d(PADDING.copy())

    PROCESS = 9  # multiprocessing.cpu_count()

    QUEUE_OUT = multiprocessing.Queue()
    QUEUE_IN = multiprocessing.Queue()
    EVENT = multiprocessing.Event()

    MANAGER = multiprocessing.Manager()

    DATA = MANAGER.dict()

    SplitSurface(PROCESS, array, QUEUE_IN, False, padding)
    new_array = QUEUE_IN.get()

    i = 0
    for array in new_array:
        DATA[i] = array
        i += 1

    t1 = time.time()

    for i in range(PROCESS):

        GaussianBoxBlur11x11(i, DATA, QUEUE_OUT, EVENT).start()

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
        SCREEN.blit(surface, (0, SIZE[1] // 2))

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
