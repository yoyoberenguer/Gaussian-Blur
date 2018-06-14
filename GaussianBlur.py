import pygame
import numpy
import time
import math

__author__ = "Yoann Berenguer"
__copyright__ = "Copyright 2007."
__credits__ = ["Yoann Berenguer"]
__license__ = "MIT License"
__version__ = "1.0.0"
__maintainer__ = "Yoann Berenguer"
__email__ = "yoyoberenguer@hotmail.com"
__status__ = "Demo"


class GaussianBoxBlur9x9_method2:

    def __init__(self, surface_, shape_):

        # kernel 5x5 separable
        self.kernel_v = numpy.array(([1, 1, 1, 1, 1, 1, 1, 1, 1]))  # vertical vector
        self.kernel_h = numpy.array(([1, 1, 1, 1, 1, 1, 1, 1, 1]))  # horizontal vector
        self.kernel_half = 4
        # Surface with extra padding
        self.surface = surface_
        # Shape of the array with extra padding
        self.shape = shape_

        self.source_array = numpy.zeros((self.shape[0], self.shape[1], 3))
        self.kernel_length = len(self.kernel_h)

    # Horizontal
    def convolution_h(self):

        for y in range(self.kernel_half, self.shape[1] - self.kernel_half):
            lock = False
            r, g, b = 0, 0, 0
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

                self.source_array[x - self.kernel_half][y - self.kernel_half] = (r / self.kernel_length,
                                                                                 g / self.kernel_length,
                                                                                 b / self.kernel_length)

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

    def convolutions(self):
        vertical_convo = self.convolution_v()
        self.surface = pygame.surfarray.make_surface(vertical_convo)
        return self.convolution_h()


class GaussianBoxBlur9x9_method1:

    def __init__(self, surface_, shape_):

        # kernel 5x5 separable
        self.kernel_v = numpy.array(([1, 1, 1, 1, 1, 1, 1, 1, 1]))  # vertical vector
        self.kernel_h = numpy.array(([1, 1, 1, 1, 1, 1, 1, 1, 1]))  # horizontal vector
        self.kernel_half = 4
        self.surface = surface_
        self.shape = shape_
        self.source_array = numpy.zeros((self.shape[0], self.shape[1], 3))
        self.kernel_length = len(self.kernel_h)
        self.array = pygame.surfarray.array3d(surface_)

    # Horizontal
    def convolution_h(self):
        self.source_array = numpy.zeros((self.shape[0], self.shape[1], 3))
        
        for y in range(0, self.shape[1]):

            for x in range(0, self.shape[0]):

                r, g, b = 0, 0, 0
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
                self.source_array[x][y] = (r / self.kernel_length,
                                           g / self.kernel_length,
                                           b / self.kernel_length)
        return self.source_array

    # Vertical
    def convolution_v(self):
        self.source_array = numpy.zeros((self.shape[0], self.shape[1], 3))

        for y in range(0, self.shape[1]):

            for x in range(0, self.shape[0]):
                r, g, b = 0, 0, 0
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
                self.source_array[x][y] = (r / self.kernel_length,
                                           g / self.kernel_length,
                                           b / self.kernel_length)
        return self.source_array

    def convolutions(self):
        vertical_convo = self.convolution_v()
        self.surface = pygame.surfarray.make_surface(vertical_convo)
        return self.convolution_h()



class GaussianBlur5x5:

    def __init__(self, surface_, shape_):

        # kernel 5x5 separable
        self.kernel_v = numpy.array(([1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16]))  # vertical vector
        self.kernel_h = numpy.array(([1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16]))  # horizontal vector

        self.kernel_half = 2
        self.surface = surface_
        self.shape = shape_
        self.source_array = numpy.zeros((self.shape[0], self.shape[1], 3))

    # Horizontal
    def convolution_h(self):

        for y in range(0, self.shape[1]):

            for x in range(0, self.shape[0]):
                r, g, b = 0, 0, 0

                for kernel_offset in range(-self.kernel_half, self.kernel_half + 1):
                    try:
                        xx = x + kernel_offset
                        k = self.kernel_h[kernel_offset + self.kernel_half]
                        color = self.surface.get_at((xx, y))
                        r += color[0] * k
                        g += color[1] * k
                        b += color[2] * k

                    except IndexError:
                        k = self.kernel_v[kernel_offset + self.kernel_half]
                        r += 128 * k
                        g += 128 * k
                        b += 128 * k

                self.source_array[x][y] = (r, g, b)

        return self.source_array

    # Vertical
    def convolution_v(self):

        for y in range(0, self.shape[1]):

            for x in range(0, self.shape[0]):
                r, g, b = 0, 0, 0
                for kernel_offset in range(-self.kernel_half, self.kernel_half + 1):
                    try:
                        yy = y + kernel_offset
                        color = self.surface.get_at((x, yy))
                        k = self.kernel_v[kernel_offset + self.kernel_half]
                        r += color[0] * k
                        g += color[1] * k
                        b += color[2] * k

                    except IndexError:
                        k = self.kernel_v[kernel_offset + self.kernel_half]
                        r += 128 * k
                        g += 128 * k
                        b += 128 * k

                self.source_array[x][y] = (r, g, b)

        return self.source_array

    def convolutions(self):
        vertical_convo = self.convolution_v()
        self.surface = pygame.surfarray.make_surface(vertical_convo)
        return self.convolution_h()


if __name__ == '__main__':
    numpy.set_printoptions(threshold=numpy.nan)

    SIZE = (800, 600)
    SCREENRECT = pygame.Rect((0, 0), SIZE)
    pygame.init()
    SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.RESIZABLE, 32)
    TEXTURE1 = pygame.image.load("Assets\\Graphics\\seychelles.jpg").convert()
    TEXTURE1 = pygame.transform.smoothscale(TEXTURE1, (SIZE[0], SIZE[1] >> 1))
    # Texture re-scale to create extra data (padding) on each sides
    PADDING = pygame.transform.smoothscale(TEXTURE1, (SIZE[0] + 8, (SIZE[1] >> 1) + 8))
    # 2.5 seconds 800x300
    # Gauss = GaussianBoxBlur9x9_method2(PADDING, PADDING.get_size())
    # 6 seconds 800x300
    Gauss = GaussianBoxBlur9x9_method1(TEXTURE1, TEXTURE1.get_size())
    # 8.7 seconds 800x300
    # Gauss = GaussianBlur5x5(TEXTURE1, TEXTURE1.get_size())

    t = time.time()
    # array = Gauss.convolutions()
    array = Gauss.convolutions()
    print(time.time() - t)

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

        surface = pygame.surfarray.make_surface(array)

        SCREEN.fill((0, 0, 0, 0))
        SCREEN.blit(TEXTURE1, (0, 0))
        SCREEN.blit(surface, (0, SIZE[1] // 2))

        pygame.display.flip()
        TIME_PASSED_SECONDS = clock.tick(120)
        FRAME += 1

    pygame.quit()
