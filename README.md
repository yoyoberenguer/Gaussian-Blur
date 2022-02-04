# Gaussian-Blur

Python implementation of 2D Gaussian blur filter methods using multiprocessing

**WIKIPEDIA**

In image processing, a Gaussian blur (also known as Gaussian smoothing) is the result of blurring an image by a Gaussian function (named after mathematician and scientist Carl Friedrich Gauss). It is a widely used effect in graphics software, typically to reduce image noise and reduce detail. The visual effect of this blurring technique is a smooth blur resembling that of viewing the image through a translucent screen, distinctly different from the bokeh effect produced by an out-of-focus lens or the shadow of an object under usual illumination. Gaussian smoothing is also used as a pre-processing stage in computer vision algorithms in order to enhance image structures at different scales—see scale space representation and scale space implementation.
Mathematically, applying a Gaussian blur to an image is the same as convolving the image with a Gaussian function. This is also known as a two-dimensional Weierstrass transform. By contrast, convolving by a circle (i.e., a circular box blur) would more accurately reproduce the bokeh effect. Since the Fourier transform of a Gaussian is another Gaussian, applying a Gaussian blur has the effect of reducing the image's high-frequency components; a Gaussian blur is thus a low pass filter.

A box blur (also known as a box linear filter) is a spatial domain linear filter in which each pixel in the resulting image has a value equal to the average value of its neighboring pixels in the input image. It is a form of low-pass ("blurring") filter. A 3 by 3 box blur can be written as 1/9 * determinant matrix:

![alt text](https://github.com/yoyoberenguer/Gaussian-Blur/blob/master/boxblur.png)



**Example with 5x5 convolution kernel and blurbox9x9 method**

![alt text](https://github.com/yoyoberenguer/Gaussian-Blur/blob/master/Assets/Graphics/Gaussian.png)


If you are interested in very fast blur algorithm kernel 5x5, please check the file
bloom.pyx from the project <<light effect improved>> 
https://github.com/yoyoberenguer/Light-effect-improved/blob/master/bloom.pyx

I developped two types of 5x5 blur techniques (using multiprocessing and cython performances) 

The first technique is using a C-buffer type data as input image.
Methods below are repectively used for 24 and 32 bit image format.
Both methods returns a pygame.Surface (blurred image)
```python
# bloom.pyx line 174 : 
cpdef blur5x5_buffer24(rgb_buffer, width, height, depth, mask=None):
    return blur5x5_buffer24_c(rgb_buffer, width, height, depth, mask=None)
    
# bloom.pyx line 177 :
cpdef blur5x5_buffer32(rgba_buffer, width, height, depth, mask=None):
    return blur5x5_buffer32_c(rgba_buffer, width, height, depth, mask=None)
```

Second method takes a numpy 3d array as input image and return a pygame Surface 

```python
bloon.pyx line 180 :
cpdef blur5x5_array24(rgb_array_, mask=None):
    return blur5x5_array24_c(rgb_array_, mask=None)

bloom.pyx line 183 :
cpdef blur5x5_array32(rgb_array_, mask=None):
    return blur5x5_array32_c(rgb_array_, mask=None)
```
    
e.g : blur5x5_array24_c

```python 
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef unsigned char [:, :, ::1] blur5x5_array24_c(unsigned char [:, :, :] rgb_array_, mask=None):
    
    # # Gaussian kernel 5x5
    #    # |1   4   6   4  1|
    #    # |4  16  24  16  4|
    #    # |6  24  36  24  6|  x 1/256
    #    # |4  16  24  16  4|
    #    # |1  4    6   4  1|
    # This method is using convolution property and process the image in two passes,
    # first the horizontal convolution and last the vertical convolution
    # pixels convoluted outside image edges will be set to adjacent edge value
    
    # :param rgb_array_: numpy.ndarray type (w, h, 3) uint8 
    # :return: Return 24-bit a numpy.ndarray type (w, h, 3) uint8
    


    cdef int w, h, dim
    try:
        w, h, dim = (<object>rgb_array_).shape[:3]

    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    # kernel 5x5 separable
    cdef:
        # float [::1] kernel = kernel_
        float[5] kernel = [1.0/16.0, 4.0/16.0, 6.0/16.0, 4.0/16.0, 1.0/16.0]
        short int kernel_half = 2
        unsigned char [:, :, ::1] convolve = numpy.empty((w, h, 3), dtype=uint8)
        unsigned char [:, :, ::1] convolved = numpy.empty((w, h, 3), dtype=uint8)
        short int kernel_length = len(kernel)
        int x, y, xx, yy
        float k, r, g, b, s
        char kernel_offset
        unsigned char red, green, blue

    with nogil:
        # horizontal convolution
        for y in prange(0, h, schedule=METHOD, num_threads=THREADS):  # range [0..h-1)

            for x in range(0, w):  # range [0..w-1]

                r, g, b = 0, 0, 0

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = kernel[kernel_offset + kernel_half]

                    xx = x + kernel_offset

                    # check boundaries.
                    # Fetch the edge pixel for the convolution
                    if xx < 0:
                        red, green, blue = rgb_array_[0, y, 0],\
                        rgb_array_[0, y, 1], rgb_array_[0, y, 2]
                    elif xx > (w - 1):
                        red, green, blue = rgb_array_[w-1, y, 0],\
                        rgb_array_[w-1, y, 1], rgb_array_[w-1, y, 2]
                    else:
                        red, green, blue = rgb_array_[xx, y, 0],\
                            rgb_array_[xx, y, 1], rgb_array_[xx, y, 2]

                    r = r + red * k
                    g = g + green * k
                    b = b + blue * k

                convolve[x, y, 0], convolve[x, y, 1], convolve[x, y, 2] = <unsigned char>r,\
                    <unsigned char>g, <unsigned char>b

        # Vertical convolution
        for x in prange(0,  w, schedule=METHOD, num_threads=THREADS):

            for y in range(0, h):
                r, g, b = 0, 0, 0

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = kernel[kernel_offset + kernel_half]
                    yy = y + kernel_offset

                    if yy < 0:
                        red, green, blue = convolve[x, 0, 0],\
                        convolve[x, 0, 1], convolve[x, 0, 2]
                    elif yy > (h -1):
                        red, green, blue = convolve[x, h-1, 0],\
                        convolve[x, h-1, 1], convolve[x, h-1, 2]
                    else:
                        red, green, blue = convolve[x, yy, 0],\
                            convolve[x, yy, 1], convolve[x, yy, 2]

                    r = r + red * k
                    g = g + green * k
                    b = b + blue * k

                convolved[x, y, 0], convolved[x, y, 1], convolved[x, y, 2] = \
                    <unsigned char>r, <unsigned char>g, <unsigned char>b

    return convolved
```
