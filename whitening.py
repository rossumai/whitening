"""
Tries to separate text/line foreground and background by 2D median filter
whitening.

Example usage:

Python API:

```
from PIL import Image

from whitening import whiten

# possible to use numpy array as input/output
image = np.asarray(Image.open('image.jpg'), dtype='uint8')
foreground, background = whiten(image, kernel_size=50, downsample=4)
Image.fromarray(foreground).save('foreground.jpg', 'jpeg')

# or directly a PIL image
image = Image.open('image.jpg')
foreground, background = whiten(image, kernel_size=50, downsample=4)
foreground.save('foreground.jpg', 'jpeg')
```

CLI:
```
$ python whitening.py --help

# whiten an image and save the foreground output
$ python whitening.py input.jpg foreground.jpg

# specify the kernel size
$ python whitening.py input.jpg foreground.jpg -k 100

# work in grayscale instead of RGB (3x faster)
$ python whitening.py input.jpg foreground.jpg -g

# downsample the image 4x (faster, but a bit less precise)
$ python whitening.py input.jpg foreground.jpg -d 4

# save also the background
$ python whitening.py input.jpg foreground.jpg -b background.jpg
```

Select kernel size that's enough for not making artifacts while small enough
to keep computation fast. A good starting point is 50 pixels.

A 9.5 Mpx image can be processed on a MacBook in 15 s, with grayscale and
downsampling 4x the run time can be reduced to 1 s! Quite good results can be
obtained even with kernel size 10 and downsampling 16x.
"""

from __future__ import print_function, division
import argparse
import time

import numpy as np
from PIL import Image
from skimage.color import rgb2gray
import skimage.filters
import skimage.morphology
import skimage.transform

def whiten(image, kernel_size, downsample=1):
    """
    Tries to separate text/line foreground and background by 2D median filter
    whitening.

    The idea is that foreground (text/lines) are spikes that can be removed by
    spatial median filter, thus leaving the background. We can then normalize
    the original image by the background, leaving the foreground.

    Input:
    `image` - input image (np.ndarray or PIL.Image.Image)
    `kernel_size` - width of the median filter kernel
    `downsample` - downsampling factor to speedup the median calculation,
    can be useful since background is usually low-frequency image

    All images are represented as a numpy array of shape (height, width,
    channels) and dtype uint8 or PIL.Image.Image.

    Output: `foreground`, `background`
    """
    input_is_image = issubclass(type(image), Image.Image)
    if input_is_image:
        # RGB/RGBA images can be converted without copying
        # L (grayscale) images must be copied to avoid
        # https://github.com/cython/cython/issues/1605
        image = np.array(image, copy=image.mode == 'L')

    is_grayscale = len(image.shape) < 3
    if is_grayscale:
        image = image.reshape(image.shape + (1,))

    channels = image.shape[-1]
    # apply 2D median filter on each channel separately
    input_image = image
    shape = np.array(image.shape)

    if downsample != 1:
        downsampled_shape = (shape[:2] // downsample) + (channels,)
        resized = skimage.transform.resize(image, downsampled_shape)
        input_image = (resized * 255).astype(np.uint8)

    kernel = skimage.morphology.square(kernel_size)
    background = np.dstack([
        skimage.filters.median(input_image[:, :, i], selem=kernel)
        for i in range(channels)])

    if downsample != 1:
        background = (skimage.transform.resize(background, shape) * 255).astype(np.uint8)

    foreground = (image.astype(np.float32) / background.astype(np.float32))
    foreground = np.minimum(foreground, 1)
    foreground = (foreground * 255).astype(np.uint8)

    if is_grayscale:
        foreground = foreground[:,:,0]
        background = background[:,:,0]

    if input_is_image:
        foreground = Image.fromarray(foreground)
        background = Image.fromarray(background)

    return foreground, background

def to_grayscale(image):
    return (rgb2gray(image) * 255).astype(np.uint8)

def to_rgb(image):
    if image.shape[-1] == 1:
        return np.broadcast_to(image, image.shape[:2] + (3,))
    else:
        return image

def timeit(method):
    """
    Decorator to measure run time of a method.
    """
    def timed(*args, **kw):
        start = time.time()
        result = method(*args, **kw)
        end = time.time()
        print('Run time: %2.2f sec' % (end - start))
        return result

    return timed

@timeit
def timed_whiten(*args, **kwargs):
    return whiten(*args, **kwargs)

def parse_args():
    parser = argparse.ArgumentParser(description='Whitens an image.')
    parser.add_argument('image', metavar='INPUT', help='input image path')
    parser.add_argument('foreground', metavar='FOREGROUND', help='foreground output image path')
    parser.add_argument('-b', '--background', help='background output image path')
    parser.add_argument('-k', '--kernel-size', type=int, default=50,
        help='size of the 2D median filter kernel')
    parser.add_argument('-d', '--downsample', type=int, default=1,
        help='downsampling factor')
    parser.add_argument('-g', '--grayscale', action='store_true', default=False,
        help='convert to grayscale (faster)')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    image = np.asarray(Image.open(args.image), dtype='uint8')
    if args.grayscale:
        image = to_grayscale(image)
    foreground, background = timed_whiten(image, kernel_size=args.kernel_size,
        downsample=args.downsample)
    if args.grayscale:
        foreground = to_rgb(foreground)
        background = to_rgb(background)
    Image.fromarray(foreground).save(args.foreground, 'jpeg')
    if args.background is not None:
        Image.fromarray(background).save(args.background, 'jpeg')
