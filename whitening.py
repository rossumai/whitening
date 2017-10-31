"""
Tries to separate text/line foreground and background by 2D median filter
whitening.

Example usage:

Python API:

```
from PIL import Image
from whitening import whiten

image = np.asarray(Image.open('image.jpg'), dtype='uint8')
whitened, background = whiten(image, kernel_size=50)
Image.fromarray(whitened).save('whitened.jpg', 'jpeg')
```

CLI:
```
$ python whitening.py --help

# whiten an image and save the whitened output
$ python whitening.py input.jpg whitened.jpg

# specify the kernel size
$ python whitening.py input.jpg whitened.jpg -k 100

# work in grayscale instead of RGB (3x faster)
$ python whitening.py input.jpg whitened.jpg -g

# downsample the image 4x (faster, but a bit less precise)
$ python whitening.py input.jpg whitened.jpg -d 4

# save also the background
$ python whitening.py input.jpg whitened.jpg -b background.jpg
```

Select kernel size that's enough for not making artifacts while small enough
to keep computation fast. A good starting point is 50 pixels.

A 9.5 Mpx image can be processed on a MacBook in 15 s, with grayscale and
downsampling 4x the run time can be reduced to 1 s!
"""

from __future__ import print_function, division
import argparse
import time

import numpy as np
from PIL import Image
from skimage.filters import median
from skimage.morphology import square
from skimage.transform import resize

def whiten(image, kernel_size, downsample=1):
    """
    Tries to separate text/line foreground and background by 2D median filter
    whitening.

    The idea is that foreground (text/lines) are spikes that can be removed by
    spatial median filter, thus leaving the background. We can then normalize
    the original image by the background, leaving the foreground.

    Input:
    `image` - input image
    `kernel_size` - width of the median filter kernel

    All images are represented as a numpy array of shape (height, width,
    channels) and dtype uint8.

    Output: `whitened` (foreground), `background`
    """
    channels = image.shape[-1]
    # apply 2D median filter on each channel separately
    input_image = image
    shape = np.array(image.shape)
    if downsample != 1:
        downsampled_shape = (shape[:2] // downsample) + (channels,)
        input_image = (resize(image, downsampled_shape) * 255).astype(np.uint8)
        print('input_image.shape', input_image.shape)
    kernel = square(kernel_size)
    background = np.dstack([
        median(input_image[:, :, i], selem=kernel)
        for i in range(channels)])
    if downsample != 1:
        background = (resize(background, shape) * 255).astype(np.uint8)
    whitened = (image.astype(np.float32) / background.astype(np.float32))
    whitened = np.minimum(whitened, 1)
    whitened = (whitened * 255).astype(np.uint8)
    return whitened, background

def to_grayscale(image):
    return image.mean(axis=-1).reshape(image.shape[:2] + (1,)).astype(image.dtype)

def to_rgb(image):
    if image.shape[-1] == 1:
        return np.broadcast_to(image, image.shape[:2] + (3,))

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
    parser.add_argument('whitened', metavar='WHITENED', help='whitened output image path')
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
    whitened, background = timed_whiten(image, kernel_size=args.kernel_size,
        downsample=args.downsample)
    if args.grayscale:
        whitened = to_rgb(whitened)
        background = to_rgb(background)
    Image.fromarray(whitened).save(args.whitened, 'jpeg')
    if args.background is not None:
        Image.fromarray(background).save(args.background, 'jpeg')
