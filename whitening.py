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
# save also the background
$ python whitening.py input.jpg whitened.jpg -b background.jpg -k 100
```

Select kernel size that's enough for not making artifacts while small enough
to keep computation fast. A good starting point is 50 pixels.
"""

from __future__ import print_function
import argparse
import time

import numpy as np
from PIL import Image
from skimage.filters import median
from skimage.morphology import square

def whiten(image, kernel_size):
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
    kernel = square(kernel_size)
    background = np.dstack([median(image[:, :, i], selem=kernel)
        for i in range(channels)])
    whitened = (image.astype(np.float32) / background.astype(np.float32))
    whitened = np.minimum(whitened, 1)
    whitened = (whitened * 255).astype(np.uint8)
    return whitened, background

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
def timed_whiten(image, kernel_size):
    return whiten(image, kernel_size)

def parse_args():
    parser = argparse.ArgumentParser(description='Whitens an image.')
    parser.add_argument('image', metavar='INPUT', help='input image path')
    parser.add_argument('whitened', metavar='WHITENED', help='whitened output image path')
    parser.add_argument('-b', '--background', help='background output image path')
    parser.add_argument('-k', '--kernel-size', type=int, default=50,
        help='size of the 2D median filter kernel')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    image = np.asarray(Image.open(args.image), dtype='uint8')
    whitened, background = timed_whiten(image, kernel_size=args.kernel_size)
    Image.fromarray(whitened).save(args.whitened, 'jpeg')
    if args.background is not None:
        Image.fromarray(background).save(args.background, 'jpeg')
