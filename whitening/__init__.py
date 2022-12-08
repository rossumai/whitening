"""
Tries to separate text/line foreground and background by 2D median filter
whitening.

Example usage:

```
import PIL.Image

from whitening import whiten

# possible to use numpy array as input/output
image = np.asarray(PIL.Image.open('image.jpg'), dtype='uint8')
foreground, background = whiten(image, kernel_size=20, downsample=4)
PIL.Image.fromarray(foreground).save('foreground.jpg', 'jpeg')

# or directly a PIL image
image = PIL.Image.open('image.jpg')
foreground, background = whiten(image, kernel_size=20, downsample=4)
foreground.save('foreground.jpg', 'jpeg')
```

Select kernel size that's enough for not making artifacts while small enough
to keep computation fast. A good starting point is 50 pixels.

A 9.5 Mpx image can be processed on a MacBook in 15 s, with grayscale and
downsampling 4x the run time can be reduced to 1 s! Quite good results can be
obtained even with kernel size 10 and downsampling 16x.

More info: http://bohumirzamecnik.cz/blog/2015/image-whitening/
"""

import PIL.Image
import numpy as np
import skimage.filters
import skimage.morphology
import skimage.transform
from skimage.color import rgb2gray


def whiten(image, kernel_size=10, downsample=1):
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
    input_is_image = issubclass(type(image), PIL.Image.Image)
    if input_is_image:
        # RGB/RGBA images can be converted without copying
        # L (grayscale) images must be copied to avoid
        # https://github.com/cython/cython/issues/1605
        image = np.array(image, copy=image.mode == 'L')

    is_grayscale = len(image.shape) < 3
    if is_grayscale:
        image = image.reshape(image.shape + (1,))

    channels = image.shape[-1]

    input_image = image
    shape = np.array(image.shape)

    if downsample != 1:
        downsampled_shape = (shape[:2] // downsample) + (channels,)
        # converts to np.float32 with scale [0., 1.]
        resized = skimage.transform.resize(image, downsampled_shape, mode='edge')
        input_image = to_byte_format(resized)

    # apply 2D median filter on each channel separately

    kernel = skimage.morphology.square(kernel_size)

    def filter_channel(channel_index):
        """
        Filter an RGB image channel via median filter, ignore alpha channel.
        input/output data format: uint8
        """
        image_channel = input_image[:, :, channel_index]
        if channel_index < 3:
            return skimage.filters.median(image_channel, footprint=kernel)
        else:
            # do not filter alpha channel
            return image_channel

    background = np.dstack([filter_channel(i) for i in range(channels)])

    if downsample != 1:
        # upsample the computed background to original size
        # resize converts to np.float32 with scale [0., 1.]
        background_float = skimage.transform.resize(background, shape, mode='edge')
        background = to_byte_format(background_float)
    else:
        background_float = from_byte_format(background)

    # We assume the original images is a product of foreground and background,
    # thus we can recover the foreground by dividing the image by the background:
    # I = F * B => F = I / B
    # For division we use float32 format instead of uint8.
    # Inputs are scaled [0., 255.], output is scaled [0., 1.]
    image_float = from_byte_format(image)
    # prevent division by zero
    background_float = np.maximum(0.001, background_float)
    foreground = (image_float / background_float)
    # Values over 1.0 has to be clipped to prevent uint8 overflow.
    foreground = np.minimum(foreground, 1)
    foreground = to_byte_format(foreground)

    if is_grayscale:
        foreground = foreground[:, :, 0]
        background = background[:, :, 0]

    if input_is_image:
        foreground = PIL.Image.fromarray(foreground)
        background = PIL.Image.fromarray(background)

    return foreground, background


def to_grayscale(image):
    return to_byte_format(rgb2gray(image))


def to_rgb(image):
    if image.shape[-1] == 1:
        return np.broadcast_to(image, image.shape[:2] + (3,))
    else:
        return image


def to_byte_format(array):
    return (array * 255).astype(np.uint8)


def from_byte_format(array):
    return array.astype(np.float32) / 255

