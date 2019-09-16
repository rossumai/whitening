# - it should support image as numpy array or as a PIL.Image
# - the output type should be the same as input type (np->np, PIL->PIL)
# - it should support input formats:
#   - RGB (h, w, 3)
#   - RGBA (h, w, 4) - do not whiten the alpha channel
#   - greyscale (h, w)
# - if greyscale preprocessing is enabled, output would be always grayscale

import numpy as np
import PIL.Image

from whitening import whiten

image = PIL.Image.open('data/source_images/IMG_3262_denoised.jpg')
assert image.mode == 'RGB'
assert image.size == (800, 800)


def test_whiten_pil_rgb():
    fg, bg = whiten(image, kernel_size=10)
    for output in (fg, bg):
        assert isinstance(output, PIL.Image.Image)
        assert output.mode == image.mode
        assert output.size == image.size


def test_whiten_pil_rgba():
    image_rgba = image.convert('RGBA')
    fg, bg = whiten(image_rgba, kernel_size=10)
    for output in (fg, bg):
        assert isinstance(output, PIL.Image.Image)
        assert output.mode == 'RGBA'
        assert output.size == image.size


def test_whiten_pil_grayscale():
    image_gs = image.convert('L')
    fg, bg = whiten(image_gs, kernel_size=10)
    for output in (fg, bg):
        assert isinstance(output, PIL.Image.Image)
        assert output.mode == 'L'
        assert output.size == image.size


def test_whiten_numpy_rgb():
    image_np = np.array(image)
    fg, bg = whiten(image_np, kernel_size=10)
    for output in (fg, bg):
        assert isinstance(output, np.ndarray)
        assert output.shape == image.size + (3,)


def test_whiten_numpy_rgba():
    image_np = np.array(image.convert('RGBA'))
    fg, bg = whiten(image_np, kernel_size=10)
    for output in (fg, bg):
        assert isinstance(output, np.ndarray)
        assert output.shape == image.size + (4,)


def test_whiten_numpy_gs():
    image_np = np.array(image.convert('L'))
    fg, bg = whiten(image_np, kernel_size=10)
    for output in (fg, bg):
        assert isinstance(output, np.ndarray)
        assert output.shape == image.size
