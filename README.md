# Document whitening (foreground separation)

This package tries to separate text/line foreground and background by 2D median
filter.

## Installation

Install from PyPI. Works on Python 3.

```bash
pip install whitening
```

## Example usage

Python API:

```python
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

CLI:

```bash
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

We assume the original images is a product of foreground and background,
thus we can recover the foreground by dividing the image by the background:
`I = F * B => F = I / B`. We try to approximate the background by 2D median
filtering the original image which suppresses sparse features such as text and
lines.

Select kernel size that's enough for not making artifacts while small enough
to keep computation fast. A good starting point is 50 pixels.

A 9.5 Mpx image can be processed on a MacBook in 15 s, with grayscale and
downsampling 4x the run time can be reduced to 1 s! Quite good results can be
obtained even with kernel size 10 and downsampling 16x.

More info: http://bohumirzamecnik.cz/blog/2015/image-whitening/

## License

Author: Bohumir Zamecnik

Supported by [Rossum](https://rossum.ai), creating a world without manual data entry.
