import argparse
import time

import PIL.Image
import numpy as np

from whitening import whiten, to_grayscale, to_rgb


def main():
    args = parse_args()
    image = np.asarray(PIL.Image.open(args.image), dtype='uint8')
    if args.grayscale:
        image = to_grayscale(image)
    foreground, background = timed_whiten(image, kernel_size=args.kernel_size,
                                          downsample=args.downsample)
    if args.grayscale:
        foreground = to_rgb(foreground)
        background = to_rgb(background)
    PIL.Image.fromarray(foreground).save(args.foreground)
    if args.background is not None:
        PIL.Image.fromarray(background).save(args.background)


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
    main()
