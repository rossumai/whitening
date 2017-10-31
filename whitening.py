import numpy as np
from PIL import Image
from skimage.filters import median
from skimage.morphology import square

def read_image_to_numpy(path):
    return np.asarray(Image.open(path), dtype="uint8")

def whiten(image, kernel_size):
    """
    Tries to separate text/line foreground and background by 2D median filter
    whitening.

    The idea is that foreground (text/lines) are spikes that can be removed by
    spatial median filter, thus leaving the background. We can then normalize
    the original image by the background, leaving the foreground.

    Input:
    image - input image

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

if __name__ == '__main__':
    import os
    data_dir = 'data/source_images/'
    image = read_image_to_numpy(os.path.join(data_dir, 'IMG_3262_denoised.jpg'))
    whitened, background = whiten(image, kernel_size=50)
    Image.fromarray(whitened).save('IMG_3262_whitened.jpg', 'jpeg')
    Image.fromarray(background).save('IMG_3262_background.jpg', 'jpeg')
