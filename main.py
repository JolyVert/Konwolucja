import numpy as np
from skimage import io
from scipy.ndimage import convolve
from skimage.color import rgb2gray


def laplace_operator():
    image = io.imread("circle.jpg")
    laplace_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    filtered_image = np.dstack([
        convolve(image[:, :, channel], laplace_filter, mode="constant", cval=0.0)
        for channel in range(3)
    ])
    return filtered_image


def gauss_kernel():
    image = io.imread("panda.jpg")
    mean_filter = np.array([[1, 2, 1], [1, 4, 1], [1, 2, 1]]) / 16
    filtered_image = np.dstack([
        convolve(image[:, :, channel], mean_filter, mode="constant", cval=0.0)
        for channel in range(3)
    ])
    return filtered_image


def sharpening():
    image = io.imread("panda.jpg")
    mean_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    mean_filter = mean_filter / mean_filter.sum()
    image = rgb2gray(image)
    filtered_image = convolve(image, mean_filter, mode="constant", cval=0.0)
    filtered_image = filtered_image - filtered_image.min()
    filtered_image = filtered_image / filtered_image.max()
    return filtered_image


# io.imsave("filtered_laplace.jpg", laplace_operator())
# io.imsave("filtered_gauss.jpg", gauss_kernel())
# io.imsave("filtered_sharpened.jpg", sharpening())
# io.imshow(sharpening())
io.imshow(sharpening(),)
