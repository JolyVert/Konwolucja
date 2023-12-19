import numpy as np
from skimage import io
from scipy.ndimage import convolve
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


def laplace_operator():
    image = io.imread("circle.jpg")
    laplace_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    filtered_image = np.dstack([
        convolve(image[:, :, channel], laplace_filter, mode="constant", cval=0.0)
        for channel in range(3)
    ])
    return filtered_image


def sobel_operator():
    image = io.imread("circle.jpg")
    sobel_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    filtered_image = np.dstack([
        convolve(image[:, :, channel], sobel_filter, mode="constant", cval=0.0)
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


def averaging_kernel():
    image = io.imread("panda.jpg")
    mean_filter = np.ones([9, 9]) / (9 ** 2)
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




#io.imsave("filtered_laplace.jpg", laplace_operator())
#io.imsave("filtered_gauss.jpg", gauss_kernel())
#io.imsave("filtered_sharpened.jpg", sharpening())
#io.imsave("filtered_average.jpg", averaging_kernel())
#io.imsave("filtered_sobel.jpg", sobel_operator())
io.imshow(sharpening(), cmap="Greys")
plt.show()
