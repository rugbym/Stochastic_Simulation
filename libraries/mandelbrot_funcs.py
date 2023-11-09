import numpy as np
import matplotlib.pyplot as plt

def mandelbrot(z_n, c):
    """
    returns next iteration of the input mandelbrot number.

    Args:
        z_n (_type_): _description_
        c (_type_): _description_

    Returns:
        _type_: _description_
    """
    return z_n**2 + c


def mandelbrot_set(c):
    z_n = 0
    for i in range(100):
        z_n = mandelbrot(z_n, c)
        if abs(z_n) > 2:
            return False
    return True


def mandelbrot_set_image(xmin, xmax, ymin, ymax, width, height):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    image = np.zeros((height, width))
    for i in range(width):
        for j in range(height):
            image[j, i] = mandelbrot_set(complex(x[i], y[j]))
    return image


def plot_mandelbrot_set(xmin, xmax, ymin, ymax, width, height):
    image = mandelbrot_set_image(xmin, xmax, ymin, ymax, width, height)
    plt.imshow(image)
    plt.show()