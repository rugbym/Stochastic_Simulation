import numpy as np
import matplotlib.pyplot as plt

def mandelbrot(z_n, c):
    """
    returns next iteration of the input mandelbrot number.

    Args:
        z_n (_type_): previous mandelbrot number
        c (_type_): number to test

    Returns:
        _type_: next iteration of the mandelbrot number
    """
    return z_n**2 + c


def mandelbrot_set(c, i_iterations:int=100):

    """
    Assess whether a number is part of the mandelbrot set or not

    Args:
        c: number to test
        i_iterations (int): number of iterations, i, to check the number for
    Returns:
        bool: whether the number belongs to the mandelbrotset
    """
    z_n = 0
    for i in range(100):
        z_n = mandelbrot(z_n, c)
        # check if the number is diverging
        if abs(z_n) > 2:
            return False
    return True


def mandelbrot_set_image(xmin, xmax, ymin, ymax, width, height, i_iterations:int=100):
    """
    Creates an image of the mandelbrot set.

    Args:
        xmin (_type_): minimum x value
        xmax (_type_): maximum x value
        ymin (_type_): minimum y value
        ymax (_type_): maximum y value
        width (_type_): width of the image
        height (_type_): height of the image
        i_iterations (int): number of iterations, i, to check the number for

    Returns:
        _type_: _description_
    """
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    image = np.zeros((height, width))
    for i in range(width):
        for j in range(height):
            image[j, i] = mandelbrot_set(complex(x[i], y[j]))
    return image


def plot_mandelbrot_set(xmin, xmax, ymin, ymax, width, height, i_iterations:int=100):
    """
    Plots an image of the mandelbrot set.

    Args:
        xmin (_type_): minimum x value
        xmax (_type_): maximum x value
        ymin (_type_): minimum y value
        ymax (_type_): maximum y value
        width (_type_): width of the image
        height (_type_): height of the image
        i_iterations (int, optional): _description_. Defaults to 100.
    """
    image = mandelbrot_set_image(xmin, xmax, ymin, ymax, width, height)
    plt.imshow(image)
    plt.show()