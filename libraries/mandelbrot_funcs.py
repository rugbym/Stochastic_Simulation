"""
Description:
    This file contains functions to compute the mandelbrot set. 
    
Funcs:
    mandelbrot(z_n, c): returns next iteration of the input mandelbrot number.
    mandelbrot_set(c, i_iterations:int=100):  Assess whether a number, c, is part of the mandelbrot set or not
    mandelbrot_set_array(c, i_iterations:int=100): Assess whether a number, c, is part of the mandelbrot set or not. Computes the mandelbrot set for an array of numbers all at once. When not using jit, this is faster than using mandelbrot_set for each number individually.
    compute_mandelbrot_array(c, i_iterations:int=100): Computes the mandelbrot set for an array of numbers.
    mandelbrot_set_image(xmin, xmax, ymin, ymax, width, height, i_iterations:int=100):  Creates an image of the mandelbrot set.
    plot_mandelbrot_set(xmin, xmax, ymin, ymax, width, height, i_iterations:int=100): Plots an image of the mandelbrot set.

"""
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

@jit(nopython=True)
def mandelbrot(z_n, c):
    """
    returns next iteration of the input mandelbrot number.

    Args:
        z_n (_type_): previous mandelbrot number
        c (_type_): number to test

    Returns:
        _type_: next iteration of the mandelbrot number
    """
    return np.square(z_n)+ c

@jit(nopython=True)
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
    for _ in range(i_iterations):
        z_n = mandelbrot(z_n, c)
        # check if the number is diverging
        if abs(z_n) > 2:
            return False
    return True


def mandelbrot_set_array(c, i_iterations:int=100):
    """
    Assess whether a number is part of the mandelbrot set or not. Computes the mandelbrot set for an array of numbers all at once. When not using jit, this is faster than using mandelbrot_set for each number individually.

    Note: If using simply complex numbers as input, please use mandelbrot_set instead.

    Args:
        c: number to test, complex number, preferably an array.
        i_iterations (int): number of iterations, i, to check the number for
    Returns:
        bool: whether the number belongs to the mandelbrotset
    """
    
    z_n = np.zeros(c.shape, dtype=np.complex128)
    for _ in np.arange(i_iterations):
        z_n = mandelbrot(z_n, c)
    results = np.abs(z_n) < 2
    return results

@jit(nopython=True)
def compute_mandelbrot_array(c, i_iterations:int=100):
    """
    Computes the mandelbrot set for an array of numbers.

    Note: If using simply complex numbers as input, please use mandelbrot_set instead.

    Args:
        c: number to test, complex number, preferably an array.
        i_iterations (int): number of iterations, i, to check the number for
    Returns:
        bool: whether the number belongs to the mandelbrotset
    """
    
    results = np.zeros(c.shape).flatten()
    for i, sub_c in enumerate(c.flatten()):
        results[i] = mandelbrot_set(sub_c, i_iterations)
    return results

@jit(nopython=True)
def mandelbrot_set_image(xmin, xmax, ymin, ymax, width, height, i_iterations:int=100):
    """
    Creates an image of the mandelbrot set.

    Args:
        xmin (float): minimum x value
        xmax (float): maximum x value
        ymin (float): minimum y value
        ymax (float): maximum y value
        width (int): width of the image
        height (int): height of the image
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
    
    plt.figure(figsize=(5,5),dpi=600)
    plt.title('Mandelbrot Set')
    plt.imshow(image,extent=[xmin, xmax, ymin, ymax])
    
    num_ticks = 5
    plt.xticks(np.linspace(xmin, xmax, num_ticks))
    plt.yticks(np.linspace(ymin, ymax, num_ticks))
    
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.show()