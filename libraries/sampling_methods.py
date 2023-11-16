"""
Assess whether a number is part of the mandelbrot set or not.

Funcs:

    area_mandelbrot(i_iterations, s_samples, sampling='random', n_h_bins=8000, n_v_bins=8000)
    complete_range(*args)
"""
from .mandelbrot_funcs import compute_mandelbrot_array
import numpy as np


class MandelbrotSetMC:
    """
    Class to compute the area of the mandelbrot set using the Monte Carlo method.

    Args:
        n_h_bins (int): number of horizontal bins
        n_v_bins (int): number of vertical bins
    Funcs:
        area_mandelbrot(sampling='random'): computes the area of the mandelbrot set
        complete_range(): returns a range of indexes for the complete range of the grid
    """

    def __init__(self, n_h_bins=8000, n_v_bins=8000):
        self.n_h_bins = n_h_bins
        self.n_v_bins = n_v_bins
        self.xrange = (-2, 0.47)

        self.yrange = (-1.12, 1.12)
        # create a dictionary with the sampling methods
        self.sampling_methods = {'complete': self.complete_range, 'random': np.random.rand}

    def area_mandelbrot(self, s_samples, i_iterations, sampling='random'):
        """
        Computes the area of the mandelbrot set using the Monte Carlo method.

        Args:
            s_samples (int): number of samples to take
            i_iterations (int): number of iterations, i, to check the number for
            sampling (str): 'random' or 'complete'


        Returns:
            float: area of the mandelbrot set
        """
        # define the range of the grid
        self.s_samples = s_samples
        self.i_iterations = i_iterations
        xmin, xmax = self.xrange
        ymin, ymax = self.yrange
        # choose the samples randomly using the sampling method and rescale
        # them to the range of the grid
        x_samples = self.sampling_methods[sampling](self.s_samples) * (min(self.xrange) - max(self.xrange)) + max(
            self.xrange)
        y_samples = self.sampling_methods[sampling](self.s_samples) * (min(self.yrange) - max(self.yrange)) + max(
            self.yrange)

        # create the complex numbers
        c = x_samples + 1j * y_samples
        # compute the complex numbers belong to the mandelbrot set
        mandel = compute_mandelbrot_array(c, self.i_iterations)
        # compute the area of the mandelbrot set
        self.area = (xmax - xmin) * (ymax - ymin) * np.sum(mandel) / self.s_samples
        return self.area

    def complete_range(self, *args):
        return np.arange(self.n_v_bins * self.n_h_bins)

class MandelbrotSetLHS:
    """
        Class to compute the area of the mandelbrot set using the Latin hypercube sampling method.

        Args:
            n_h_bins (int): number of horizontal bins
            n_v_bins (int): number of vertical bins
        Funcs:
            hypercube(): creates latin hypercube samples in the mandelbrot set
            area_mandelbrot(): computes the area of the mandelbrot set using LHS

        """
    def __init__(self, n_h_bins=8000, n_v_bins=8000):
        self.n_h_bins = n_h_bins
        self.n_v_bins = n_v_bins
        self.xrange = (-2, 0.47)

        self.yrange = (-1.12, 1.12)

    def hypercube(self, n_samples):
        """
                Function that creates latin hypercube samples in the mandelbrot range

                Args:
                    n_samples (int): number of samples to generate

                Returns:
                     tuple: tuple of arrays (samples_x,samples_y) that depict the x and y coordinates of the samples
                """

        intervals =np.linspace(0, 1,n_samples + 1)[:-1]
        np.random.shuffle(intervals)
        samples_x = intervals * (self.xrange[1] -self.xrange[0]) + self.xrange[0]
        np.random.shuffle(intervals )
        samples_y = intervals * (self.yrange[1] -self.yrange[0]) + self.yrange[0]
        return samples_x, samples_y

    def area_mandelbrot(self, s_samples, i_iterations):
        """
               Computes the area of the mandelbrot set using the LHS method.

               Args:
                   s_samples (int): number of samples to take
                   i_iterations (int): number of iterations, i, to check the number for

               Returns:
                   float: area of the mandelbrot set
               """
        x_samples, y_samples = self.hypercube(s_samples)
        c = x_samples + 1j *y_samples
        mandel= compute_mandelbrot_array(c, i_iterations)
        area = (self.xrange[1] - self.xrange[0])* (self.yrange[1] - self.yrange[0]) * np.sum(mandel)/ s_samples
        return area



class MandelbrotSetOrt:
    """
            Class to compute the area of the mandelbrot set using an orthogonal sampling method.

            Args:
                n_h_bins (int): number of horizontal bins
                n_v_bins (int): number of vertical bins
            Funcs:
                orthogonal(): creates orthogonal samples in the mandelbrot set
                area_mandelbrot(): computes the area of the mandelbrot set using orthogonal sampling

            """

    def __init__(self, n_h_bins=8000, n_v_bins=8000):
        self.n_h_bins = n_h_bins
        self.n_v_bins = n_v_bins
        self.xrange = (-2, 0.47)
        self.yrange = (-1.12, 1.12)

    def orthogonal(self, n_samples):
        """
                Function that orthogonal samples in the mandelbrot range

                Args:
                    n_samples (int): number of samples to generate

                Returns:
                    tuple: tuple of arrays (samples_x,samples_y) that depict the x and y coordinates of the samples
                """
        grid_x = np.linspace(self.xrange[0], self.xrange[1], self.n_h_bins)
        grid_y = np.linspace(self.yrange[0], self.yrange[1], self.n_v_bins)

        indices_x = np.random.choice(self.n_h_bins, n_samples,replace=False)
        indices_y = np.random.choice(self.n_v_bins, n_samples,replace=False)

        samples_x = grid_x[indices_x]
        samples_y = grid_y[indices_y]

        return samples_x, samples_y

    def area_mandelbrot(self, s_samples, i_iterations):
        """
                Computes the area of the mandelbrot set using the orthogonal sampling method.

                Args:
                    s_samples (int): number of samples to take
                    i_iterations (int): number of iterations, i, to check the number for

                Returns:
                    float: area of the mandelbrot set
                """
        x_samples, y_samples = self.orthogonal(s_samples)
        c = x_samples + 1j * y_samples
        mandel = compute_mandelbrot_array(c, i_iterations)
        area = (self.xrange[1] - self.xrange[0]) * (self.yrange[1] - self.yrange[0]) * np.sum(mandel) / s_samples
        return area
