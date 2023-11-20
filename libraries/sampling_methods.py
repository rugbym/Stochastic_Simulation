"""
Assess whether a number is part of the mandelbrot set or not.

Funcs:
    pure_random_sampling(s_samples): creates random samples in the (0,1) interval
    hypercube_sampling(s_samples): creates latin hypercube samples in the (0,1) interval
    class MandelbrotSetMC: class to compute the area of the mandelbrot set using the Monte Carlo method.
        __init__(n_h_bins=8000, n_v_bins=8000): constructor of the class
        pure_random_sampling_(s_samples): creates random samples in the mandelbrot range
        hypercube_sampling_(s_samples): creates latin hypercube samples in the mandelbrot range
        orthogonal_sampling(s_samples): creates orthogonal samples in the mandelbrot range
        area_mandelbrot(s_samples, i_iterations, sampling='PR'): computes the area of the mandelbrot set
        
"""
from .mandelbrot_funcs import compute_mandelbrot_array
import numpy as np

def pure_random_sampling(s_samples):
    """
    Function that creates 2D random samples in the (0,1) interval

    Args:
        s_samples (int): number of samples to generate

    Returns:
        tuple: tuple of arrays (samples_x,samples_y) that depict the x and y coordinates of the samples
    """
    x_samples = np.random.rand(s_samples) 
    y_samples = np.random.rand(s_samples) 
    return x_samples, y_samples

def hypercube_sampling(s_samples):
    """
    Function that creates 2D latin hypercube samples in the (0,1) interval

    Args:
        s_samples (int): number of samples to generate


    Returns:
        tuple: tuple of arrays (samples_x,samples_y) that depict the x and y coordinates of the samples
    """
    intervals =np.linspace(0, 1,s_samples + 1)[:-1]
    np.random.shuffle(intervals)
    samples_x = intervals.copy()
    np.random.shuffle(intervals)
    samples_y = intervals
    return samples_x, samples_y
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
        self.sampling_methods = {'complete': self.complete_range, 'PR': self.pure_random_sampling_,'LHS': self.hypercube_sampling_, 'ORT': self.orthogonal_sampling}
    
    def pure_random_sampling_(self, s_samples):
        """
        Function that creates random samples in the mandelbrot range

        Args:
            s_samples (int): number of samples to generate

        Returns:
            tuple: tuple of arrays (samples_x,samples_y) that depict the x and y coordinates of the samples
        """
        x_samples, y_samples = pure_random_sampling(s_samples)
        x_samples = x_samples * (self.xrange[1] - self.xrange[0]) + self.xrange[0]
        y_samples = y_samples * (self.yrange[1] - self.yrange[0]) + self.yrange[0]
        return x_samples, y_samples
    
    def hypercube_sampling_(self, s_samples):
        """
        Function that creates latin hypercube samples in the mandelbrot range

        Args:
            n_samples (int): number of samples to generate

        Returns:
                tuple: tuple of arrays (samples_x,samples_y) that depict the x and y coordinates of the samples
        """

        samples_x,samples_y = hypercube_sampling(s_samples)
        samples_x = samples_x * (self.xrange[1] - self.xrange[0]) + self.xrange[0] 
        samples_y = samples_y * (self.yrange[1] - self.yrange[0]) + self.yrange[0]
        return samples_x, samples_y
    
    
    def orthogonal_sampling(self, s_samples):
        """
        Function that orthogonal samples in the mandelbrot range

        Args:
            n_samples (int): number of samples to generate

        Returns:
            tuple: tuple of arrays (samples_x,samples_y) that depict the x and y coordinates of the samples
        """
        grid_x = np.linspace(self.xrange[0], self.xrange[1], self.n_h_bins)
        grid_y = np.linspace(self.yrange[0], self.yrange[1], self.n_v_bins)

        indices_x = np.random.choice(self.n_h_bins, s_samples,replace=False)
        indices_y = np.random.choice(self.n_v_bins, s_samples,replace=False)

        samples_x = grid_x[indices_x]
        samples_y = grid_y[indices_y]

        return samples_x, samples_y
    
    def area_mandelbrot(self, s_samples, i_iterations, sampling='PR'):
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
        x_samples,y_samples = self.sampling_methods[sampling](s_samples)

        # create the complex numbers
        c = x_samples + 1j * y_samples
        # compute the complex numbers belong to the mandelbrot set
        mandel = compute_mandelbrot_array(c, self.i_iterations)
        # compute the area of the mandelbrot set
        self.area = (xmax - xmin) * (ymax - ymin) * np.sum(mandel) / self.s_samples
        return self.area

    def complete_range(self, *args):
        return np.arange(self.n_v_bins * self.n_h_bins)


