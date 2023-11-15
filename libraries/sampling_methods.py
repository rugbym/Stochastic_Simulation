"""
Assess whether a number is part of the mandelbrot set or not. 

Funcs:
    
    area_mandelbrot(i_iterations, s_samples, sampling='random', n_h_bins=8000, n_v_bins=8000)
    complete_range(*args)
"""
from .mandelbrot_funcs import compute_mandelbrot_array
import numpy as np

class MandelbrotSet:
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
        x_samples = self.sampling_methods[sampling](self.s_samples) * (min(self.xrange) - max(self.xrange)) + max(self.xrange)
        y_samples = self.sampling_methods[sampling](self.s_samples) * (min(self.yrange) - max(self.yrange)) + max(self.yrange)
        
        # create the complex numbers  
        c = x_samples + 1j*y_samples
        # compute the complex numbers belong to the mandelbrot set
        mandel = compute_mandelbrot_array(c, self.i_iterations)
        # compute the area of the mandelbrot set
        self.area = (xmax - xmin) * (ymax - ymin) * np.sum(mandel) / self.s_samples 
        return self.area

    def complete_range(self, *args):
        return np.arange(self.n_v_bins * self.n_h_bins)