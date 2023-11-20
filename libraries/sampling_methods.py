"""
Assess whether a number is part of the mandelbrot set or not.

Funcs:

    area_mandelbrot(i_iterations, s_samples, sampling='random', n_h_bins=8000, n_v_bins=8000)
    complete_range(*args)
"""
from .mandelbrot_funcs import compute_mandelbrot_array, mandelbrot_set_array
import numpy as np


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
        x_samples = np.random.rand(s_samples)
        y_samples = np.random.rand(s_samples)
        x_samples = x_samples * (self.xrange[1] - self.xrange[0]) + self.xrange[0]
        y_samples = y_samples * (self.yrange[1] - self.yrange[0]) + self.yrange[0]
        return x_samples, y_samples

    def hypercube_sampling_(self, s_samples):
        """
        Function that creates latin hypercube samples in the mandelbrot range

        Args:
            s_samples (int): number of samples to generate

        Returns:
                tuple: tuple of arrays (samples_x,samples_y) that depict the x and y coordinates of the samples
        """

        intervals = np.linspace(0, 1,s_samples + 1)[:-1]
        np.random.shuffle(intervals)
        samples_x = intervals.copy()
        np.random.shuffle(intervals)
        samples_y = intervals

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
        intervals = np.linspace(0, 1, s_samples + 1)[:-1]
        samples_x = intervals + np.random.rand(s_samples) / s_samples
        np.random.shuffle(samples_x)
        samples_y = intervals + np.random.rand(s_samples) / s_samples
        np.random.shuffle(samples_y)

        samples_x = samples_x * (self.xrange[1] - self.xrange[0]) + self.xrange[0]
        samples_y = samples_y * (self.yrange[1] - self.yrange[0]) + self.yrange[0]

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




class MandelbrotSetMCBias:
    """
    Class to compute the area of the mandelbrot set using the Monte Carlo method.

    Args:
        n_h_bins (int): number of horizontal bins
        n_v_bins (int): number of vertical bins
    Funcs:
        area_mandelbrot(sampling='random'): computes the area of the mandelbrot set
        complete_range(): returns a range of indexes for the complete range of the grid
    """

    def __init__(self, n_h_bins=800, n_v_bins=800):
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

        complexity = complexityMandelbrot(c, self.i_iterations)

        # normalize the complexity values to [0, 1]
        norm_complexity = complexity / self.i_iterations

        # adjust the sampling probability based on complexity
        sample_probabilities = norm_complexity /np.sum(norm_complexity)

        indices = np.random.choice(np.arange(self.s_samples), size=self.s_samples, replace=True,
                                          p=sample_probabilities)

        # use the chosen indices to select corresponding samples
        x_samples = x_samples[indices]
        y_samples = y_samples[indices]

        c = x_samples + 1j * y_samples
        # compute the complex numbers belong to the mandelbrot set
        mandel = compute_mandelbrot_array(c, self.i_iterations)
        # compute the area of the mandelbrot set
        self.area = (xmax - xmin) * (ymax - ymin) * np.sum(mandel) / self.s_samples
        return self.area

    def complete_range(self, *args):
        return np.arange(self.n_v_bins * self.n_h_bins)


def complexityMandelbrot(c, max_iter):

    z= np.zeros_like(c, dtype=np.complex128)
    complexity= np.zeros_like(c, dtype=np.int32)

    for i in range(max_iter):
        z = z**2 +c
        m = np.abs(z) < 1000
        complexity +=m

        if not np.any(m):
            break

    return complexity
