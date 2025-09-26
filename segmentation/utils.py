
"""
This module provides utility classes for handling circular arrays and computing moving moments and covariance ratios.

Classes:
    CircularArray: A class for managing circular arrays for multivariate data.
    CircularArrayUnivariate: A subclass of CircularArray tailored for univariate data.
    MovingMoments: A class for computing and updating the mean and covariance of a dataset.
    MovingCovarianceRatio: A class for computing the divergence of a dataset based on moving covariance.

CircularArray:
    Manages a circular array structure for multivariate data, supporting operations such as adding new samples,
    and retrieving the first, last, and middle samples.

CircularArrayUnivariate:
    Extends CircularArray for univariate data, providing methods to add new samples and check if the middle sample
    is a maximum with respect to the entire array.

MovingMoments:
    Computes and updates the mean and covariance of a dataset. Supports updating the dataset with new samples
    and removing old ones.

MovingCovarianceRatio:
    Utilizes MovingMoments to compute the divergence of a dataset based on the covariance of samples before and
    after a midpoint in a sliding window approach.

"""

from typing import Dict, Tuple, List, Protocol
import numpy as np
from scipy.fftpack import fft, ifft, fftshift, ifftshift
import pywt



class CircularArray:
    """
    A class representing a circular array.

    Attributes:
        samples (np.array): The array of samples.
        start (int): The starting index of the circular array.

    Methods:
        dimension() -> int: Returns the dimension of the samples.
        n_samples() -> int: Returns the number of samples.
        middle_index() -> int: Returns the index of the middle sample.
        get_first() -> np.array: Returns the first sample.
        get_last() -> np.array: Returns the last sample.
        get_middle() -> np.array: Returns the middle sample.
        add(sample: np.array): Adds a new sample to the circular array.
        get_samples() -> np.array: Returns all the samples in the circular array.

    The CircularArray class is designed to manage a circular array of samples. It supports operations such as adding new samples, and retrieving the first, last, and middle samples. The class automatically handles the circular nature of the array, ensuring that indices wrap around as necessary.
    """

    def __new__(cls, samples: np.array):
        """
        Create a new instance of CircularArray.

        Args:
            samples (np.array): The array of samples.

        Returns:
            object: Returns an instance of CircularArray if the input samples are 
                multidimensional, otherwise returns an instance of CircularArrayUnivariate.
        """

        samples = np.squeeze(samples)
        if len(samples.shape) > 1:
            return object.__new__(cls)
        else:
            return object.__new__(CircularArrayUnivariate)


    def __init__(self, samples: np.array):
        """
        Initializes a CircularArray object.

        Args:
            samples (np.array): The array of samples.
        """
        self.samples = samples
        self.start = 0

    @property
    def dimension(self) -> int:
        """
        Returns the dimension of the samples.

        Returns:
            int: The dimension of the samples.
        """
        return self.samples.shape[0]

    @property
    def n_samples(self) -> int:
        """
        Returns the number of samples.

        Returns:
            int: The number of samples.
        """
        return self.samples.shape[1]
    
    @property
    def middle_index(self) -> int:
        """
        Returns the index of the middle sample.

        Returns:
            int: The index of the middle sample.
        """
        return (self.start + self.n_samples//2) % self.n_samples  

    def get_first(self) -> np.array:
        """
        Returns the first sample.

        Returns:
            np.array: The first sample.
        """
        return self.samples[:, self.start, np.newaxis]
    
    def get_last(self) -> np.array:
        """
        Returns the last sample.

        Returns:
            np.array: The last sample.
        """
        end = (self.start - 1) % self.n_samples
        return self.samples[:, end, np.newaxis]

    def get_middle(self) -> np.array:
        """
        Returns the middle sample.

        Returns:
            np.array: The middle sample.
        """
        return self.samples[:, self.middle_index, np.newaxis]

    def add(self, sample: np.array):
        """
        Adds a new sample to the circular array.

        Args:
            sample (np.array): The new sample to be added.
        """
        self.samples[:, self.start] = sample.squeeze()
        self.start = (self.start + 1) % self.n_samples
    
    def get_samples(self) -> np.array:
        """
        Returns all the samples in the circular array.

        Returns:
            np.array: The array of samples.
        """
        samples = np.hstack((self.samples[:, self.start:], self.samples[:, :self.start]))
        return samples
 

class CircularArrayUnivariate(CircularArray):
    """
    A class representing a univariate circular array, derived from CircularArray.

    This class is intended for use when the input samples are unidimensional. It inherits from CircularArray and overrides methods as necessary to accommodate the unidimensional nature of its samples.

    Attributes and methods are inherited from CircularArray, with modifications as needed for unidimensional data handling.

    Methods:
        is_middle_maximum() -> bool: Check if the middle element of the circular array is the maximum.
    """

    def __init__(self, samples: np.array):
        """
        Initialize the CircularArrayUnivariate object.

        Args:
            samples (np.array): The univariate data samples.

        """
        samples = np.reshape(samples, (1, len(samples.squeeze())))
        super().__init__(samples)
    
    def get_first(self) -> float:
        """
        Get the first element of the circular array.

        Returns:
            float: The first element of the circular array.

        """
        return super().get_first().squeeze()
    
    def get_last(self) -> float:
        """
        Get the last element of the circular array.

        Returns:
            float: The last element of the circular array.

        """
        return super().get_last().squeeze()
    
    def add(self, sample: float):
        """
        Add a new sample to the circular array.

        Args:
            sample (float): The new sample to be added.

        """
        super().add(np.array(sample).reshape((1, 1)))
    
    def is_middle_maximum(self) -> bool:
        """
        Check if the middle element of the circular array is the maximum.

        Returns:
            bool: True if the middle element is the maximum and unique, False otherwise.

        """
        middle = self.get_middle()
        is_maximum = np.all(middle >= self.samples)
        is_unique = np.sum(middle == self.samples) == 1
        return is_maximum and is_unique


class MovingMoments:
    """
    Class for computing moving moments of a set of samples.

    Attributes:
        _mean (np.array): The mean of the samples.
        _covariance (np.array): The covariance matrix of the samples.
        n_samples (int): The number of samples.

    Methods:
        __init__(samples: np.array): Initializes the MovingMoments object with the given samples.
        dimension() -> int: Returns the dimension of the samples.
        update(old_sample: np.array, new_sample: np.array): Updates the moving moments with a new sample.
        mean() -> np.array: Returns the mean of the samples.
        covariance() -> np.array: Returns the covariance matrix of the samples.
    """

    def __init__(self, samples: np.array):
        """
        Initializes the MovingMoments object with the given samples.

        Args:
            samples (np.array): The samples to compute the moving moments on.
        """
        self._mean = np.mean(samples, axis=1, keepdims=True)
        self._covariance = np.cov(samples)
        self.n_samples = samples.shape[1]
    
    @property
    def dimension(self) -> int:
        """
        Returns the dimension of the samples.

        Returns:
            int: The dimension of the samples.
        """
        return len(self._mean)

    def update(self, old_sample: np.array, new_sample: np.array):
        """
        Updates the moving moments with a new sample.

        Args:
            old_sample (np.array): The old sample to be replaced.
            new_sample (np.array): The new sample to be added.
        """
        old_sample = np.reshape(old_sample, (self.dimension, 1))
        new_sample = np.reshape(new_sample, (self.dimension, 1))

        mean_old = self._mean
        covariance_old = self._covariance

        self._mean = mean_old + (new_sample - old_sample) / self.n_samples
        self._covariance = covariance_old + 1 / (self.n_samples -1) * \
            (new_sample @ new_sample.T - old_sample @ old_sample.T - \
            mean_old @ (new_sample.T - old_sample.T) - \
            (new_sample - old_sample) @ self._mean.T)
    
    @property
    def mean(self) -> np.array:
        """
        Returns the mean of the samples.

        Returns:
            np.array: The mean of the samples.
        """
        return self._mean
    
    @property
    def covariance(self) -> np.array:
        """
        Returns the covariance matrix of the samples.

        Returns:
            np.array: The covariance matrix of the samples.
        """
        return self._covariance


class MovingCovarianceRatio:
    """
    Calculates the moving covariance ratio for a given set of samples.

    Attributes:
        samples (np.array): The array of samples used for calculating the moving covariance ratio.
        moments_total (MovingMoments): The moving moments object for the total set of samples.
        moments_before (MovingMoments): The moving moments object for the samples before the middle sample.
        moments_after (MovingMoments): The moving moments object for the samples after the middle sample.

    Methods:
        __init__(samples: np.array): Initializes the MovingCovarianceRatio object with the given samples.
        update(new_sample: np.array): Updates the moving covariance ratio with a new sample.
        get_divergence() -> float: Calculates the divergence of the moving covariance ratio.
    """

    def __init__(self, samples: np.array):
        self.samples = CircularArray(samples)

        if len(samples.shape) == 1:
            samples = samples[np.newaxis, :]
 
        self.moments_total = MovingMoments(samples)
        self.moments_before = MovingMoments(samples[:, :samples.shape[1]//2])
        self.moments_after = MovingMoments(samples[:, samples.shape[1]//2:])

    def update(self, new_sample: np.array):
        """
        Updates the moving covariance ratio with a new sample.

        Args:
            new_sample (np.array): The new sample to be added to the moving covariance ratio calculation.
        """
        middle_sample = self.samples.get_middle()
        old_sample = self.samples.get_first()

        self.moments_total.update(old_sample=old_sample, new_sample=new_sample)
        self.moments_before.update(old_sample=old_sample, new_sample=middle_sample)
        self.moments_after.update(old_sample=middle_sample, new_sample=new_sample)

        self.samples.add(new_sample)

    def get_divergence(self) -> float:
        """
        Calculates the divergence of the moving covariance ratio.

        Returns:
            float: The divergence of the moving covariance ratio.
        """
        I = 1e-6 * np.eye(self.samples.dimension)

        y_total = np.linalg.slogdet(self.moments_total.covariance + I)[1]
        y_before = np.linalg.slogdet(self.moments_before.covariance + I)[1]
        y_after = np.linalg.slogdet(self.moments_after.covariance + I)[1]

        mean_before = self.moments_before.mean
        mean_after = self.moments_after.mean

        #return y_total - 0.5 * (y_before + y_after) + 0.5 * np.log(np.linalg.norm(mean_after - mean_before))
        return y_total - 0.5 * (y_before + y_after)


def MVMD(f, alpha, tau, K, DC, init, tol):
    """
    Multivariate Variational Mode Decomposition (MVMD) algorithm.
    This function decomposes a multivariate signal into K modes using the MVMD algorithm. One central frequency for each channel in each subband (difference with respect to online implementations, of course slower)
    Input and Parameters:
    ---------------------
    f       - the time domain signal (1D) to be decomposed
    alpha   - the balancing parameter of the data-fidelity constraint
    tau     - time-step of the dual ascent ( pick 0 for noise-slack )
    K       - the number of modes to be recovered
    DC      - true if the first mode is put and kept at DC (0-freq)
    init    - 0 = all omegas start at 0
                       1 = all omegas start uniformly distributed
                      2 = all omegas initialized randomly
    tol     - tolerance of convergence criterion; typically around 1e-6

    Output:
    -------
    u       - the collection of decomposed modes
    u_hat   - spectra of the modes
    omega   - estimated mode center-frequencies
    """
    x,y = f.shape
    if x > y:
        C = y #number of channels
        T = x #number of samples
        f = f.T
    else:
        C = x
        T = y


    # sampling frequency
    fs = 1./T
    # BUG: problem with odd indexes
    ltemp = T//2
    fMirr = np.zeros([C, 2*T])

    # Period and sampling frequency of input signal
    fMirr[:, :T//2] = np.flip(f[:,:ltemp],axis = 0)
    fMirr[:, T//2:T//2+T] = f[:,:]
    fMirr[:, T//2+T:] = np.flip(f[:,ltemp:],axis = 0) 
        
    # Time Domain 0 to T (of mirrored signal)
    T = len(fMirr[0])
    t = np.arange(1,T+1)/T  

    #frequencies
    freqs = t-0.5-(1/T)

    Niter = 500

    Alpha = alpha*np.ones(K)

    #construct and center f_hat, which is the fourier transform of fMirr
    f_hat = np.fft.fftshift(np.fft.fft(fMirr, axis=1), axes=1)
    f_hat_plus = np.copy(f_hat) #copy f_hat
    f_hat_plus[:,:T//2] = 0

    # Initialization of omega_k
    omega_plus = np.zeros([Niter, C, K])

    if init == 1:
        omega_plus[0,:,:] = (0.5/K)*np.arange(K)
    elif init == 2:
        omega_plus[0,:,:] = np.sort(np.exp(np.log(fs) + (np.log(0.5)-np.log(fs))*np.random.rand(1,K)))

    # if DC mode imposed, set its omega to 0
    if DC:
        omega_plus[0,:,0] = 0
    
    #start with empty dual variables
    lambda_hat = np.zeros([Niter, C, len(freqs)], dtype = complex)

    #other inits
    uDiff = tol+np.spacing(1) #initialize the difference between the iterate at (k+1)th and kth iteration
    n = 0 # loop counter
    sum_uk = np.zeros([C, len(freqs)],dtype = complex) # sum of all differences
    u_hat_plus = np.zeros([Niter, C, len(freqs), K],dtype = complex) # matrix keeping track of every iterant // could be discarded for mem

    #*** Main loop for iterative updates***
    while ( uDiff > tol and  n < Niter-1 ): # not converged and below iterations limit
        k = 0
        sum_uk = u_hat_plus[n, :, :, K - 1] + sum_uk - u_hat_plus[n, :, :, k]
        u_hat_plus[n + 1, :, :, k] = (f_hat_plus - sum_uk - lambda_hat[n, :, :] / 2) / (1 + Alpha[k] * (freqs - omega_plus[n, :, k][:, None]) ** 2)
        if not(DC):
            omega_plus[n + 1, :, k] = np.sum(freqs[T//2:T] * np.abs(u_hat_plus[n + 1, :, T//2:T, k]) ** 2, axis=1) / \
                                      np.sum(np.abs(u_hat_plus[n + 1, :, T // 2:T, k]) ** 2, axis=1)
        for k in np.arange(1,K):
            sum_uk = u_hat_plus[n+1, :, :, k - 1] + sum_uk - u_hat_plus[n, :, :, k]
            u_hat_plus[n + 1, :, :, k] = (f_hat_plus - sum_uk - lambda_hat[n, :, :] / 2) / (1 + Alpha[k] * (freqs - omega_plus[n, :, k][:, None]) ** 2)
            omega_plus[n + 1, :, k] = np.sum(freqs[T//2:T] * np.abs(u_hat_plus[n + 1, :, T//2:T, k]) ** 2, axis=1) / \
                                        np.sum(np.abs(u_hat_plus[n + 1, :, T // 2:T, k]) ** 2, axis=1)

        lambda_hat[n + 1, :, :] = lambda_hat[n, :, :] + tau * (
                np.sum(u_hat_plus[n + 1, :, :, :], axis=2) - f_hat_plus)

        n += 1
        # Calcola la differenza tra l'iterazione corrente e la precedente
        uDiff = np.spacing(1) + (1 / T) * np.sum(np.abs(u_hat_plus[n, :, :, :] - u_hat_plus[n - 1, :, :, :]) ** 2)
        # Prende il valore assoluto finale
        uDiff = np.abs(uDiff)



    
    #Postprocessing and cleanup

    #discard empty space if converged early
    Niter = np.min([Niter,n])
    omega = omega_plus[:Niter,:,:]

    idxs = np.flip(np.arange(1,T//2+1),axis = 0)
    # Signal reconstruction
    u_hat = np.zeros([C, T, K],dtype = complex)
    u_hat[:,T//2:T,:] = u_hat_plus[Niter-1,:,T//2:T,:]
    u_hat[:,idxs,:] = np.conj(u_hat_plus[Niter-1,:,T//2:T,:])
    u_hat[:,0,:] = np.conj(u_hat[:,-1,:])

    u = np.zeros([K, len(t), C])
    for k in range(K):
        for c in range(C):
            u[k,:,c] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[c,:,k])))
    
    # remove mirror part
    u = u[:,T//4:3*T//4,:]

    # recompute spectrum
    u_hat = np.zeros([K,u.shape[1],C],dtype = complex)
    for k in range(K):
        for c in range(C):
            u_hat[k,:,c] = np.fft.fftshift(np.fft.fft(u[k,:,c]))
        
    return u, u_hat, omega[-1]




    
                                    

