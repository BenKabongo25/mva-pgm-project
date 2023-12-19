import numpy as np
import scipy.optimize
from kernels import Kernel, ExponentiatedQuadraticKernel
from utils import compute_inv_Psi_individual_i, concatenate_Theta_Sigma_i, retrieve_Theta_Sigma_i, log_likelihood_GP
from typing import *


class GaussianProcess:
    """Gaussian Process"""

    def __init__(self, 
                mean_function: Callable=lambda x: 0, 
                kernel: Kernel=ExponentiatedQuadraticKernel,
                theta: Union[int, float, list, np.ndarray]=0,
                sigma: Union[int, float]=1,
                T: Union[list, np.ndarray]=None,
                Y: Union[list, np.ndarray]=None):

        self.mean_function = mean_function
        self.kernel = kernel
        self.theta = theta
        assert sigma > 0
        self.sigma = sigma
        if T is not None:
            self.fit(T, Y)


    def fit(self, T: Union[list, np.ndarray], Y: Union[list, np.ndarray]):
        assert len(T) == len(Y)
        self.T = T if isinstance(T, np.ndarray) else np.array(T)
        self.Y = Y if isinstance(Y, np.ndarray) else np.array(Y)

        theta_sigma0 = concatenate_Theta_Sigma_i(self.theta, self.sigma)
        theta_sigma = scipy.optimize.minimize(
            #TODO:
        ).x
        self.theta, self.sigma = retrieve_Theta_Sigma_i(theta_sigma)
        self.K, self.inv_K = compute_inv_Psi_individual_i(self.kernel, self.theta, self.sigma, self.T, None)


    def predict(self, T_p):
        if self.T is None or self.Y is None:
            raise ValueError("The model must be trained before making predictions.")

        n = len(self.T)
        all_T = np.concatenate([self.T, T_p])
        Sigma, _ = compute_inv_Psi_individual_i(self.kernel, self.theta, self.sigma, all_T, None)

        K_star = Sigma[:n, n:]
        K_star_star = Sigma[n:, n:]

        mean = self.mean_function(T_p) + K_star.T.dot(self.inv_K).dot(self.Y - self.mean_function(self.T))
        covariance = K_star_star - K_star.T.dot(self.inv_K).dot(K_star)

        return mean, covariance
