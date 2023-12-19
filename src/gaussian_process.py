import numpy as np
import scipy.optimize
from kernels import Kernel, ExponentiatedQuadraticKernel
from utils import compute_inv_Psi_individual_i, concatenate_Theta_Sigma_i, retrieve_Theta_Sigma_i, log_likelihood_GP
from typing import *


class GaussianProcess:
    """Gaussian Process"""

    def __init__(self, 
                mean_function: Callable=lambda x: 0, 
                kernel_k: Kernel=ExponentiatedQuadraticKernel,
                theta: Union[int, float, list, np.ndarray]=np.array([1., 1.]),
                sigma: Union[int, float]=1.,
                T: Union[list, np.ndarray]=None,
                Y: Union[list, np.ndarray]=None):

        self.mean_function = mean_function
        self.kernel_k = kernel_k
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
            fun=lambda x: log_likelihood_GP(x, self.kernel_k, self.T, self.Y, minimize=True, derivative=True),
            jac=True,
            x0=theta_sigma0,
            method='L-BFGS-B',
            options={'disp': True}
        ).x
        
        theta , sigma = retrieve_Theta_Sigma_i(theta_sigma)
        self.theta = theta
        self.sigma = sigma 
        self.K = self.kernel_k.compute(self.theta, self.T, self.T)
        self.C = self.K + (self.sigma) * np.eye(len(self.T))
        self.inv_C = scipy.linalg.pinv(self.C) + 1e-6 * np.eye(len(self.T))
        
    def get_GP_params(self):
        return np.zeros_like(self.T), self.C


    def predict(self, T_p):
        if self.T is None or self.Y is None:
            raise ValueError("The model must be trained before making predictions.")
        print(self.theta, self.sigma)
        predictions = []
        for t_p in T_p:
            c = self.kernel_k.compute(self.theta, t_p, t_p) + (self.sigma)
            k = self.kernel_k.compute(self.theta, self.T, t_p)
            mean = (k.T).dot(self.inv_C).dot(self.Y)
            variance = c - (k.T).dot(self.inv_C).dot(k)
            y = scipy.stats.norm(mean, np.sqrt(variance)).rvs()
            predictions.append(y)

        return predictions
