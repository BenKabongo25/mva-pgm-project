import numpy as np
import scipy.linalg
import scipy.optimize
from kernels import Kernel, ExponentiatedQuadraticKernel
from utils import  (compute_inv_Psi_individual_i, 
                    concatenate_Theta_Sigma_i, 
                    retrieve_Theta_Sigma_i, 
                    log_likelihood_learn_new_parameters)
from typing import *


class GaussianProcess:
    """Gaussian Process"""

    def __init__(self, 
                mean_function: Callable=lambda x: np.zeros_like(x), 
                kernel: Kernel=ExponentiatedQuadraticKernel,
                theta: Union[int, float, list, np.ndarray]=np.array([1., 1.]),
                sigma: Union[int, float]=1.,
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
        self.mean = self.mean_function(self.T)
        theta_sigma0 = concatenate_Theta_Sigma_i(self.theta, self.sigma)
        theta_sigma = scipy.optimize.minimize(
            fun=lambda x: log_likelihood_learn_new_parameters(x, self.kernel, self.T, self.Y, self.mean,
                                                            minimize=True, derivative=True),
            jac=True,
            x0=theta_sigma0,
            method='L-BFGS-B',
            options={'disp': False}
        ).x
        theta , sigma = retrieve_Theta_Sigma_i(theta_sigma)
        self.theta = theta
        self.sigma = sigma 


    def predict(self, T_p):
        if self.T is None or self.Y is None:
            raise ValueError("The model must be trained before making predictions.")

        n = len(self.T)
        all_T = np.concatenate([self.T, T_p])
        Sigma, _ = compute_inv_Psi_individual_i(self.kernel, self.theta, self.sigma, all_T, None)

        K = Sigma[:n, :n] + 1e-6 * np.identity(n)
        inv_K = scipy.linalg.pinv(K)
        K_star = Sigma[:n, n:]
        K_star_star = Sigma[n:, n:]

        mean = self.mean_function(T_p) + K_star.T.dot(inv_K).dot(self.Y - self.mean_function(self.T))
        covariance = K_star_star - K_star.T.dot(inv_K).dot(K_star)

        return mean, covariance
