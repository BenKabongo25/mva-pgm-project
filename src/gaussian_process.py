import numpy as np
from kernels import Kernel, ExponentiatedQuadraticKernel
from utils import compute_inv_K_theta0
from typing import *


class GaussianProcess:
    """Gaussian Process """

    def __init__(self, 
                mean_function: Callable=lambda x: 0, 
                kernel: Kernel=ExponentiatedQuadraticKernel,
                theta: dict={},
                T: Union[list, np.ndarray]=None,
                Y: Union[list, np.ndarray]=None):

        self.mean_function = mean_function
        self.kernel = kernel
        self.theta = theta
        if T is not None:
            self.fit(T, Y)


    def fit(self, T: Union[list, np.ndarray], Y: Union[list, np.ndarray]):
        assert len(T) == len(Y)
        self.T = T if isinstance(T, np.ndarray) else np.array(T)
        self.Y = Y if isinstance(Y, np.ndarray) else np.array(Y)
        self.K, self.inv_K = compute_inv_K_theta0(self.kernel, self.theta, self.T)


    def predict(self, T_p):
        if self.T is None or self.Y is None:
            raise ValueError("The model must be trained before making predictions.")

        n = len(self.T)
        all_T = np.concatenate([self.T, T_p])
        Sigma, _ = compute_inv_K_theta0(self.kernel, self.theta, all_T)

        K_star = Sigma[:n, n:]
        K_star_star = Sigma[n:, n:]

        mean = self.mean_function(T_p) + K_star.T.dot(self.inv_K).dot(self.Y - self.mean_function(self.T))
        covariance = K_star_star - K_star.T.dot(self.inv_K).dot(K_star)

        return mean, covariance
