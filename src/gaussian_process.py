import numpy as np
import scipy.optimize
from typing import Callable, Union, List, Tuple
from kernels import Kernel, ExponentiatedQuadraticKernel
from utils import (
    concatenate_Theta_Sigma_i,
    retrieve_Theta_Sigma_i,
    compute_inv_Psi_individual_i,
    log_likelihood_learn_new_parameters,)

class GaussianProcess:
    """
    Gaussian Process regression model.

    This class represents a Gaussian Process (GP) regression model, which is a non-parametric
    approach for regression tasks. It models the distribution over functions, making it
    suitable for various applications, including surrogate modeling and uncertainty quantification.

    Attributes:
        - mean_function (Callable): Function representing the mean of the Gaussian Process.
        - kernel (Kernel): Kernel function defining the covariance between data points.
        - theta (Union[int, float, list, np.ndarray]): Parameters of the kernel function.
        - sigma (Union[int, float]): Noise level in the data.
        - T (np.ndarray): Training input data.
        - Y (np.ndarray): Training output data.
    """

    def __init__(self, 
                mean_function: Callable=lambda x: np.zeros_like(x), 
                kernel: Kernel=ExponentiatedQuadraticKernel,
                theta: Union[int, float, list, np.ndarray]=np.array([1., 1.]),
                sigma: Union[int, float]=1.,
                T: Union[list, np.ndarray]=None,
                Y: Union[list, np.ndarray]=None):
        """
        Initialize the Gaussian Process model.

        Parameters:
            - mean_function: Callable, optional
                Function representing the mean of the Gaussian Process.
                Default is a function returning zeros.
            - kernel: Kernel, optional
                Kernel function defining the covariance between data points.
                Default is the ExponentiatedQuadraticKernel.
            - theta: Union[int, float, list, np.ndarray], optional
                Parameters of the kernel function.
                Default is np.array([1., 0.5]).
            - sigma: Union[int, float], optional
                Noise level in the data.
                Default is 1. .
            - T: Union[list, np.ndarray], optional
                Training input data.
            - Y: Union[list, np.ndarray], optional
                Training output data.
        """
        self.mean_function = mean_function
        self.kernel = kernel
        self.theta = theta
        assert sigma > 0
        self.sigma = sigma
        if T is not None:
            self.fit(T, Y)

    def fit(self, T: Union[list, np.ndarray], Y: Union[list, np.ndarray]):
        """
        Fit the Gaussian Process model to the training data.

        Parameters:
            - T: Union[list, np.ndarray]
                Training input data.
            - Y: Union[list, np.ndarray]
                Training output data.
        """
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
        """
        Make predictions using the Gaussian Process model.

        Parameters:
            - T_p: Union[list, np.ndarray]
                Input data for predictions.

        Returns:
            Tuple[np.ndarray, np.ndarray]
                Mean and covariance of the predictions.
        """
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

    """
    def fit(self, T: Union[list, np.ndarray], Y: Union[list, np.ndarray]):
        assert len(T) == len(Y)
        self.T = T if isinstance(T, np.ndarray) else np.array(T)
        self.Y = Y if isinstance(Y, np.ndarray) else np.array(Y)

        theta_sigma0 = concatenate_Theta_Sigma_i(self.theta, self.sigma)
        theta_sigma = scipy.optimize.minimize(
            fun=lambda x: log_likelihood_GP(x, self.kernel, self.T, self.Y, minimize=True, derivative=True),
            jac=True,
            x0=theta_sigma0,
            method='L-BFGS-B',
            tol = 1e-2,
            options={'maxiter': 100, 'disp': True}
            
        ).x
        
        theta = theta_sigma[:-1]
        sigma = theta_sigma[-1]
        self.theta = theta
        self.sigma = abs(sigma)
        self.K = self.kernel.compute(self.theta, self.T, self.T)
        self.C = self.K + (self.sigma) * np.eye(len(self.T))
        self.inv_C = scipy.linalg.pinv(self.C) 
        
    def get_GP_params(self):
        return np.zeros_like(self.T), self.C


    def predict(self, T_p):
        if self.T is None or self.Y is None:
            raise ValueError("The model must be trained before making predictions.")
        print("theta, sigma")
        print(self.theta, self.sigma)
        predictions = []
        for t_p in T_p:
            c = self.kernel._apply(self.theta[0],self.theta[1], t_p, t_p) + (self.sigma)
            k = self.kernel.compute(self.theta, self.T, t_p)
            mean = (k.T).dot(self.inv_C).dot(self.Y)
            variance = c - (k.T).dot(self.inv_C).dot(k)
            print("mean, variance")
            print(mean, variance)
            y = scipy.stats.norm(mean, np.sqrt(variance)).rvs()
            predictions.append(y)

        return predictions"""