import numpy as np

from kernels import *
from typing import *


def multivariate_normal_density(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, inv_Sigma: np.ndarray=None,
                                log: bool=False) -> Union[float, np.ndarray]:

    d, z, det_Sigma, inv_Sigma = _check_params_multivariate_normal(x, mu, Sigma, inv_Sigma)
    
    if z.ndim == 1:
        exponent = -0.5 * ((z.T @ inv_Sigma) @ z)
    else:
        exponent = np.array([-0.5 * ((zi.T @ inv_Sigma) @ zi) for zi in z])
    
    if not log:
        return (1 / np.sqrt((2 * np.pi) ** d * det_Sigma)) * np.exp(exponent)
    return - (d/2) * np.log(2 * np.pi) - 0.5 * np.log(det_Sigma) + exponent


def _check_params_multivariate_normal(x, mu, Sigma, inv_Sigma):
    assert mu.ndim == 1
    d = len(mu)

    if x.ndim == 1:
        assert len(x) == d
    elif x.ndim == 2:
        assert x.shape[1] == d
        mu = np.tile(mu, (len(x), 1))

    assert Sigma.shape == (d, d)
    
    det_Sigma = np.linalg.det(Sigma)
    assert det_Sigma >= 0
    
    if inv_Sigma is None:
        inv_Sigma = np.linalg.inv(Sigma)
    else:
        assert inv_Sigma.shape == Sigma.shape

    return d, x - mu, det_Sigma, inv_Sigma


def _flatten_Theta_Sigma(Theta: Union[int, float, np.ndarray], Sigma: Union[int, float, np.ndarray], 
                        common_hp_flag: bool=True) -> np.ndarray:
    if common_hp_flag:
        if isinstance(Theta, (int, float)):
            Theta = np.array([Theta])
        if isinstance(Sigma, (int, float)):
            Sigma = np.array([Sigma])

    Theta_flat = Theta.flatten()
    Sigma_flat = Sigma.flatten()

    return np.concatenate(Theta_flat, Sigma_flat)


def _retrieve_Theta_Sigma(self, Theta_Sigma: np.ndarray, n_individuals: int,
                        common_hp_flag: bool=True) -> list[Union[int, float, np.ndarray], Union[int, float, np.ndarray]]:
        if common_hp_flag:
            Sigma = Theta_Sigma[-1:]
            Theta = Theta_Sigma[:-1]

        else:
            Sigma = Theta_Sigma[-n_individuals:]
            Theta_flat = Theta_Sigma[:-n_individuals]
            Theta = Theta_flat.reshape((n_individuals, -1))

        return Sigma, Theta


def log_likelihood_theta0(theta0: np.ndarray, 
                        kernel_k: Kernel, 
                        common_T: np.ndarray,
                        m0: np.ndarray,
                        m0_estim: np.ndarray, 
                        K_estim: np.ndarray) -> float:
    K_theta0 = kernel_k.compute_all(theta0, common_T)
    return _log_likelihood(m0_estim, m0, K_theta0, K_estim)


def derivate_log_likelihood_theta0(theta0: np.ndarray, 
                        kernel_k: Kernel, 
                        common_T: np.ndarray,
                        m0: np.ndarray,
                        m0_estim: np.ndarray, 
                        K_estim: np.ndarray) -> np.ndarray:
    pass
    

def _log_likelihood(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, K_estim: np.ndarray) -> float:
    inv_Sigma = np.linalg.inv(Sigma)
    return multivariate_normal_density(x, mu, Sigma) - 0.5 * np.trace(K_estim @ inv_Sigma)