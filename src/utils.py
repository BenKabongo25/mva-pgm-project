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


def concatenate_Theta_Sigma_i(Theta_i: Union[int, float, np.ndarray], Sigma_i: Union[int, float, np.ndarray]) -> np.ndarray:
    if isinstance(Theta_i, (int, float)):
        Theta_i = np.array([Theta_i])
    if isinstance(Sigma_i, (int, float)):
        Sigma_i = np.array([Sigma_i])
    return np.concatenate(Theta_i, Sigma_i)


def retrieve_Theta_Sigma_i(Theta_Sigma_i: np.ndarray) -> list[Union[int, float], Union[int, float, np.ndarray]]:
    Sigma = Theta_Sigma_i[-1:]
    Theta = Theta_Sigma_i[:-1]
    return Sigma, Theta


def _log_likelihood(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, K_estim: np.ndarray) -> float:
    inv_Sigma = np.linalg.inv(Sigma)
    return multivariate_normal_density(x, mu, Sigma, inv_Sigma) - 0.5 * np.trace(K_estim @ inv_Sigma)


def log_likelihood_theta0(
        theta0: np.ndarray, 
        kernel_k: Kernel,                 
        common_T: np.ndarray,
        m0: np.ndarray,
        m0_estim: np.ndarray, 
        K_estim: np.ndarray) -> float:
    K_theta0 = kernel_k.compute_all(theta0, common_T)
    return _log_likelihood(m0_estim, m0, K_theta0, K_estim)


def derivate_log_likelihood_theta0( 
        theta0: np.ndarray, 
        kernel_k: Kernel, 
        common_T: np.ndarray,
        m0: np.ndarray,
        m0_estim: np.ndarray, 
        K_estim: np.ndarray) -> np.ndarray:

    d_theta0 = np.zeros_like(theta0)
    # TODO :
    return d_theta0
    

def log_likelihood_Theta_Sigma_Common_HP(
        Theta_Sigma: np.ndarray,
        kernel_c: Kernel,
        common_T: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray, # Y_norm instead ??
        m0_estim: np.ndarray,
        K_estim: np.ndarray) -> float:

    n_individuals = len(Y)
    n_common_T = len(common_T)
    Theta, Sigma = retrieve_Theta_Sigma_i(Theta_Sigma)

    C_Theta = kernel_c.compute_all(Theta, common_T)
    Psi_Theta_Sigma = C_Theta + Sigma * np.identity(n_common_T)
    return np.sum([ _log_likelihood(y, m0_estim, Psi_Theta_Sigma, K_estim) for y in Y])

    ## or ??
    logL = 0
    for i in range(n_individuals):
        C_Theta_i = kernel_c.compute_all(Theta, T[i])
        Psi_Theta_Sigma_i = C_Theta_i + Sigma * np.identity(len(T[i]))
        LogL += _log_likelihood(Y[i], m0_estim, Psi_Theta_Sigma_i, K_estim)
    return logL


def derivate_log_likelihood_Theta_Sigma_Common_HP(
        Theta_Sigma: np.ndarray,
        kernel_c: Kernel,
        common_T: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray, # Y_norm instead ??
        m0_estim: np.ndarray,
        K_estim: np.ndarray) -> np.ndarray:

    Theta, Sigma = retrieve_Theta_Sigma_i(Theta_Sigma)
    d_Theta = np.zeros_like(Theta)
    d_Sigma = 0
    # TODO:
    return concatenate_Theta_Sigma_i(d_Theta, d_Sigma)


def log_likelihood_Theta_Sigma_i_Different_HP(
        Theta_Sigma_i: np.ndarray,
        kernel_c: Kernel,
        common_T: np.ndarray,
        Ti: np.ndarray,
        Yi: np.ndarray, # Y_norm_i instead ??
        m0_estim: np.ndarray,
        K_estim: np.ndarray) -> float: 

    n_common_T = len(common_T)
    Theta, Sigma = retrieve_Theta_Sigma_i(Theta_Sigma_i)
    C_Theta = kernel_c.compute_all(Theta, common_T)
    Psi_Theta_Sigma = C_Theta + Sigma * np.identity(n_common_T)
    return _log_likelihood(Yi, m0_estim, Psi_Theta_Sigma, K_estim)

    ## or ??
    C_Theta = kernel_c.compute_all(Theta, Ti)
    Psi_Theta_Sigma = C_Theta + Sigma * np.identity(len(Ti))
    return _log_likelihood(Yi, m0_estim, Psi_Theta_Sigma, K_estim)


def derivate_log_likelihood_Theta_Sigma_i_Different_HP(
        Theta_Sigma_i: np.ndarray,
        kernel_c: Kernel,
        common_T: np.ndarray,
        Ti: np.ndarray,
        Yi: np.ndarray, # Y_norm_i instead ??
        m0_estim: np.ndarray,
        K_estim: np.ndarray) -> np.ndarray:
        
    Theta, Sigma = retrieve_Theta_Sigma_i(Theta_Sigma_i)
    d_Theta = np.zeros_like(Theta)
    d_Sigma = 0
    # TODO:
    return concatenate_Theta_Sigma_i(d_Theta, d_Sigma)


def log_likelihood_monitoring(
        mu0: np.ndarray, 
        m0: np.ndarray,
        Y: np.ndarray,
        inv_K_theta0: np.ndarray,
        inv_Psi_Theta_Sigma: np.ndarray) -> float:
    
    return (mu0.T @ (inv_K_theta0 + np.sum(inv_Psi_Theta_Sigma, axis=0)) @ mu0 
            - 2 * mu0.T @ (inv_K_theta0 @ m0, + np.sum(inv_Psi_Theta_Sigma, axis=0) @ Y)) ## ?? humm...


def derivate_log_likelihood_monitoring(
        mu0: np.ndarray, 
        m0: np.ndarray,
        Y: np.ndarray,
        inv_K_theta0: np.ndarray,
        inv_Psi_Theta_Sigma: np.ndarray) -> np.ndarray:
    # TODO:
    pass
