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
    return - (d/2) * np.log(2 * np.pi) - 0.5 * np.log(det_Sigma) + exponent  #il y a np.linalg.slogdet qui est plus stable


def _check_params_multivariate_normal(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, inv_Sigma: np.ndarray
    ) -> list[int, np.ndarray, float, np.ndarray]:
    assert mu.ndim == 1
    d = len(mu)

    if x.ndim == 1: assert len(x) == d
    elif x.ndim == 2:
        assert x.shape[1] == d
        mu = np.tile(mu, (len(x), 1))

    assert Sigma.shape == (d, d)
    det_Sigma = np.linalg.det(Sigma)
    det_Sigma = np.abs(det_Sigma) # ??? log (x) x > 0
    #assert det_Sigma != 0
    if inv_Sigma is None: inv_Sigma = np.linalg.inv(Sigma)
    else: assert inv_Sigma.shape == Sigma.shape

    return d, x - mu, det_Sigma, inv_Sigma


def concatenate_Theta_Sigma_i(Theta_i: Union[int, float, np.ndarray], Sigma_i: Union[int, float, np.ndarray]) -> np.ndarray:
    if isinstance(Theta_i, (int, float)):
        Theta_i = np.array([Theta_i])
    if isinstance(Sigma_i, (int, float)):
        Sigma_i = np.array([Sigma_i])
    return np.concatenate([Theta_i, Sigma_i])


def retrieve_Theta_Sigma_i(Theta_Sigma_i: np.ndarray) -> list[Union[int, float], Union[int, float, np.ndarray]]:
    Theta = Theta_Sigma_i[:-1]
    Sigma = Theta_Sigma_i[-1]
    return Theta, Sigma


def _log_likelihood(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, inv_Sigma: np.ndarray, K_estim: np.ndarray) -> float:
    return multivariate_normal_density(x, mu, Sigma, inv_Sigma, log=True) - 0.5 * np.trace(K_estim @ inv_Sigma)


def log_likelihood_theta0(
        theta0: np.ndarray, 
        kernel_k: Kernel,                 
        common_T: np.ndarray,
        m0: np.ndarray,
        m0_estim: np.ndarray, 
        K_estim: np.ndarray,
        minimize: bool=False,
        derivative: bool=False) -> list[float, np.ndarray]:

    factor = -1 if minimize else 1

    K_theta0 = kernel_k.compute_all(theta0, common_T)
    inv_K_theta0 = np.linalg.inv(K_theta0) #pinv(K_theta0) for numerical stability

    LL_theta0 = factor * _log_likelihood(m0_estim, m0, K_theta0, inv_K_theta0, K_estim)
    #print("LL_theta0 :", LL_theta0)
    if not derivative:
        return LL_theta0

    z = (m0_estim - m0)[:, np.newaxis]
    d_theta0 = np.zeros_like(theta0)        
    d_K_theta0 = (- 0.5 * inv_K_theta0 
                  + 0.5 * inv_K_theta0 @ z @ z.T @ inv_K_theta0 
                  + 0.5 * K_estim @ inv_K_theta0 @ inv_K_theta0) # pas sur mais je crois que les dot sont plus stables numériquement
    d_theta0_of_K_theta0 = kernel_k.derivate_parameters(theta0, common_T)
    for i in range(len(theta0)):
        d_theta0[i] = (d_K_theta0 * d_theta0_of_K_theta0[i]).sum()

    return LL_theta0, d_theta0


def log_likelihood_Theta_Sigma_Common_HP(
        Theta_Sigma: np.ndarray,
        kernel_c: Kernel,
        common_T: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        m0_estim: np.ndarray,
        K_estim: np.ndarray,
        minimize: bool=False,
        derivative: bool=False) -> list[float, np.ndarray]:

    factor = -1 if minimize else 1

    n_individuals = len(Y)
    n_common_T = len(common_T)
    Theta, Sigma = retrieve_Theta_Sigma_i(Theta_Sigma)

    C_Theta = kernel_c.compute_all(Theta, common_T)
    Psi_Theta_Sigma = C_Theta + Sigma * np.identity(n_common_T)
    inv_Psi_Theta_Sigma = np.linalg.inv(Psi_Theta_Sigma)

    LL_Theta_Sigma = 0
    if derivative:
        d_Theta_Sigma = np.zeros_like(Theta_Sigma)
        d_Psi_Theta_Sigma = 0

    for i in range(n_individuals):
        #C_Theta_i = kernel_c.compute_all(Theta, T[i])
        #Psi_Theta_Sigma_i = C_Theta_i + Sigma * np.identity(len(T[i]))
        #inv_Psi_Theta_Sigma_i = np.linalg.inv(Psi_Theta_Sigma_i)
        LL_Theta_Sigma += _log_likelihood(Y[i], m0_estim, Psi_Theta_Sigma, inv_Psi_Theta_Sigma, K_estim)

        if derivative:
            z = (Y[i] - m0_estim)[:, np.newaxis]
            d_Psi_Theta_Sigma += (- 0.5 * inv_Psi_Theta_Sigma 
                                  + 0.5 * inv_Psi_Theta_Sigma @ z @ z.T @ inv_Psi_Theta_Sigma 
                                  + 0.5 * K_estim @ inv_Psi_Theta_Sigma @ inv_Psi_Theta_Sigma)

    LL_Theta_Sigma = factor * LL_Theta_Sigma
    #print("LL_Theta_Sigma :", LL_Theta_Sigma)
    if not derivative:
        return LL_Theta_Sigma

    d_Theta_of_Psi_Theta_Sigma = kernel_c.derivate_parameters(Theta, common_T)
    for i in range(len(Theta)):
        d_Theta_Sigma[i] = (d_Psi_Theta_Sigma * d_Theta_of_Psi_Theta_Sigma[i]).sum()
    d_Theta_Sigma[-1] = (d_Psi_Theta_Sigma * Sigma * np.identity(n_common_T)).sum()
    d_Theta_Sigma = factor * d_Theta_Sigma

    return LL_Theta_Sigma, d_Theta_Sigma


def log_likelihood_Theta_Sigma_i_Different_HP(
        Theta_Sigma_i: np.ndarray,
        kernel_c: Kernel,
        common_T: np.ndarray,
        Ti: np.ndarray,
        Yi: np.ndarray,
        m0_estim: np.ndarray,
        K_estim: np.ndarray,
        minimize: bool=False,
        derivative: bool=False) -> list[float, np.ndarray]: 

    factor = -1 if minimize else 1

    n_common_T = len(common_T)
    Theta, Sigma = retrieve_Theta_Sigma_i(Theta_Sigma_i)

    C_Theta = kernel_c.compute_all(Theta, common_T)
    Psi_Theta_Sigma = C_Theta + Sigma * np.identity(n_common_T)
    inv_Psi_Theta_Sigma = np.linalg.inv(Psi_Theta_Sigma)

    ## or ??
    #C_Theta = kernel_c.compute_all(Theta, Ti)
    #Psi_Theta_Sigma = C_Theta + Sigma * np.identity(len(Ti))
    #inv_Psi_Theta_Sigma = np.linalg.inv(Psi_Theta_Sigma)
    
    LL_Theta_Sigma = factor * _log_likelihood(Yi, m0_estim, Psi_Theta_Sigma, inv_Psi_Theta_Sigma, K_estim)
    #print("LL_Theta_Sigma :", LL_Theta_Sigma)
    if not derivative:
        return LL_Theta_Sigma

    z = (Yi - m0_estim)[:, np.newaxis]
    d_Theta_Sigma = np.zeros_like(Theta_Sigma_i)
    d_Psi_Theta_Sigma = (- 0.5 * inv_Psi_Theta_Sigma 
                        + 0.5 * inv_Psi_Theta_Sigma @ z @ z.T @ inv_Psi_Theta_Sigma 
                        + 0.5 * K_estim @ inv_Psi_Theta_Sigma @ inv_Psi_Theta_Sigma)

    d_Theta_of_Psi_Theta_Sigma = kernel_c.derivate_parameters(Theta, common_T)
    for i in range(len(Theta)):
        d_Theta_Sigma[i] = (d_Psi_Theta_Sigma * d_Theta_of_Psi_Theta_Sigma[i]).sum()
    d_Theta_Sigma[-1] = (d_Psi_Theta_Sigma * Sigma * np.identity(n_common_T)).sum()
    d_Theta_Sigma = factor * d_Theta_Sigma

    return LL_Theta_Sigma, d_Theta_Sigma
    
