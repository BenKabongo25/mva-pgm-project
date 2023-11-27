import numpy as np
import scipy.linalg
from kernels import *
from typing import *


def make_grids(X: Union[list, np.ndarray]) -> list[np.ndarray, np.ndarray]:
    X = X if isinstance(X, np.ndarray) else np.ndarray(X)
    N = len(X)
    XX = np.tile(X, (N, 1)).T
    YY = XX.T
    return XX, YY


def mask_square(mask: np.ndarray) -> np.ndarray:
    _XX, _YY = make_grids(mask)
    return _XX * _YY


def compute_inv_K_theta0(
        kernel_k: Kernel, 
        theta0: Union[int, float, list, np.ndarray], 
        T: Union[list, np.ndarray]) -> list[np.ndarray, np.ndarray]:
    K_theta0 = kernel_k.compute_all(theta0, T)
    inv_K_theta0 = scipy.linalg.pinv(K_theta0)
    return K_theta0, inv_K_theta0

    
def compute_inv_Psi_individual_i(
        kernel_c: Kernel, 
        Theta: Union[int, float, list, np.ndarray], 
        sigma: Union[int, float],
        Ti: np.ndarray, 
        mask: np.ndarray=None) -> list[np.ndarray, np.ndarray]:
    C_Theta_i = kernel_c.compute_all(Theta, Ti)
    Psi_Theta_Sigma_i = C_Theta_i + sigma * np.identity(len(Ti))
    if mask is not None:
        Psi_Theta_Sigma_i = mask_square(mask) * Psi_Theta_Sigma_i
    inv_Psi_Theta_Sigma_i = scipy.linalg.pinv(Psi_Theta_Sigma_i)
    return Psi_Theta_Sigma_i, inv_Psi_Theta_Sigma_i


def multivariate_normal_density(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, inv_Sigma: np.ndarray=None,
                                log: bool=False) -> Union[float, np.ndarray]:
    """Compute the multivariate normal density.

    Args:
        x (np.ndarray): Data points.
        mu (np.ndarray): Mean vector.
        Sigma (np.ndarray): Covariance matrix.
        inv_Sigma (np.ndarray, optional): Inverse of the covariance matrix. Defaults to None.
        log (bool, optional): Whether to compute the log-density. Defaults to False.

    Returns:
        Union[float, np.ndarray]: Multivariate normal density or log-density.
    """
    d, z, inv_Sigma = _check_params_multivariate_normal(x, mu, Sigma, inv_Sigma)
    
    if z.ndim == 1:
        exponent = -0.5 * ((z.T).dot(inv_Sigma).dot(z))
    else:
        exponent = np.array([-0.5 * ((zi.T).dot(inv_Sigma).dot(zi)) for zi in z])

    if not log:
        return (1 / ((np.sqrt((2 * np.pi) ** d) * np.sqrt(scipy.linalg.det(Sigma))))) * np.exp(exponent)
    return - (d/2) * np.log(2 * np.pi) - 0.5 * np.linalg.slogdet(Sigma)[1] + exponent  


def _check_params_multivariate_normal(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, inv_Sigma: np.ndarray
    ) -> list[int, np.ndarray, float, np.ndarray]:
    """Check and preprocess parameters for multivariate normal density calculation.

    Args:
        x (np.ndarray): Data points.
        mu (np.ndarray): Mean vector.
        Sigma (np.ndarray): Covariance matrix.
        inv_Sigma (np.ndarray): Inverse of the covariance matrix.

    Returns:
        list[int, np.ndarray, float, np.ndarray]: List containing processed parameters.
    """
    assert mu.ndim == 1 # Ensure mu is a vector (1D array of the means)
    d = len(mu)

    if x.ndim == 1: assert len(x) == d
    elif x.ndim == 2:
        assert x.shape[1] == d
        mu = np.tile(mu, (len(x), 1))

    assert Sigma.shape == (d, d)
    if inv_Sigma is None: inv_Sigma = np.linalg.pinv(Sigma)  # Use pseudo-inverse for numerical stability
    else: assert inv_Sigma.shape == Sigma.shape

    return d, x - mu, inv_Sigma


def concatenate_Theta_Sigma_i(Theta_i: Union[int, float, np.ndarray], Sigma_i: Union[int, float, np.ndarray]) -> np.ndarray:
    """Concatenate Theta_i and Sigma_i into a single array.

    Args:
        Theta_i (Union[int, float, np.ndarray]): Theta_i parameter.
        Sigma_i (Union[int, float, np.ndarray]): Sigma_i parameter.

    Returns:
        np.ndarray: Concatenated array.
    """
    if isinstance(Theta_i, (int, float)):
        Theta_i = np.array([Theta_i])
    if isinstance(Sigma_i, (int, float)):
        Sigma_i = np.array([Sigma_i])
    return np.concatenate([Theta_i, Sigma_i])


def retrieve_Theta_Sigma_i(Theta_Sigma_i: np.ndarray) -> list[Union[int, float, np.ndarray], Union[int, float]]:
    """Extract Theta and Sigma from the concatenated array.

    Args:
        Theta_Sigma_i (np.ndarray): Concatenated array.

    Returns:
        list[Union[int, float, np.ndarray], Union[int, float]]: List containing Theta and Sigma.
    """
    Theta = Theta_Sigma_i[:-1]
    Sigma = Theta_Sigma_i[-1]
    Sigma = Sigma + 1e-6
    return Theta, Sigma


def _log_likelihood(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, inv_Sigma: np.ndarray, K_estim: np.ndarray) -> float:
    """Compute the log-likelihood for a given set of parameters.

    Args:
        x (np.ndarray): Data points.
        mu (np.ndarray): Mean vector.
        Sigma (np.ndarray): Covariance matrix.
        inv_Sigma (np.ndarray): Inverse of the covariance matrix.
        K_estim (np.ndarray): Estimated covariance matrix at the current iteration.

    Returns:
        float: Log-likelihood value.
    """
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
    """Compute the log-likelihood for theta0.

    Args:
        theta0 (np.ndarray): Parameter vector.
        kernel_k (Kernel): Kernel function.
        common_T (np.ndarray): Common time points.
        m0 (np.ndarray): Mean vector.
        m0_estim (np.ndarray): Estimated mean vector.
        K_estim (np.ndarray): Estimated covariance matrix.
        minimize (bool, optional): Whether to minimize or maximize. Defaults to False.
        derivative (bool, optional): Whether to compute derivatives. Defaults to False.

    Returns:
        list[float, np.ndarray]: Log-likelihood value and derivative (if requested).
    """
    factor = -1 if minimize else 1
    K_theta0, inv_K_theta0 = compute_inv_K_theta0(kernel_k, theta0, common_T)
    LL_theta0 = factor * _log_likelihood(m0_estim, m0, K_theta0, inv_K_theta0, K_estim)
    if not derivative: return LL_theta0

    z = (m0_estim - m0)[:, np.newaxis]
    d_theta0 = np.zeros_like(theta0)        
    d_K_theta0 = (-0.5 * inv_K_theta0 
                  + 0.5 * inv_K_theta0.dot(np.outer(z, z)).dot(inv_K_theta0)
                  + 0.5 * inv_K_theta0.dot(K_estim).dot(inv_K_theta0))
    d_theta0_of_K_theta0 = kernel_k.derivate_parameters(theta0, common_T)
    for i in range(len(theta0)):
        d_theta0[i] = (d_K_theta0 * d_theta0_of_K_theta0[i]).sum()

    return LL_theta0, d_theta0


def log_likelihood_Theta_Sigma_Common_HP(
        Theta_Sigma: np.ndarray,
        kernel_c: Kernel,
        common_T: np.ndarray,
        T_masks: np.ndarray,
        Y: np.ndarray,
        m0_estim: np.ndarray,
        K_estim: np.ndarray,
        minimize: bool=False,
        derivative: bool=False) -> list[float, np.ndarray]:
    """Compute the log-likelihood for Theta and Sigma with common hyperparameters.

    Args:
        Theta_Sigma (np.ndarray): Concatenated array of Theta and Sigma.
        kernel_c (Kernel): Kernel function.
        common_T (np.ndarray): Common time points.
        T_masks (np.ndarray): Individual time mask.
        Y (np.ndarray): Data points.
        m0_estim (np.ndarray): Estimated mean vector.
        K_estim (np.ndarray): Estimated covariance matrix.
        minimize (bool, optional): Whether to minimize or maximize. Defaults to False.
        derivative (bool, optional): Whether to compute derivatives. Defaults to False.

    Returns:
        list[float, np.ndarray]: Log-likelihood value and derivative (if requested).
    """
    factor = -1 if minimize else 1

    n_individuals = len(Y)
    n_common_T = len(common_T)
    Theta, Sigma = retrieve_Theta_Sigma_i(Theta_Sigma)

    LL_Theta_Sigma = 0
    if derivative:
        d_Theta_Sigma = np.zeros_like(Theta_Sigma)
        d_Psi_Theta_Sigma = 0

    for i in range(n_individuals):
        Ti_mask = None if T_masks is None else T_masks[i]
        Psi_Theta_Sigma_i, inv_Psi_Theta_Sigma_i = compute_inv_Psi_individual_i(kernel_c, Theta, Sigma, common_T, Ti_mask)
        LL_Theta_Sigma += _log_likelihood(Y[i], m0_estim, Psi_Theta_Sigma_i, inv_Psi_Theta_Sigma_i, K_estim)

        if derivative:
            z = (Y[i] - m0_estim)[:, np.newaxis]
            d_Psi_Theta_Sigma += (- 0.5 * inv_Psi_Theta_Sigma_i 
                                  + 0.5 * inv_Psi_Theta_Sigma_i.dot(np.outer(z, z)).dot(inv_Psi_Theta_Sigma_i)
                                  + 0.5 * inv_Psi_Theta_Sigma_i.dot(K_estim).dot(inv_Psi_Theta_Sigma_i))

    LL_Theta_Sigma = factor * LL_Theta_Sigma
    
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
        Ti_mask: np.ndarray,
        Yi: np.ndarray,
        m0_estim: np.ndarray,
        K_estim: np.ndarray,
        minimize: bool=False,
        derivative: bool=False) -> list[float, np.ndarray]: 
    """Compute the log-likelihood for individual Theta and Sigma with different hyperparameters.

    Args:
        Theta_Sigma_i (np.ndarray): Concatenated array of individual Theta and Sigma.
        kernel_c (Kernel): Kernel function.
        common_T (np.ndarray): Common time points.
        Ti_mask (np.ndarray): Individual time mask.
        Yi (np.ndarray): Data points.
        m0_estim (np.ndarray): Estimated mean vector.
        K_estim (np.ndarray): Estimated covariance matrix.
        minimize (bool, optional): Whether to minimize or maximize. Defaults to False.
        derivative (bool, optional): Whether to compute derivatives. Defaults to False.

    Returns:
        list[float, np.ndarray]: Log-likelihood value and derivative (if requested).
    """
    factor = -1 if minimize else 1
    Theta, Sigma = retrieve_Theta_Sigma_i(Theta_Sigma_i)
    Psi_Theta_Sigma, inv_Psi_Theta_Sigma = compute_inv_Psi_individual_i(kernel_c, Theta, Sigma, common_T, Ti_mask)
    LL_Theta_Sigma = factor * _log_likelihood(Yi, m0_estim, Psi_Theta_Sigma, inv_Psi_Theta_Sigma, K_estim)
    if not derivative: return LL_Theta_Sigma

    n_common_T = len(common_T)
    z = (Yi - m0_estim)[:, np.newaxis]
    d_Theta_Sigma = np.zeros_like(Theta_Sigma_i)
    d_Psi_Theta_Sigma = (- 0.5 * inv_Psi_Theta_Sigma 
                         + 0.5 * inv_Psi_Theta_Sigma.dot(np.outer(z, z)).dot(inv_Psi_Theta_Sigma)
                         + 0.5 * inv_Psi_Theta_Sigma.dot(K_estim).dot(inv_Psi_Theta_Sigma))

    d_Theta_of_Psi_Theta_Sigma = kernel_c.derivate_parameters(Theta, common_T)
    for i in range(len(Theta)):
        d_Theta_Sigma[i] = (d_Psi_Theta_Sigma * d_Theta_of_Psi_Theta_Sigma[i]).sum()
    d_Theta_Sigma[-1] = (d_Psi_Theta_Sigma * Sigma * np.identity(n_common_T)).sum()
    d_Theta_Sigma = factor * d_Theta_Sigma

    return LL_Theta_Sigma, d_Theta_Sigma
