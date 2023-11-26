import numpy as np
import scipy.optimize
from typing import *

from kernels import (Kernel, 
                    ExponentiatedQuadraticKernel, 
                    GaussianKernel)
from utils import (log_likelihood_theta0, 
                   log_likelihood_Theta_Sigma_Common_HP,
                   log_likelihood_Theta_Sigma_i_Different_HP,
                   log_likelihood_monitoring,
                   concatenate_Theta_Sigma_i,
                   retrieve_Theta_Sigma_i)


class MAGMA:
    """Multi-tAsk Gaussian processes with common MeAn

    :param T
    :param Y
    :param common_T
    :param m0: prior mean
    :param theta0: hyper-parameters for covariance kernel k_theta0(.|.)
    :param Theta: hyper-parameters for covariance kernel c_theta_i(.|.)
    :param Sigma: noise variance associated with the i-th individual 
    :param kernel_k: kernel k_theta0 object
    :param kernel_c: kernel c_theta_i object
    :param common_hp_flag: true => common hyper parameters theta and sigma
    :param save_history_flag
    """

    def __init__(self,
                T: Union[np.ndarray, list[np.ndarray]],
                Y: Union[np.ndarray, list[np.ndarray]],
                common_T: Union[list, np.ndarray]=None,
                m0: Union[int, float, list, np.ndarray]=None, 
                theta0: Union[int, float, list, np.ndarray]=None, 
                Theta: Union[int, float, list, np.ndarray]=None, 
                Sigma: Union[int, float, list, np.ndarray]=None,
                common_hp_flag: bool=True,
                save_history_flag: bool=False,
                kernel_k: Kernel=ExponentiatedQuadraticKernel,
                kernel_c: Kernel=ExponentiatedQuadraticKernel,
        ):
        self.set_common_T(common_T, T)
        self.n_common_T = len(common_T)

        self.set_TY(T, Y) # self.T, self.Y
        self.n_individuals = len(self.Y)

        self.common_hp_flag = common_hp_flag
        self.save_history_flag = save_history_flag

        self.kernel_k = kernel_k
        self.kernel_c = kernel_c

        self.m0_estim = None
        self.K = None

        self.set_m0(m0)
        self.set_theta0(theta0)
        self.set_Theta(Theta)
        self.set_Sigma(Sigma)
        self.init_history()


    def set_common_T(self, common_T: Union[list, np.ndarray], T: Union[np.ndarray, list[np.ndarray]]=None) -> None: 
        if common_T is not None:
            self.common_T = common_T
        else:
            assert T is not None
            all_ti = []
            for t in T: all_ti.extend(list(t))
            self.common_T = np.unique(all_ti)


    def set_TY(self, T: Union[np.ndarray, list[np.ndarray]], Y: Union[np.ndarray, list[np.ndarray]]) -> None:
        if T is None:
            assert self.common_T is not None
            T = np.tile(self.common_T, (len(Y), 1))

        for (t, y) in zip(T, Y):
            assert len(t) == len(y)

        self.T = T
        self.Y = Y

    
#    def _normalize_Y(self, T: Union[np.ndarray, list(np.ndarray)], Y: Union[np.ndarray, list[np.ndarray]]) -> None:
#        assert self.common_T is not None
#        Y_norm = np.zeros(self.n_individuals, self.n_common_T, len(Y[0][0]))
#        for idx in range(self.n_individuals):
#            uniques_ti, idx_in_T = np.unique(T[idx], return_index=True)
#            idx_in_common_T = np.where(np.isin(self.common_T, uniques_ti, assume_unique=True))[0]
#            Y_norm[idx, idx_in_common_T] = Y[idx][idx_in_T]
#        self.Y_norm = Y_norm


    def set_m0(self, m0: Union[int, float, list, np.ndarray]) -> None:
        if m0 is None:
            m0 = np.zeros(self.n_common_T)
        elif isinstance(m0, (int, float)):
            m0 = m0 * np.ones(self.n_common_T)
        elif isinstance(m0, (list, tuple)):
            m0 = np.array(m0)
        assert isinstance(m0, np.ndarray) and len(m0) == self.n_common_T
        self.m0 = m0


    def set_theta0(self, theta0: Union[int, float, list, np.ndarray]) -> None:
        if theta0 is None:
            theta0 = self.kernel_k.init_parameters()
        self.theta0 = theta0


    def set_Theta(self, Theta: Union[int, float, list, np.ndarray]) -> None:
        if Theta is None:
            if self.common_hp_flag: 
                Theta = self.kernel_c.init_parameters()
            else: Theta = np.array([self.kernel_c.init_parameters() for _ in range(self.n_individuals)])
        else:
            if not self.common_hp_flag: 
                assert len(Theta) == self.n_individuals
        self.Theta = Theta


    def set_Sigma(self, Sigma: Union[int, float, list, np.ndarray]) -> None:
        if Sigma is None:
            if self.common_hp_flag: Sigma = np.random.random()
            else: Sigma = np.random.random(self.n_individuals)
        else:
            if not self.common_hp_flag:
                assert len(Sigma) == self.n_individuals
        self.Sigma = Sigma


    def init_history(self) -> None:
        self.history = []
        self.save_history()


    def save_history(self) -> None:
        if self.save_history_flag:
            self.history.append({
                "m0": self.m0_estim,
                "K": self.K,
                "theta0": self.theta0,
                "Theta": self.Theta,
                "Sigma": self.Sigma
            })


    def E_step(self):
        K_theta0 = self.kernel_k.compute_all(self.theta0, self.common_T)
        inv_K_theta0 = np.linalg.inv(K_theta0)

        if self.common_hp_flag:
            C_Theta = self.kernel_c.compute_all(self.Theta, self.common_T)
            Psi_Theta_Sigma = C_Theta + self.Sigma * np.identity(self.n_common_T)
            inv_Psi_Theta_Sigma = np.linalg.inv(Psi_Theta_Sigma)
            inv_Psi_Theta_Sigma_dot_Y = (self.Y @ inv_Psi_Theta_Sigma).sum(axis=0)

        else:
            Psi_Theta_Sigma = []
            inv_Psi_Theta_Sigma = []
            inv_Psi_Theta_Sigma_dot_Y = 0

            for i in range(self.n_individuals):
                C_Theta_i = self.kernel_c.compute_all(self.Theta[i], self.common_T)
                Psi_Theta_Sigma_i = C_Theta_i + self.Sigma[i] * np.identity(self.n_common_T)
                inv_Psi_Theta_Sigma_i = np.linalg.inv(Psi_Theta_Sigma_i)
                inv_Psi_Theta_Sigma_dot_Y += inv_Psi_Theta_Sigma_i @ self.Y[i]
                
                Psi_Theta_Sigma.append(Psi_Theta_Sigma_i)
                inv_Psi_Theta_Sigma.append(inv_Psi_Theta_Sigma_i)

            Psi_Theta_Sigma = np.array(Psi_Theta_Sigma)
            inv_Psi_Theta_Sigma = np.array(inv_Psi_Theta_Sigma)

        self.K = np.linalg.inv(inv_K_theta0 + inv_Psi_Theta_Sigma.sum(axis=0))
        self.m0_estim = self.K @ (inv_K_theta0 @ self.m0 + inv_Psi_Theta_Sigma_dot_Y)


    def M_step(self):
        theta0 = scipy.optimize.minimize(
            fun=lambda x: -log_likelihood_theta0(x, self.kernel_k, self.common_T, self.m0, self.m0_estim, self.K),
            jac=True,
            x0=self.theta0,
            method="L-BFGS-B"
        ).x

        if self.common_hp_flag:
            Theta_Sigma0 = concatenate_Theta_Sigma_i(self.Theta, self.Sigma)
            Theta_Sigma = scipy.optimize.minimize(
                fun=lambda x: -log_likelihood_Theta_Sigma_Common_HP(x, self.kernel_c, self.common_T, self.T, self.Y, self.m0_estim, self.K), 
                jac=True,
                x0=Theta_Sigma0,
                method="L-BFGS-B"
            ).x
            Theta, Sigma = retrieve_Theta_Sigma_i(Theta_Sigma)

        else:
            Theta = np.zeros_like(self.Theta)
            Sigma = np.zeros_like(self.Sigma)
            for i in range(self.n_individuals):
                Theta_Sigma0 = concatenate_Theta_Sigma_i(self.Theta[i], self.Sigma[i])
                Theta_Sigma_i = scipy.optimize.minimize(
                    fun=lambda x: -log_likelihood_Theta_Sigma_i_Different_HP(x, self.kernel_c, self.common_T, self.T[i], self.Y[i], self.m0_estim, self.K), 
                    jac=True,
                    x0=Theta_Sigma0,
                    method="L-BFGS-B"
                ).x
                Theta_i, Sigma_i = retrieve_Theta_Sigma_i(Theta_Sigma_i)
                Theta[i] = Theta_i
                Sigma[i] = Sigma_i

        self.theta0 = theta0
        self.Theta = Theta
        self.Sigma = Sigma


    def fit(self, max_iterations: int=20, eps: float=1e-3):
        for i in range(max_iterations):
            # last_log_likelihood_monitoring = ...

            _m0_estim, _theta0, _Theta, _Sigma = self.m0_estim, self.theta0, self.Theta, self.Sigma
            self.E_step()
            self.M_step()
            self.save_history()

            ## test convergence on log likelihood instead
            #if log_likelihood_monitoring(mu0, self.m0, ...)
            if (np.linalg.norm(_m0_estim - self.m0_estim) < eps and 
                np.linalg.norm(_theta0 - self.theta0) < eps and 
                np.linalg.norm(np.array(_Theta) - np.array(self.Theta)) < eps and
                np.linalg.norm(np.array(_Sigma) - np.array(self.Sigma)) < eps):
                break
