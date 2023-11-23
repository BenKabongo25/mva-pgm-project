import numpy as np
import scipy.optimize

from typing import *
from kernels import *
from utils import *


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
    :param normalize_Y_flag:  true => normalize Y
    :param save_history_flag
    """

    def __init__(self,
                T: Union[np.ndarray, list(np.ndarray)],
                Y: Union[np.ndarray, list[np.ndarray]],
                common_T: Union[list, np.ndarray]=None,
                m0: Union[int, float, list, np.ndarray]=None, 
                theta0: Union[int, float, list, np.ndarray]=None, 
                Theta: Union[int, float, list, np.ndarray]=None, 
                Sigma: Union[int, float, list, np.ndarray]=None,
                common_hp_flag: bool=True,
                normalize_Y_flag: bool=True,
                save_history_flag: bool=False,
                kernel_k: Kernel=ExponentiatedQuadraticKernel,
                kernel_c: Kernel=ExponentiatedQuadraticKernel,
        ):
        self.common_T = self.set_common_T(common_T, T)
        self.n_common_T = len(common_T)

        self.set_TY(T, Y) # self.T, self.Y
        self.n_individuals = len(self.Y)

        self.common_hp_flag = common_hp_flag
        self.normalize_Y_flag = normalize_Y_flag
        self.save_history_flag = save_history_flag

        self.kernel_k = kernel_k
        self.kernel_c = kernel_c

        self.set_m0(m0)
        self.set_theta0(theta0)
        self.set_Theta(Theta)
        self.set_Sigma(Sigma)
        self.init_history()


    def set_common_T(self, common_T: Union[list, np.ndarray], T: Union[np.ndarray, list(np.ndarray)]=None) -> None: 
        if common_T is not None:
            self.common_T = common_T
        else:
            assert T is not None
            all_ti = []
            for t in T: all_ti.extend(list(t))
            self.common_T = np.unique(all_ti)


    def set_TY(self, T: Union[np.ndarray, list(np.ndarray)], Y: Union[np.ndarray, list[np.ndarray]]) -> None:
        if T is None:
            assert self.common_T is not None
            T = np.tile(self.common_T, (len(Y), 1))

        for (t, y) in zip(T, Y):
            assert len(t) == len(y)

        self.T = T
        self.Y = Y

        if self.normalize_Y_flag:
            self._normalize_Y() # self.Y_norm
        else:
            assert np.all(list(map(len, Y)) == self.n_common_T)
            self.Y_norm = Y

    
    def _normalize_Y(self, T: Union[np.ndarray, list(np.ndarray)], Y: Union[np.ndarray, list[np.ndarray]]) -> None:
        assert self.common_T is not None
        Y_norm = np.zeros(self.n_individuals, self.n_common_T, len(Y[0][0]))
        for idx in range(self.n_individuals):
            uniques_ti, idx_in_T = np.unique(T[idx], return_index=True)
            idx_in_common_T = np.where(np.isin(self.common_T, uniques_ti, assume_unique=True))[0]
            Y_norm[idx, idx_in_common_T] = Y[idx][idx_in_T]
        self.Y_norm = Y_norm


    def set_m0(self, m0: Union[int, float, list, np.ndarray]) -> None:
        if m0 is None:
            m0 = np.zeros(self.n_common_T)
        elif isinstance(m0, (int, float)):
            m0 = m0 * np.ones(self.n_common_T)
        elif isinstance(m0, (list, tuple)):
            m0 = np.array(m0)
        assert isinstance(m0, np.ndarray) and len(m0) == self.n_common_T
        self.m0 = m0
        self.m0_estim = None


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
            assert not self.common_hp_flag and len(Theta) == self.n_individuals
        self.Theta = Theta


    def set_Sigma(self, Sigma: Union[int, float, list, np.ndarray]) -> None:
        if Sigma is None:
            if self.common_hp_flag: Sigma = np.random.random()
            else: Sigma = np.random.random(self.n_individuals)
        else:
            assert not self.common_hp_flag and len(Sigma) == self.n_individuals
        self.Sigma = Sigma


    def init_history(self) -> None:
        self.history = []
        self.save_history()


    def save_history(self) -> None:
        if self.save_history_flag:
            self.history.append({
                "m0": self.m0_estim,
                "theta0": self.theta0,
                "Theta": self.Theta,
                "Sigma": self.Sigma
            })


    def compute_kernels(self):
        self.K_theta0 = self.kernel_k.compute_all(self.theta0, self.common_T)
        self.inv_K_theta0 = np.linalg.inv(self.K_theta0)

        if self.common_hp_flag:
            C_theta = self.kernel_c.compute_all(self.Theta, self.common_T)
            self.Psi_theta_sigma = C_theta + self.Sigma * np.identity(self.n_common_T)
            self.inv_Psi_theta_sigma = np.linalg.inv(self.Psi_theta_sigma)

        else:
            self.Psi_theta_sigma = []
            self.inv_Psi_theta_sigma = []

            for i in range(self.n_individuals):
                C_theta_i = self.kernel_c.compute_all(self.Theta[i], self.common_T)
                Psi_theta_sigma_i = C_theta_i + self.Sigma[i] * np.identity(self.n_common_T)
                inv_Psi_theta_sigma_i = np.linalg.inv(Psi_theta_sigma_i)
                self.Psi_theta_sigma.append(Psi_theta_sigma_i)
                self.inv_Psi_theta_sigma.append(inv_Psi_theta_sigma_i)
        

    def E_step(self):
        sum_inv_Psi_theta_sigma = np.sum(self.inv_Psi_theta_sigma, axis=0)
        self.K = np.linalg.inv(self.inv_K_theta0 + sum_inv_Psi_theta_sigma)
        self.m0_estim = self.K @ (self.inv_K_theta0 @ self.m0 + sum_inv_Psi_theta_sigma @ self.Y_norm)


    def M_step(self):
        # max logL == min -logL
        theta0 = scipy.optimize.minimize(
            fun=lambda x, kernel_k, common_T, m0, m0_estim, K: -log_likelihood_theta0(x, kernel_k, common_T, m0, m0_estim, K),
            jac=lambda x, kernel_k, common_T, m0, m0_estim, K: -derivate_log_likelihood_theta0(x, kernel_k, common_T, m0, m0_estim, K),
            x0=self.theta0,
            args=[self.kernel_k, self.common_T, self.m0, self.m0_estim, self.K],
            method="L-BFGS-B"
        ).x

        if self.common_hp_flag:
            # TODO
            fun = None
            jac = None
            args = None
        else:
            # TODO: 
            fun = None
            jac = None
            args = None

        Theta_Sigma0 = _flatten_Theta_Sigma(self.Theta, self.Sigma, self.common_hp_flag)
        Theta_Sigma = scipy.optimize.minimize(
            fun=fun,
            jac=jac,
            x0=Theta_Sigma0,
            args=args,
            method="L-BFGS-B"
        ).x
        Theta, Sigma = _retrieve_Theta_Sigma(Theta_Sigma, self.n_individuals, self.common_hp_flag)

        self.theta0 = theta0
        self.Theta = Theta
        self.Sigma = Sigma
        self.compute_kernels()


    def fit(self, max_iterations: int=20, eps: float=1e-3):
        for i in range(max_iterations):
            _m0_estim, _theta0, _Theta, _Sigma = self.m0_estim, self.theta0, self.Theta, self.Sigma
            self.E_step(); self.M_step()
            self.save_history()

            ## test convergence on log likelihood instead
            if (np.linalg.norm(_m0_estim - self.m0_estim) < eps and 
                np.linalg.norm(_theta0 - self.theta0) < eps and 
                np.linalg.norm(np.array(_Theta) - np.array(self.Theta)) < eps and
                np.linalg.norm(np.array(_Sigma) - np.array(self.Sigma)) < eps):
                break

