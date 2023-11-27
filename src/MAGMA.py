import numpy as np
import scipy.linalg
import scipy.optimize
from tqdm import tqdm
from typing import *

from kernels import (Kernel, 
                    ExponentiatedQuadraticKernel)
from utils import (log_likelihood_theta0, 
                   log_likelihood_Theta_Sigma_Common_HP,
                   log_likelihood_Theta_Sigma_i_Different_HP,
                   concatenate_Theta_Sigma_i,
                   retrieve_Theta_Sigma_i)


class MAGMA:
    """Multi-task Gaussian processes with common mean.

    Args:
        T (Union[np.ndarray, list[np.ndarray]]): Time points for each individual.
        Y (Union[np.ndarray, list[np.ndarray]]): Observations for each individual.
        common_T (Union[list, np.ndarray], optional): Common time points for all individuals.
        m0 (Union[int, float, list, np.ndarray], optional): Prior mean.
        m0_function (Callable): Prior mean function.
        theta0 (Union[int, float, list, np.ndarray], optional): Hyperparameters for covariance kernel k_theta0(.|.).
        Theta (Union[int, float, list, np.ndarray], optional): Hyperparameters for covariance kernel c_theta_i(.|.).
        Sigma (Union[int, float, list, np.ndarray], optional): Noise variance associated with the i-th individual.
        common_hp_flag (bool, optional): True -> common hyperparameters theta and sigma.
        save_history_flag (bool, optional): Flag to save optimization history.
        scipy_optimize_display (bool, optional): Flag to display optimization information.
        kernel_k (Kernel, optional): Kernel k_theta0 object.
        kernel_c (Kernel, optional): Kernel c_theta_i object.

    Attributes:
        common_T (np.ndarray): Common time points.
        n_common_T (int): Number of common time points.
        T (np.ndarray): Time points for each individual.
        Y (np.ndarray): Observations for each individual.
        n_individuals (int): Number of individuals.
        common_hp_flag (bool): True -> common hyperparameters theta and sigma.
        save_history_flag (bool): Flag to save optimization history.
        scipy_optimize_display (bool): Flag to display optimization information.
        kernel_k (Kernel): Kernel k_theta0 object.
        kernel_c (Kernel): Kernel c_theta_i object.
        m0 (np.ndarray): Prior mean.
        m0_function (np.ndarray): Prior mean function.
        theta0 (np.ndarray): Hyperparameters for covariance kernel k_theta0(.|.).
        Theta (np.ndarray): Hyperparameters for covariance kernel c_theta_i(.|.).
        Sigma (np.ndarray): Noise variance associated with the i-th individual.
        m0_estim (np.ndarray): Estimated prior mean.
        K (np.ndarray): Inverse covariance matrix.
        LL_theta0 (float): Log-likelihood for k_theta0.
        LL_Theta_Sigma (float): Log-likelihood for c_theta_i and Sigma.

    Methods:
        set_common_T: Set common time points.
        set_TY: Set time points and observations.
        set_m0: Set prior mean.
        set_theta0: Set hyperparameters for covariance kernel k_theta0.
        set_Theta: Set hyperparameters for covariance kernel c_theta_i.
        set_Sigma: Set noise variance associated with the i-th individual.
        init_history: Initialize optimization history.
        save_history: Save current state to the optimization history.
        E_step: Perform the E-step of the optimization algorithm.
        M_step: Perform the M-step of the optimization algorithm.
        compute_log_likelihood: Compute log-likelihood based on the current parameters.
        fit: Fit the model using the EM algorithm.
        predict: Predict the output of a new individual.
    """

    def __init__(self,
                T: Union[np.ndarray, list[np.ndarray]],
                Y: Union[np.ndarray, list[np.ndarray]],
                common_T: Union[list, np.ndarray]=None,
                m0: Union[int, float, list, np.ndarray]=None, 
                m0_function: Callable=None,
                theta0: Union[int, float, list, np.ndarray]=None, 
                Theta: Union[int, float, list, np.ndarray]=None, 
                Sigma: Union[int, float, list, np.ndarray]=None,
                common_hp_flag: bool=True,
                save_history_flag: bool=False,
                scipy_optimize_display: bool=False,
                kernel_k: Kernel=ExponentiatedQuadraticKernel,
                kernel_c: Kernel=ExponentiatedQuadraticKernel,
        ):
        # Initialize common time points and individuals' data
        self.set_common_T(common_T, T)
        self.n_common_T = len(self.common_T)

        # Initialize time points and observations
        self.set_TY(T, Y)  # self.T, self.Y
        self.n_individuals = len(self.Y)

        # Set flags and parameters
        self.common_hp_flag = common_hp_flag
        self.save_history_flag = save_history_flag
        self.scipy_optimize_display = scipy_optimize_display

        # Initialize kernels
        self.kernel_k = kernel_k
        self.kernel_c = kernel_c

        # Initialize parameters
        self.m0_estim = None
        self.K = None
        self.LL_theta0 = -np.inf
        self.LL_Theta_Sigma = -np.inf

        self.set_m0(m0, m0_function)
        self.set_theta0(theta0)
        self.set_Theta(Theta)
        self.set_Sigma(Sigma)
        self.init_history()


    def set_common_T(self, common_T: Union[list, np.ndarray], T: Union[np.ndarray, list[np.ndarray]]=None) -> None: 
        """Set common time points."""
        if common_T is not None:
            self.common_T = common_T
        else:
            assert T is not None
            all_ti = []
            for t in T: all_ti.extend(list(t))
            self.common_T = np.unique(all_ti)

    
    def set_TY(self, T: Union[np.ndarray, list[np.ndarray]], Y: Union[np.ndarray, list[np.ndarray]]) -> None:
        """Set time points and observations."""
        if T is None:
            assert self.common_T is not None
            T = np.tile(self.common_T, (len(Y), 1))

        for (t, y) in zip(T, Y):
            assert len(t) == len(y)

        self.T = T
        self.Y = Y


    def set_m0(self, m0: Union[int, float, list, np.ndarray], m0_function: Callable) -> None:
        """Set prior mean."""
        assert isinstance(m0_function, Callable)
        if m0 is None:
            m0 = m0_function(self.n_common_T)
        elif isinstance(m0, (int, float)):
            m0 = m0 * np.ones(self.n_common_T)
        elif isinstance(m0, (list, tuple)):
            m0 = np.array(m0)
        assert isinstance(m0, np.ndarray) and len(m0) == self.n_common_T
        self.m0 = m0
        self.m0_function = m0_function


    def set_theta0(self, theta0: Union[int, float, list, np.ndarray]) -> None:
        """Set hyperparameters for covariance kernel k_theta0."""
        if theta0 is None:
            theta0 = self.kernel_k.init_parameters()
        self.theta0 = theta0


    def set_Theta(self, Theta: Union[int, float, list, np.ndarray]) -> None:
        """Set hyperparameters for covariance kernel c_theta_i."""
        if Theta is None:
            if self.common_hp_flag: 
                Theta = self.kernel_c.init_parameters()
            else: Theta = np.array([self.kernel_c.init_parameters() for _ in range(self.n_individuals)])
        else:
            if not self.common_hp_flag: 
                assert len(Theta) == self.n_individuals
        self.Theta = Theta


    def set_Sigma(self, Sigma: Union[int, float, list, np.ndarray]) -> None:
        """Set noise variance associated with the i-th individual."""
        if Sigma is None:
            if self.common_hp_flag: Sigma = np.random.random()
            else: Sigma = np.random.random(self.n_individuals)
        else:
            if not self.common_hp_flag:
                assert len(Sigma) == self.n_individuals
        self.Sigma = Sigma


    def init_history(self) -> None:
        """Initialize optimization history."""
        self.history = []
        self.save_history()


    def save_history(self) -> None:
        """Save current state to the optimization history."""
        if self.save_history_flag:
            self.history.append({
                "m0": self.m0_estim,
                "K": self.K,
                "theta0": self.theta0,
                "Theta": self.Theta,
                "Sigma": self.Sigma,
                "LL_theta0": self.LL_theta0,
                "LL_Theta_Sigma": self.LL_Theta_Sigma
            })


    def E_step(self):
        """Perform the E-step of the optimization algorithm."""
        K_theta0 = self.kernel_k.compute_all(self.theta0, self.common_T)
        inv_K_theta0 = scipy.linalg.pinv(K_theta0)

        if self.common_hp_flag:
            C_Theta = self.kernel_c.compute_all(self.Theta, self.common_T)
            Psi_Theta_Sigma = C_Theta + self.Sigma * np.identity(self.n_common_T)
            inv_Psi_Theta_Sigma = scipy.linalg.pinv(Psi_Theta_Sigma)
            inv_Psi_Theta_Sigma_dot_Y = ((self.Y).dot(inv_Psi_Theta_Sigma)).sum(axis=0)

        else:
            Psi_Theta_Sigma = []
            inv_Psi_Theta_Sigma = []
            inv_Psi_Theta_Sigma_dot_Y = 0

            for i in range(self.n_individuals):
                C_Theta_i = self.kernel_c.compute_all(self.Theta[i], self.common_T)
                Psi_Theta_Sigma_i = C_Theta_i + self.Sigma[i] * np.identity(self.n_common_T)
                inv_Psi_Theta_Sigma_i = scipy.linalg.pinv(Psi_Theta_Sigma_i)
                inv_Psi_Theta_Sigma_dot_Y += inv_Psi_Theta_Sigma_i.dot(self.Y[i])
                
                Psi_Theta_Sigma.append(Psi_Theta_Sigma_i)
                inv_Psi_Theta_Sigma.append(inv_Psi_Theta_Sigma_i)

            Psi_Theta_Sigma = np.array(Psi_Theta_Sigma)
            inv_Psi_Theta_Sigma = np.array(inv_Psi_Theta_Sigma)

        self.K_theta0 = K_theta0
        self.inv_K_theta0 = inv_K_theta0
        self.Psi_Theta_Sigma = Psi_Theta_Sigma
        self.inv_Psi_Theta_Sigma = inv_Psi_Theta_Sigma

        self.K = scipy.linalg.pinv(inv_K_theta0 + inv_Psi_Theta_Sigma.sum(axis=0))
        self.m0_estim = (self.K).dot(inv_K_theta0.dot(self.m0) + inv_Psi_Theta_Sigma_dot_Y)


    def M_step(self):
        """Perform the M-step of the optimization algorithm."""
        if self.scipy_optimize_display:
            print("=" * 100)
            print("theta0")

        theta0 = scipy.optimize.minimize(
            fun=lambda x: log_likelihood_theta0(x, self.kernel_k, self.common_T, self.m0, self.m0_estim, self.K, 
                                                minimize=True, derivative=True),
            jac=True,
            x0=self.theta0,
            method="L-BFGS-B",
            options={"disp": self.scipy_optimize_display}
        ).x

        if self.common_hp_flag:
            if self.scipy_optimize_display:
                print("=" * 100)
                print("Theta & Sigma")

            Theta_Sigma0 = concatenate_Theta_Sigma_i(self.Theta, self.Sigma)
            Theta_Sigma = scipy.optimize.minimize(
                fun=lambda x: log_likelihood_Theta_Sigma_Common_HP(x, self.kernel_c, self.common_T, self.T, self.Y, 
                                                                  self.m0_estim, self.K, minimize=True, derivative=True), 
                jac=True,
                x0=Theta_Sigma0,
                method="L-BFGS-B",
                options={"disp": self.scipy_optimize_display}
            ).x
            Theta, Sigma = retrieve_Theta_Sigma_i(Theta_Sigma)

        else:
            Theta = np.zeros_like(self.Theta)
            Sigma = np.zeros_like(self.Sigma)

            for i in range(self.n_individuals):
                if self.scipy_optimize_display:
                    print("=" * 100)
                    print(f"Theta & Sigma {i}")

                Theta_Sigma0 = concatenate_Theta_Sigma_i(self.Theta[i], self.Sigma[i])
                Theta_Sigma_i = scipy.optimize.minimize(
                    fun=lambda x: log_likelihood_Theta_Sigma_i_Different_HP(x, self.kernel_c, self.common_T, 
                                                                            self.T[i], self.Y[i], self.m0_estim, self.K, 
                                                                            minimize=True, derivative=True), 
                    jac=True,
                    x0=Theta_Sigma0,
                    method="L-BFGS-B",
                    options={"disp": self.scipy_optimize_display}
                ).x
                Theta_i, Sigma_i = retrieve_Theta_Sigma_i(Theta_Sigma_i)
                Theta[i] = Theta_i
                Sigma[i] = Sigma_i

        self.theta0 = theta0
        self.Theta = Theta
        self.Sigma = Sigma


    def compute_log_likelihood(self):
        """Compute log-likelihood based on the current parameters."""
        LL_theta0 = log_likelihood_theta0(self.theta0, self.kernel_k, self.common_T, self.m0, self.m0_estim, self.K, 
                                          minimize=False, derivative=False)
        LL_Theta_Sigma = 0
        if self.common_hp_flag:
            Theta_Sigma = concatenate_Theta_Sigma_i(self.Theta, self.Sigma)
            LL_Theta_Sigma = log_likelihood_Theta_Sigma_Common_HP(Theta_Sigma, self.kernel_c, self.common_T, self.T, self.Y,
                                                                  self.m0_estim, self.K, minimize=False, derivative=False)
        else:
            for i in range(self.n_individuals):
                Theta_Sigma_i = concatenate_Theta_Sigma_i(self.Theta[i], self.Sigma[i])
                LL_Theta_Sigma += log_likelihood_Theta_Sigma_i_Different_HP(Theta_Sigma_i, self.kernel_c, self.common_T,
                                                                            self.T[i], self.Y[i], self.m0_estim, self.K,
                                                                            minimize=False, derivative=False)
        
        self.LL_theta0 = LL_theta0
        self.LL_Theta_Sigma = LL_Theta_Sigma


    def fit(self, max_iterations: int=30, eps: float=1e-6):
        """Fit the model using the EM algorithm."""
        for _ in tqdm(range(max_iterations), "MAGMA Training"):
            LL = self.LL_theta0 + self.LL_Theta_Sigma
            self.E_step()
            self.M_step()
            self.compute_log_likelihood()
            self.save_history()
            if ((self.LL_theta0 + self.LL_Theta_Sigma) - LL) ** 2 < eps:
                break

    
    def _predict_posterior_inference(self, Tp: np.ndarray=None, Yp: np.ndarray=None):
        assert Yp is not None

        use_common_T_flag = True
        if Tp is not None:
            use_common_T_flag = len(self.common_T) == len(Tp) and np.allclose(self.common_T, Tp)

        if use_common_T_flag:
            assert len(self.common_T) == len(Yp)
            m0_p = self.m0
            K_theta0_p = self.K_theta0
            inv_K_theta0_p = self.inv_K_theta0
            Psi_Theta_Sigma_p = self.Psi_Theta_Sigma
            inv_Psi_Theta_Sigma_p = self.inv_Psi_Theta_Sigma
            Y_common_p = self.Y

        else:
            assert Tp is not None
            assert len(Tp) == len(Yp)

            m0_Tp = self.m0_function(Tp)

            intersect_T = np.intersect1d(self.common_T, Tp)

            index_T = []
            # TODO

        inv_Psi_Theta_Sigma_dot_Y_common_p = 0
        if self.common_hp_flag:
            inv_Psi_Theta_Sigma_dot_Y_common_p = ((Y_common_p).dot(inv_Psi_Theta_Sigma_p)).sum(axis=0)
        else:
            for i in range(self.n_individuals):
                inv_Psi_Theta_Sigma_dot_Y_common_p += inv_Psi_Theta_Sigma_p[i].dot(Y_common_p[i])

        K_p = scipy.linalg.pinv(inv_K_theta0_p + inv_Psi_Theta_Sigma_p.sum(axis=0))
        m0_estim_p = (K_p).dot(inv_K_theta0_p.dot(self.m0) + inv_Psi_Theta_Sigma_dot_Y_common_p)

        return K_p, m0_estim_p


    def predict(self, Tp: np.ndarray, Yp: np.ndarray) -> np.ndarray:
        """Predict the output of a new individual."""
        K_p, m0_estim_p = self._predict_posterior_inference(Tp, Yp) 
        # TODO:
