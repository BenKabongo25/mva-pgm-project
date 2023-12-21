import numpy as np
import pickle
import scipy.linalg
import scipy.optimize
from tqdm import tqdm
from typing import *

from kernels import (Kernel, 
                    ExponentiatedQuadraticKernel)
from utils import (compute_inv_K_theta0,
                   compute_inv_Psi_individual_i,
                   log_likelihood_theta0, 
                   log_likelihood_Theta_Sigma_Common_HP,
                   log_likelihood_Theta_Sigma_i_Different_HP,
                   log_likelihood_learn_new_parameters,
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
        common_hp_flag (bool): If true, common hyperparameters theta and sigma.
        common_grid_flag (bool): If true, common times for all individuals.
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
        compute_kernels: Calculation of data covariance matrices.
        E_step: Perform the E-step of the optimization algorithm.
        M_step: Perform the M-step of the optimization algorithm.
        compute_log_likelihood: Compute log-likelihood based on the current parameters.
        fit: Fit the model using the EM algorithm.
        predict: Predict the output of a new individual.
    """

    def __init__(self,
                T: Union[np.ndarray, list],
                Y: Union[np.ndarray, list],
                common_T: Union[list, np.ndarray]=None,
                m0: Union[int, float, list, np.ndarray]=None, 
                m0_function: Callable=None,
                theta0: Union[int, float, list, np.ndarray]=None, 
                Theta: Union[int, float, list, np.ndarray]=None, 
                Sigma: Union[int, float, list, np.ndarray]=None,
                common_hp_flag: bool=True,
                common_grid_flag: bool=True,
                save_history_flag: bool=False,
                scipy_optimize_display: bool=False,
                kernel_k: Kernel=ExponentiatedQuadraticKernel,
                kernel_c: Kernel=ExponentiatedQuadraticKernel,
        ):
        """
        Initialize the MAGMA model.

        Parameters:
            T (Union[np.ndarray, list[np.ndarray]]): Time points for each individual.
            Y (Union[np.ndarray, list[np.ndarray]]): Observations for each individual.
            common_T (Union[list, np.ndarray], optional): Common time points for all individuals.
            m0 (Union[int, float, list, np.ndarray], optional): Prior mean.
            m0_function (Callable): Prior mean function.
            theta0 (Union[int, float, list, np.ndarray], optional): Hyperparameters for covariance kernel k_theta0(.|.).
            Theta (Union[int, float, list, np.ndarray], optional): Hyperparameters for covariance kernel c_theta_i(.|.).
            Sigma (Union[int, float, list, np.ndarray], optional): Noise variance associated with the i-th individual.
            common_hp_flag (bool, optional): True -> common hyperparameters theta and sigma.
            common_grid_flag (bool, optional): True -> common times for all individuals.
            save_history_flag (bool, optional): Flag to save optimization history.
            scipy_optimize_display (bool, optional): Flag to display optimization information.
            kernel_k (Kernel, optional): Kernel k_theta0 object.
            kernel_c (Kernel, optional): Kernel c_theta_i object.
        """
        
        # Set flags
        self.common_hp_flag = common_hp_flag
        self.common_grid_flag = common_grid_flag
        self.save_history_flag = save_history_flag
        self.scipy_optimize_display = scipy_optimize_display

        # Initialize common time points and individuals' data
        self.set_common_T(common_T, T)
        self.n_common_T = len(self.common_T)

        # Initialize time points and observations
        self.set_TY(T, Y)  # self.T, self.Y
        self.n_individuals = len(self.Y)

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

    def set_common_T(self, common_T: Union[list, np.ndarray], T: Union[np.ndarray, list]=None) -> None: 
        """
        Set common time points.

        Parameters:
            common_T (Union[list, np.ndarray]): Common time points for all individuals.
            T (Union[np.ndarray, list], optional): Time points for each individual.
        """
        if common_T is not None:
            self.common_T = common_T # sorted common_T
            
        else:
            assert T is not None

            if self.common_grid_flag:
                self.common_T = T[0]
            else:
                all_ti = []
                for t in T: all_ti.extend(list(t))
                self.common_T = np.unique(all_ti)

    
    def set_TY(self, T: Union[np.ndarray, list], Y: Union[np.ndarray, list]) -> None:
        """
        Set time points and observations.

        Parameters:
            T (Union[np.ndarray, list]): Time points for each individual.
            Y (Union[np.ndarray, list]): Observations for each individual.
        """
        if T is None:
            self.common_grid_flag = True
            assert self.common_T is not None

        T_masks = None
        Y_normalized = None

        if self.common_grid_flag:
            T = None
            for i, y in enumerate(Y): 
                assert len(y) == self.n_common_T, f"Individual {i}"
            Y_normalized = Y

        else:
            assert T is not None
            assert len(T) == len(Y)
            for i, (t, y) in enumerate(zip(T, Y)):
                assert len(t) == len(y), f"Individual {i}"

            n_individuals = len(Y)
            T_masks = np.zeros((n_individuals, self.n_common_T))
            Y_normalized = np.zeros((n_individuals, self.n_common_T))

            for i in range(n_individuals):
                Yi_sorted = Y[i][np.argsort(T[i])]
                Ti_sorted = np.sort(T[i])

                mask = np.isin(self.common_T, Ti_sorted)
                Yi_normalized = np.zeros(self.n_common_T)
                Yi_normalized[np.where(mask)[0]] = Yi_sorted
                T_masks[i] = mask
                Y_normalized[i] = Yi_normalized

        self.T = T
        self.Y = Y
        self.T_masks = T_masks
        self.Y_normalized = Y_normalized


    def set_m0(self, m0: Union[int, float, list, np.ndarray], m0_function: Callable) -> None:
        """
        Set prior mean.

        Parameters:
            m0 (Union[int, float, list, np.ndarray]): Prior mean.
            m0_function (Callable): Prior mean function.
        """
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
        """
        Set hyperparameters for covariance kernel k_theta0.

        Parameters:
            theta0 (Union[int, float, list, np.ndarray]): Hyperparameters for covariance kernel k_theta0(.|.).
        """
        if theta0 is None:
            theta0 = self.kernel_k.init_parameters()
        self.theta0 = theta0


    def set_Theta(self, Theta: Union[int, float, list, np.ndarray]) -> None:
        """
        Set hyperparameters for covariance kernel c_theta_i.

        Parameters:
            Theta (Union[int, float, list, np.ndarray]): Hyperparameters for covariance kernel c_theta_i(.|.).
        """
        if Theta is None:
            if self.common_hp_flag: 
                Theta = self.kernel_c.init_parameters()
            else: Theta = np.array([self.kernel_c.init_parameters() for _ in range(self.n_individuals)])
        else:
            if not self.common_hp_flag: 
                assert len(Theta) == self.n_individuals
        self.Theta = Theta


    def set_Sigma(self, Sigma: Union[int, float, list, np.ndarray]) -> None:
        """
        Set noise variance associated with the i-th individual.

        Parameters:
            Sigma (Union[int, float, list, np.ndarray]): Noise variance associated with the i-th individual.
        """
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


    def save_model(self, model_file) -> None:
        """
        Save the model.

        Parameters:
            model_file (str): Filepath to save the model.
        """
        with open(model_file, "wb") as f:
            pickle.dump(self, f)


    def load_model(self, model_file):
        """
        Load the model.

        Parameters:
            model_file (str): Filepath to load the model.
        """
        with open(model_file, "rb") as f:
            return pickle.load(f)
            

    def get_individual(self, i: int) -> tuple:
        """
        Return input and output of i-th individual.

        Parameters:
            i (int): Index of the individual.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Input and output of the i-th individual.
        """
        assert 0 <= i < self.n_individuals
        if self.common_grid_flag:
            return self.common_T, self.Y[i], None, None
        return self.T[i], self.Y[i], self.T_masks[i], self.Y_normalized[i]


    def E_step(self):
        """Perform the E-step of the Expectation-Maximization algorithm."""
        K = None
        m0_estim = None

        _, inv_K_theta0 = compute_inv_K_theta0(self.kernel_k, self.theta0, self.common_T)

        if self.common_hp_flag and self.common_grid_flag:
            _, inv_Psi_Theta_Sigma = compute_inv_Psi_individual_i(self.kernel_c, self.Theta, self.Sigma, self.common_T, None)
            K = scipy.linalg.pinv(inv_K_theta0 + self.n_individuals * inv_Psi_Theta_Sigma) + 1e-6 * np.eye(self.n_common_T)
            m0_estim = (K).dot(inv_K_theta0.dot(self.m0) + ((self.Y).dot(inv_Psi_Theta_Sigma)).sum(axis=0))

        else:
            inv_Psi_Theta_Sigma_dot_Y = 0
            inv_Psi_Theta_Sigma = []

            if self.common_hp_flag: 
                Theta, sigma = self.Theta, self.Sigma

            for i in range(self.n_individuals):
                
                if not self.common_hp_flag: 
                    Theta, sigma = self.Theta[i], self.Sigma[i]
                
                if self.common_grid_flag: 
                    mask, Yi = None, self.Y[i]
                else: 
                    _, _, mask, Yi = self.get_individual(i)

                _, inv_Psi_Theta_Sigma_i = compute_inv_Psi_individual_i(self.kernel_c, Theta, sigma, self.common_T, mask)
                inv_Psi_Theta_Sigma_dot_Y += inv_Psi_Theta_Sigma_i.dot(Yi)
                inv_Psi_Theta_Sigma.append(inv_Psi_Theta_Sigma_i)

            inv_Psi_Theta_Sigma = np.array(inv_Psi_Theta_Sigma)
            
            K = scipy.linalg.pinv(inv_K_theta0 + inv_Psi_Theta_Sigma.sum(axis=0))
            m0_estim = (K).dot(inv_K_theta0.dot(self.m0) + inv_Psi_Theta_Sigma_dot_Y)

        self.K = K
        self.m0_estim = m0_estim


    def M_step(self):
        """Perform the M-step of the Expectation-Maximization algorithm."""
        if self.scipy_optimize_display:
            print("=" * 100)
            print("theta0")

        theta0 = scipy.optimize.minimize(
            fun=lambda x: log_likelihood_theta0(x, self.kernel_k, self.common_T, 
                                                self.m0, self.m0_estim, self.K, 
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
                fun=lambda x: log_likelihood_Theta_Sigma_Common_HP(x, self.kernel_c, self.common_T, 
                                                                  self.T_masks, self.Y_normalized, 
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
                Ti_mask = None if self.T_masks is None else self.T_masks[i]
                Theta_Sigma_i = scipy.optimize.minimize(
                    fun=lambda x: log_likelihood_Theta_Sigma_i_Different_HP(x, self.kernel_c, self.common_T, 
                                                                            Ti_mask, self.Y_normalized[i], 
                                                                            self.m0_estim, self.K, 
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
        LL_theta0 = log_likelihood_theta0(self.theta0, self.kernel_k, self.common_T, 
                                          self.m0, self.m0_estim, self.K, 
                                          minimize=False, derivative=False)
        LL_Theta_Sigma = 0
        if self.common_hp_flag:
            Theta_Sigma = concatenate_Theta_Sigma_i(self.Theta, self.Sigma)
            LL_Theta_Sigma = log_likelihood_Theta_Sigma_Common_HP(Theta_Sigma, self.kernel_c, self.common_T, 
                                                                  self.T_masks, self.Y_normalized,
                                                                  self.m0_estim, self.K, 
                                                                  minimize=False, derivative=False)
        else:
            for i in range(self.n_individuals):
                Theta_Sigma_i = concatenate_Theta_Sigma_i(self.Theta[i], self.Sigma[i])
                Ti_mask = None if self.T_masks is None else self.T_masks[i]
                LL_Theta_Sigma += log_likelihood_Theta_Sigma_i_Different_HP(Theta_Sigma_i, self.kernel_c, self.common_T,
                                                                            Ti_mask, self.Y_normalized[i], 
                                                                            self.m0_estim, self.K,
                                                                            minimize=False, derivative=False)
        
        self.LL_theta0 = LL_theta0
        self.LL_Theta_Sigma = LL_Theta_Sigma


    def fit(self, max_iterations: int=30, eps: float=1e-6):
        """
        Fit the model using the EM algorithm.

        Parameters:
            max_iterations (int, optional): Maximum number of iterations.
            eps (float, optional): Convergence threshold.
        """
        for _ in tqdm(range(max_iterations), "MAGMA Training"):
            LL = self.LL_theta0 + self.LL_Theta_Sigma
            self.E_step()
            self.M_step()
            self.compute_log_likelihood()
            self.save_history()
            if ((self.LL_theta0 + self.LL_Theta_Sigma) - LL) ** 2 < eps:
                break

    
    def _predict_posterior_inference(self, T_new: np.ndarray=None) -> list:
        """
        Predict the posterior inference for new data.

        Parameters:
            T_new (np.ndarray, optional): Time points for new data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Predicted mean and covariance.
        """       
        assert T_new is not None

        K_new = None
        m0_estim_new = None

        n_new = len(T_new)
        m0_new = self.m0_function(T_new)
        K_theta0_new, inv_K_theta0_new = compute_inv_K_theta0(self.kernel_k, self.theta0, T_new)

        inv_K_new_acc = inv_K_theta0_new
        m0_estim_new_acc = inv_K_theta0_new.dot(m0_new)

        if not np.any(np.isin(T_new, self.common_T)):
            K_new = K_theta0_new
            m0_estim_new = K_new.dot(m0_estim_new_acc)
            return K_new, m0_estim_new

        for i in range(self.n_individuals):
            T_i = self.common_T if self.common_grid_flag else self.T[i]
            Y_i = self.Y[i] 
            Theta_i = self.Theta if self.common_hp_flag else self.Theta[i]
            sigma_i = self.Sigma if self.common_hp_flag else self.Sigma[i]
            Psi_i, _ = compute_inv_Psi_individual_i(self.kernel_c, Theta_i, sigma_i, T_i, None)

            Y_new_i = np.zeros(n_new)
            Psi_new_i = np.zeros((n_new, n_new))
            
            mask_new_i = np.isin(T_new, T_i)
            mask_i_new = np.isin(T_i, T_new)
            idx_i_new = np.where(mask_i_new)[0]
            grid_mask_i_new = np.ix_(mask_i_new, mask_i_new)
            idx_new_i = np.where(mask_new_i)[0]
            grid_idx_new_i = np.ix_(idx_new_i, idx_new_i)
            
            Y_new_i[idx_new_i] = Y_i[idx_i_new]
            Psi_new_i[grid_idx_new_i] = Psi_i[grid_mask_i_new]

            inv_Psi_new_i = scipy.linalg.pinv(Psi_new_i)

            inv_K_new_acc += inv_Psi_new_i
            m0_estim_new_acc += inv_Psi_new_i.dot(Y_new_i)

        K_new = scipy.linalg.pinv(inv_K_new_acc)
        m0_estim_new = K_new.dot(m0_estim_new_acc)

        return K_new, m0_estim_new


    def _learn_new_parameters(self, T_obs: np.ndarray, Y_obs: np.ndarray, m0_estim_obs: np.ndarray) -> list:
        """
        Learn new parameters using EM.

        Parameters:
            T_obs (np.ndarray): Observed time points.
            Y_obs (np.ndarray): Observed data.
            m0_estim_obs (np.ndarray): Estimated prior mean for observed data.

        Returns:
            Tuple[np.ndarray, float]: Estimated parameters and log-likelihood.
        """
        if self.common_hp_flag:
            return self.Theta, self.Sigma
        
        i = np.random.randint(0, self.n_individuals - 1)
        Theta = self.Theta[i]
        Sigma = self.Sigma[i]
        Theta_Sigma0 = concatenate_Theta_Sigma_i(Theta, Sigma)
        Theta_Sigma = scipy.optimize.minimize(
            fun=lambda x: log_likelihood_learn_new_parameters(x, self.kernel_c, T_obs, Y_obs, m0_estim_obs,
                                                            minimize=True, derivative=True),
            jac=True,
            x0=Theta_Sigma0,
            method="L-BFGS-B",
            options={"disp": self.scipy_optimize_display}
        ).x
        Theta, Sigma = retrieve_Theta_Sigma_i(Theta_Sigma)
        return Theta, Sigma


    def predict(self, T_p: np.ndarray, T_obs: np.ndarray, Y_obs: np.ndarray) -> np.ndarray:
        """
        Predict the output of a new individual.

        Parameters:
            T_p (np.ndarray): Time points for prediction.
            T_obs (np.ndarray): Observed time points.
            Y_obs (np.ndarray): Observed data.

        Returns:
            np.ndarray: Predicted output for the new individual.
        """
        assert T_p is not None
        assert T_obs is not None and Y_obs is not None
        assert len(T_obs) == len(Y_obs)

        n_p = len(T_p)
        n_obs = len(T_obs)
        T_p_obs = np.concatenate([T_p, T_obs])

        argsort_p = np.argsort(T_p)
        argsort_p_obs = np.argsort(T_p_obs)
        T_p_obs = np.sort(T_p_obs)

        if len(T_p_obs) == len(self.common_T) and np.allclose(T_p_obs, self.common_T):
            K_p_obs, m0_estim_p_obs = self.K.copy(), self.m0_estim.copy()
        else:
            K_p_obs, m0_estim_p_obs = self._predict_posterior_inference(T_p_obs) 

        m0_estim_p_obs_argsort = np.zeros_like(m0_estim_p_obs)
        m0_estim_p_obs_argsort[argsort_p_obs] = m0_estim_p_obs
        m0_estim_p   = m0_estim_p_obs_argsort[:n_p]
        m0_estim_obs = m0_estim_p_obs_argsort[n_p:]

        Theta, Sigma = self._learn_new_parameters(T_obs, Y_obs, m0_estim_obs)
        Psi_p_obs, _ = compute_inv_Psi_individual_i(self.kernel_c, Theta, Sigma, T_p_obs, None)

        Rho_p_obs = K_p_obs + Psi_p_obs
        Rho_p_obs_argsort = np.zeros_like(Rho_p_obs)
        Rho_p_obs_argsort[np.ix_(argsort_p_obs, argsort_p_obs)] = Rho_p_obs
        Rho_p       = Rho_p_obs_argsort[:n_p, :n_p]
        Rho_obs     = Rho_p_obs_argsort[n_p:, n_p:] + 1e-6 * np.identity(n_obs)
        Rho_pobs    = Rho_p_obs_argsort[:n_p, n_p:]
        Rho_obsp    = Rho_p_obs_argsort[n_p:, :n_p]
        inv_Rho_obs = scipy.linalg.pinv(Rho_obs)
        
        mu0 = m0_estim_p + (Rho_pobs).dot(inv_Rho_obs).dot(Y_obs - m0_estim_obs)
        Rho = Rho_p - (Rho_pobs).dot(inv_Rho_obs).dot(Rho_obsp)

        return mu0, Rho
    
