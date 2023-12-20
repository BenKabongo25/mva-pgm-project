import numpy as np
from MAGMA import MAGMA
from gaussian_process import GaussianProcess
from kernels import Kernel, ExponentiatedQuadraticKernel
from typing import *


class Trainer_GP_MAGMA:

    def init_models(
        self,
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

        magma_model = MAGMA(
            T=T, 
            Y=Y, 
            common_T=common_T, 
            m0=m0, 
            m0_function=m0_function, 
            theta0=theta0, 
            Theta=Theta, 
            Sigma=Sigma, 
            common_hp_flag=common_hp_flag, 
            common_grid_flag=common_grid_flag,
            save_history_flag=save_history_flag, 
            scipy_optimize_display=scipy_optimize_display, 
            kernel_k=kernel_k, 
            kernel_c=kernel_c
        )

        gp_models = [] # n individuals GP models

        theta, sigma = None, None
        if magma_model.common_hp_flag:
            theta, sigma = magma_model.Theta, magma_model.Sigma

        for i in range(magma_model.n_individuals):
            if not magma_model.common_hp_flag:
                theta, sigma = magma_model.Theta[i], magma_model.Sigma[i]

            gp_i = GaussianProcess(mean_function=m0_function, kernel=kernel_c, theta=theta, sigma=sigma)
            gp_models.append(gp_i)

        self.magma_model = magma_model
        self.gp_models = gp_models


    def train_models(self, max_iterations: int=30, eps: float=1e-6):
        self.magma_model.fit(max_iterations=max_iterations, eps=eps)

        T = None
        if self.magma_model.common_grid_flag:
            T = self.magma_model.common_T

        for i in range(len(self.gp_models)):
            if not self.magma_model.common_grid_flag:
                T = self.magma_model.T[i]
            Y = self.magma_model.Y[i]
            gp_i = self.gp_models
            gp_i.fit(T, Y)
            self.gp_models[i] = gp_i


    def predict_models(self, T_p: np.ndarray, T_obs: np.ndarray, Y_obs: np.ndarray) -> tuple:
        magma_pred = self.magma_model.predict(T_p, T_obs, Y_obs)
        gps_preds = []
        for i in range(len(self.gp_models)):
            gp_pred = self.gp_models[i].predict(T_p) #TODO: what about T_obs, Y_obs ??
            gps_preds.append(gp_pred)
        return (magma_pred, gps_preds)


    def compute_MSE(self, mean_pred: np.ndarray, mean_true: np.ndarray) -> float:
        return ((mean_pred - mean_true) ** 2).mean()


    def evaluate_models(self, mean_true: np.ndarray, T_p: np.ndarray, T_obs: np.ndarray, Y_obs: np.ndarray) -> tuple:
        magma_pred, gps_preds = self.predict_models(T_p, T_obs, Y_obs)
        # TODO: implement MAGMA MSE and CI95
        magma_mse = self.compute_MSE(magma_pred[0], mean_true)
        gp_mse = np.mean([self.compute_MSE(gps_preds[i][0], mean_true) for i in range(len(self.gp_models))])
        return (magma_mse, gp_mse)


    def predict_models_all(self, T_ps: np.ndarray, T_obss: np.ndarray, Y_obss: np.ndarray) -> list:
        all_preds = []
        assert len(T_ps) == len(T_obss) == len(Y_obss)
        for i in range(len(T_ps)):
            T_p, T_obs, Y_obs = T_ps[i], T_obss[i], Y_obss[i]
            preds = self.predict_models(T_p, T_obs, Y_obs)
            all_preds.append(preds)
        return all_preds


    def evaluate_models_all(self, means_true, T_ps: np.ndarray, T_obss: np.ndarray, Y_obss: np.ndarray) -> list:
        all_res = []
        assert len(means_true) == len(T_ps) == len(T_obss) == len(Y_obss)
        for i in range(len(T_ps)):
            mean_true, T_p, T_obs, Y_obs = means_true[i], T_ps[i], T_obss[i], Y_obss[i]
            res = self.evaluate_models(mean_true, T_p, T_obs, Y_obs)
            all_res.append(res)
        return all_res 
