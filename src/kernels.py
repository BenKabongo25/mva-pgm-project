import numpy as np
from typing import *


def make_grids(X: Union[list, np.ndarray]) -> list[np.ndarray, np.ndarray]:
    X = X if isinstance(X, np.ndarray) else np.ndarray(X)
    N = len(X)
    XX = np.tile(X, (N, 1)).T
    YY = XX.T
    return XX, YY


class Kernel:
    """Abstract base class for kernels

    Kernels must inherit from this class and derive the __call__ function to implement the Kernel formula.

    Kernel classes implement their own class functions :
    - init_parameters: for initializing kernel parameters
    - compute: to compute the kernel function without creating a class object 
    - derivate_parameters: derivative function of parameters with respect to output
    """

    def __init__(self, parameters: Union[list, np.ndarray]=None):
        if parameters is not None:
            self.update_parameters(parameters)

    def update_parameters(self, parameters=None):
        raise NotImplementedError
        

class ExponentiatedQuadraticKernel(Kernel):
    """Exponentiated Quadratic Kernel
    """

    def update_parameters(self, parameters: Union[np.ndarray, list[float, float]]=None):
        self._v, self._l = ExponentiatedQuadraticKernel._check_parameters(parameters)

    def __call__(self, x: Union[int, float, np.ndarray], y: Union[int, float, np.ndarray]) -> Union[float, np.ndarray]:
        return ExponentiatedQuadraticKernel._apply(self._v, self._l, x, y)

    @classmethod
    def init_parameters(cls) -> np.ndarray:
        return np.random.random(2)

    @classmethod
    def _check_parameters(cls, parameters: Union[np.ndarray, list[float, float]]):
        assert len(parameters) == 2
        _v, _l = parameters
        #assert _v > 0 and _l > 0
        assert _l != 0
        return _v, _l

    @classmethod
    def _apply(cls, 
                _v: Union[int, float],
                _l: Union[int, float],
                x: Union[int, float, np.ndarray],
                y: Union[int, float, np.ndarray]
        ) -> Union[float, np.ndarray]:
        return (_v ** 2) * np.exp(- ((x - y) ** 2) / (2 * (_l ** 2)) )

    @classmethod
    def compute(cls, 
                parameters: Union[np.ndarray, list[float, float]], 
                x: Union[int, float, np.ndarray],
                y: Union[int, float, np.ndarray]
        ) -> Union[float, np.ndarray]:
        _v, _l = ExponentiatedQuadraticKernel._check_parameters(parameters)
        return ExponentiatedQuadraticKernel._apply(_v, _l, x, y)

    @classmethod
    def compute_all(cls,
                    parameters: Union[np.ndarray, list[float, float]],
                    X: Union[list, np.ndarray]
        ) -> np.ndarray:
        XX, YY = make_grids(X)
        return ExponentiatedQuadraticKernel.compute(parameters, XX, YY)

    @classmethod
    def derivate_parameters(cls, 
                            parameters: Union[np.ndarray, list[float, float]],
                            X: Union[list, np.ndarray]
        ) -> np.ndarray:
        _v, _l = ExponentiatedQuadraticKernel._check_parameters(parameters)
        XX, YY = make_grids(X)
        XX_diff_square = (XX - YY) ** 2
        exp_XX_diff_square = np.exp(- XX_diff_square / (2 * (_l ** 2)) )
        d_v = 2 * _v * exp_XX_diff_square
        d_l = - ((2 * _v ** 2) / (_l ** 3)) * XX_diff_square * exp_XX_diff_square
        d_v, d_l = d_v[np.newaxis, :], d_l[np.newaxis, :]
        return np.concatenate([d_v, d_l])


class GaussianKernel(Kernel):
    """Gaussian (Radial Basis Function) Kernel
    """

    def update_parameters(self, parameters=None):
        self._sigma = GaussianKernel._check_parameters(parameters)

    def __call__(self, x: Union[int, float, np.ndarray], y: Union[int, float, np.ndarray]) -> Union[float, np.ndarray]:
        return GaussianKernel._apply(self._sigma, x, y)

    @classmethod
    def init_parameters(cls) -> float:
        return np.random.random()

    @classmethod
    def _check_parameters(cls, parameters: Union[int, float]) -> Union[int, float]:
        if isinstance(parameters, (int, float)):
            _sigma = parameters
        else:
            assert len(parameters) == 1
            _sigma = parameters[0]
        assert _sigma > 0
        return _sigma

    @classmethod
    def _apply(cls, 
                _sigma: Union[int, float],
                x: Union[int, float, np.ndarray],
                y: Union[int, float, np.ndarray]
        ) -> Union[float, np.ndarray]:
        return np.exp(-((x - y) ** 2) / (2 * (_sigma ** 2)))

    @classmethod
    def compute(cls, 
                parameters: Union[int, float], 
                x: Union[int, float, np.ndarray],
                y: Union[int, float, np.ndarray]
        ) -> Union[float, np.ndarray]:
        _sigma = GaussianKernel._check_parameters(parameters)
        return GaussianKernel._apply(_sigma, x, y)

    @classmethod
    def compute_all(cls,
                    parameters: Union[int, float],
                    X: Union[list, np.ndarray]
        ) -> np.ndarray:
        XX, YY = make_grids(X)
        return GaussianKernel.compute(parameters, XX, YY)

    @classmethod
    def derivate_parameters(cls, 
                            parameters:  Union[int, float],
                            X: Union[list, np.ndarray]
        ) -> np.ndarray:
        raise NotImplementedError

