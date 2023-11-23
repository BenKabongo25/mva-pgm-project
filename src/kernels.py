import numpy as np
from typing import *


def make_grids(X: Union[list, np.ndarray]) -> list[np.ndarray, np.ndarray]:
    X = X if isinstance(X, np.ndarray) else np.ndarray(X)
    N = len(X)
    XX = np.tile(X, (N, 1))
    YY = XX.T
    return XX, YY


class Kernel:
    """Abstract base class for kernels

    Kernels must inherit from this class and derive the __call__ function to implement the Kernel formula.

    Kernel classes implement their own class functions :
    - init_parameters: for initializing kernel parameters
    - compute: to compute the kernel function without creating a class object 
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

    def __call__(self, x: Union[int, float, np.ndarray], y: Union[int, float, np.ndarray]):
        return ExponentiatedQuadraticKernel._apply(self._v, self._l, x, y)

    @classmethod
    def init_parameters(cls) -> np.ndarray:
        return np.random.random(2)

    @classmethod
    def _check_parameters(cls, parameters: Union[np.ndarray, list[float, float]]):
        assert len(parameters) == 2
        _v, _l = parameters
        return _v, _l

    @classmethod
    def _apply(cls, 
                _v: Union[int, float],
                _l: Union[int, float],
                x: Union[int, float, np.ndarray],
                y: Union[int, float, np.ndarray]
        ):
        return (_v ** 2) * np.exp(- ((x - y) ** 2) / (2 * (_l ** 2)) )

    @classmethod
    def compute(cls, 
                parameters: Union[np.ndarray, list[float, float]], 
                x: Union[int, float, np.ndarray],
                y: Union[int, float, np.ndarray]
        ):
        _v, _l = ExponentiatedQuadraticKernel._check_parameters(parameters)
        return ExponentiatedQuadraticKernel._apply(_v, _l, x, y)

    @classmethod
    def compute_all(cls,
                    parameters: Union[np.ndarray, list[float, float]],
                    X: Union[list, np.ndarray]
        ):
        XX, YY = make_grids(X)
        return ExponentiatedQuadraticKernel.compute(parameters, XX, YY)


class GaussianKernel(Kernel):
    """Gaussian (Radial Basis Function) Kernel
    """

    def __init__(self, parameters: Union[list, np.ndarray] = None):
        super().__init__(parameters)

    def update_parameters(self, parameters=None):
        assert len(parameters) == 1
        self._sigma = parameters[0]
        self.parameters = parameters

    def __call__(self, x, y):
        return np.exp(-((x - y) ** 2) / (2 * (self._sigma ** 2)))

    @classmethod
    def init_parameters(cls) -> float:
        return np.random.random()

    @classmethod
    def compute(cls, sigma: Union[int, float], x, y):
        return np.exp(-((x - y) ** 2) / (2 * (sigma ** 2)))

    @classmethod
    def init_parameters(cls) -> np.ndarray:
        return np.random.random()

    @classmethod
    def _check_parameters(cls, parameters: Union[int, float]):
        assert isinstance(parameters, (int, float))
        return parameters

    @classmethod
    def _apply(cls, 
                _sigma: Union[int, float],
                x: Union[int, float, np.ndarray],
                y: Union[int, float, np.ndarray]
        ):
        return np.exp(-((x - y) ** 2) / (2 * (_sigma ** 2)))

    @classmethod
    def compute(cls, 
                parameters: Union[int, float], 
                x: Union[int, float, np.ndarray],
                y: Union[int, float, np.ndarray]
        ):
        _sigma = GaussianKernel._check_parameters(parameters)
        return GaussianKernel._apply(_sigma, x, y)

    @classmethod
    def compute_all(cls,
                    parameters: Union[int, float],
                    X: Union[list, np.ndarray]
        ):
        XX, YY = make_grids(X)
        return GaussianKernel.compute(parameters, XX, YY)


### TODO: Implement other kernels for experimentation