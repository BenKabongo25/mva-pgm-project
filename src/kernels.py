import numpy as np
from typing import Union


def make_grids(X: Union[list, np.ndarray]) -> list:
    """
    Generate mesh grids from input points.

    Args:
        X (Union[list, np.ndarray]): Input points.

    Returns:
        list[np.ndarray, np.ndarray]: Mesh grids.
    """
    X = X if isinstance(X, np.ndarray) else np.ndarray(X)
    N = len(X)
    XX = np.tile(X, (N, 1)).T
    YY = XX.T
    return XX, YY


class Kernel:
    """
    Abstract base class for kernels.

    Kernels must inherit from this class and derive the __call__ function to implement the Kernel formula.

    Attributes:
        parameters (Union[list, np.ndarray]): Kernel parameters.

    Methods:
        update_parameters: Update kernel parameters.
    """

    def __init__(self, parameters: Union[list, np.ndarray] = None):
        if parameters is not None:
            self.update_parameters(parameters)

    def update_parameters(self, parameters=None):
        """
        Update kernel parameters.

        Args:
            parameters: New kernel parameters.

        Raises:
            NotImplementedError: This method must be implemented in the derived class.
        """
        raise NotImplementedError


class ExponentiatedQuadraticKernel(Kernel):
    """
    Exponentiated Quadratic Kernel.

    Attributes:
        _v (float): Kernel parameter v.
        _l (float): Kernel parameter l.

    Methods:
        update_parameters: Update kernel parameters.
        __call__: Evaluate the kernel function.
        init_parameters: Initialize kernel parameters.
        _check_parameters: Check and extract kernel parameters.
        _apply: Apply the kernel formula.
        compute: Compute the kernel function.
        compute_all: Compute the kernel matrix for all pairs of input points.
        derivate_parameters: Compute the derivative of parameters.
    """

    def update_parameters(self, parameters: Union[np.ndarray, list] = None):
        """
        Update kernel parameters.

        Args:
            parameters (Union[np.ndarray, list[float, float]]): New kernel parameters.
        """
        self._v, self._l = ExponentiatedQuadraticKernel._check_parameters(parameters)

    def __call__(self, x: Union[int, float, np.ndarray], y: Union[int, float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluate the kernel function.

        Args:
            x (Union[int, float, np.ndarray]): Input x.
            y (Union[int, float, np.ndarray]): Input y.

        Returns:
            Union[float, np.ndarray]: Kernel function value.
        """
        return ExponentiatedQuadraticKernel._apply(self._v, self._l, x, y)

    @classmethod
    def init_parameters(cls) -> np.ndarray:
        """
        Initialize kernel parameters.

        Returns:
            np.ndarray: Initialized kernel parameters.
        """
        return np.random.random(2)

    @classmethod
    def _check_parameters(cls, parameters: Union[np.ndarray, list]):
        """
        Check and extract kernel parameters.

        Args:
            parameters (Union[np.ndarray, list[float, float]]): Input kernel parameters.

        Returns:
            Tuple[float, float]: Extracted parameters.

        Raises:
            AssertionError: If the number of parameters is not 2 or if l is zero.
        """
        assert len(parameters) == 2
        _v, _l = parameters
        assert _l != 0
        return _v, _l

    @classmethod
    def _apply(cls,
              _v: Union[int, float],
              _l: Union[int, float],
              x: Union[int, float, np.ndarray],
              y: Union[int, float, np.ndarray]
              ) -> Union[float, np.ndarray]:
        """
        Apply the kernel formula.

        Args:
            _v (Union[int, float]): Kernel parameter v.
            _l (Union[int, float]): Kernel parameter l.
            x (Union[int, float, np.ndarray]): Input x.
            y (Union[int, float, np.ndarray]): Input y.

        Returns:
            Union[float, np.ndarray]: Result of applying the kernel formula.
        """
        return (_v ** 2) * np.exp(-((x - y) ** 2) / (2 * (_l ** 2)))

    @classmethod
    def compute(cls,
                parameters: Union[np.ndarray, list],
                x: Union[int, float, np.ndarray],
                y: Union[int, float, np.ndarray]
                ) -> Union[float, np.ndarray]:
        """
        Compute the kernel function.

        Args:
            parameters (Union[np.ndarray, list[float, float]]): Kernel parameters.
            x (Union[int, float, np.ndarray]): Input x.
            y (Union[int, float, np.ndarray]): Input y.

        Returns:
            Union[float, np.ndarray]: Kernel function value.
        """
        _v, _l = ExponentiatedQuadraticKernel._check_parameters(parameters)
        return ExponentiatedQuadraticKernel._apply(_v, _l, x, y)

    @classmethod
    def compute_all(cls,
                    parameters: Union[np.ndarray, list],
                    X: Union[list, np.ndarray]
                    ) -> np.ndarray:
        """
        Compute the kernel matrix for all pairs of input points.

        Args:
            parameters (Union[np.ndarray, list[float, float]]): Kernel parameters.
            X (Union[list, np.ndarray]): Input points.

        Returns:
            np.ndarray: Kernel matrix.
        """
        XX, YY = make_grids(X)
        return ExponentiatedQuadraticKernel.compute(parameters, XX, YY)

    @classmethod
    def derivate_parameters(cls,
                            parameters: Union[np.ndarray, list],
                            X: Union[list, np.ndarray]
                            ) -> np.ndarray:
        """
        Compute the derivative of parameters.

        Args:
            parameters (Union[np.ndarray, list[float, float]]): Kernel parameters.
            X (Union[list, np.ndarray]): Input points.

        Returns:
            np.ndarray: Derivative of parameters.
        """
        _v, _l = ExponentiatedQuadraticKernel._check_parameters(parameters)
        XX, YY = make_grids(X)
        XX_diff_square = (XX - YY) ** 2
        exp_XX_diff_square = np.exp(-XX_diff_square / (2 * (_l ** 2)))
        d_v = 2 * _v * exp_XX_diff_square
        d_l = (_v ** 2) / (_l ** 3) * XX_diff_square * exp_XX_diff_square
        d_v, d_l = d_v[np.newaxis, :], d_l[np.newaxis, :]
        return np.concatenate([d_v, d_l])


class GaussianKernel(Kernel):
    """
    Gaussian (Radial Basis Function) Kernel.

    Attributes:
        _sigma (float): Kernel parameter sigma.

    Methods:
        update_parameters: Update kernel parameters.
        __call__: Evaluate the kernel function.
        init_parameters: Initialize kernel parameters.
        _check_parameters: Check and extract kernel parameters.
        _apply: Apply the kernel formula.
        compute: Compute the kernel function.
        compute_all: Compute the kernel matrix for all pairs of input points.
        derivate_parameters: Compute the derivative of parameters.
    """

    def update_parameters(self, parameters=None):
        """
        Update kernel parameters.

        Args:
            parameters: New kernel parameters.
        """
        self._sigma = GaussianKernel._check_parameters(parameters)

    def __call__(self, x: Union[int, float, np.ndarray], y: Union[int, float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluate the kernel function.

        Args:
            x (Union[int, float, np.ndarray]): Input x.
            y (Union[int, float, np.ndarray]): Input y.

        Returns:
            Union[float, np.ndarray]: Kernel function value.
        """
        return GaussianKernel._apply(self._sigma, x, y)

    @classmethod
    def init_parameters(cls) -> float:
        """
        Initialize kernel parameters.

        Returns:
            float: Initialized kernel parameter.
        """
        return np.random.random()

    @classmethod
    def _check_parameters(cls, parameters: Union[int, float]) -> Union[int, float]:
        """
        Check and extract kernel parameters.

        Args:
            parameters (Union[int, float]): Input kernel parameters.

        Returns:
            Union[int, float]: Extracted parameter.

        Raises:
            AssertionError: If the parameter is not positive.
        """
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
        """
        Apply the kernel formula.

        Args:
            _sigma (Union[int, float]): Kernel parameter sigma.
            x (Union[int, float, np.ndarray]): Input x.
            y (Union[int, float, np.ndarray]): Input y.

        Returns:
            Union[float, np.ndarray]: Result of applying the kernel formula.
        """
        return np.exp(-((x - y) ** 2) / (2 * (_sigma ** 2)))

    @classmethod
    def compute(cls,
                parameters: Union[int, float],
                x: Union[int, float, np.ndarray],
                y: Union[int, float, np.ndarray]
                ) -> Union[float, np.ndarray]:
        """
        Compute the kernel function.

        Args:
            parameters (Union[int, float]): Kernel parameter.
            x (Union[int, float, np.ndarray]): Input x.
            y (Union[int, float, np.ndarray]): Input y.

        Returns:
            Union[float, np.ndarray]: Kernel function value.
        """
        _sigma = GaussianKernel._check_parameters(parameters)
        return GaussianKernel._apply(_sigma, x, y)

    @classmethod
    def compute_all(cls,
                    parameters: Union[int, float],
                    X: Union[list, np.ndarray]
                    ) -> np.ndarray:
        """
        Compute the kernel matrix for all pairs of input points.

        Args:
            parameters (Union[int, float]): Kernel parameter.
            X (Union[list, np.ndarray]): Input points.

        Returns:
            np.ndarray: Kernel matrix.
        """
        XX, YY = make_grids(X)
        return GaussianKernel.compute(parameters, XX, YY)

    @classmethod
    def derivate_parameters(cls,
                            parameters: Union[int, float],
                            X: Union[list, np.ndarray]
                            ) -> np.ndarray:
        """
        Compute the derivative of parameters.

        Args:
            parameters (Union[int, float]): Kernel parameter.
            X (Union[list, np.ndarray]): Input points.

        Returns:
            np.ndarray: Derivative of parameters.
        """
        raise NotImplementedError
