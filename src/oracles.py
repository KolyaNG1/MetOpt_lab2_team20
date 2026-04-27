from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.special import expit


EXP_CLIP = 50.0


def _as_1d_array(x):
    """Return x as a contiguous 1D float64 numpy array."""
    return np.asarray(x, dtype=np.float64).reshape(-1)


class BaseSmoothOracle:
    """Base class for smooth optimization oracles."""

    def __init__(self):
        self.func_calls = 0
        self.grad_calls = 0
        self.hess_vec_calls = 0

    def func(self, x: np.ndarray) -> float:
        raise NotImplementedError

    def grad(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def hess(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def func_directional(self, x: np.ndarray, d: np.ndarray, alpha: float) -> float:
        x = _as_1d_array(x)
        d = _as_1d_array(d)
        return self.func(x + alpha * d)

    def grad_directional(self, x: np.ndarray, d: np.ndarray, alpha: float) -> float:
        x = _as_1d_array(x)
        d = _as_1d_array(d)
        return float(self.grad(x + alpha * d).dot(d))

    def hess_vec(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        self.hess_vec_calls += 1
        x = _as_1d_array(x)
        v = _as_1d_array(v)
        return _as_1d_array(self.hess(x).dot(v))


class NonConvexOracle(BaseSmoothOracle):
    r"""
    Oracle for the Beale function

        f(x, y) = (1.5   - x + x y    )^2
                + (2.25  - x + x y^2  )^2
                + (2.625 - x + x y^3  )^2.
    """

    def _residuals(self, point: np.ndarray):
        point = _as_1d_array(point)
        if point.size != 2:
            raise ValueError("Beale oracle expects a 2D point x = [x, y].")
        x, y = point
        r1 = 1.5 - x + x * y
        r2 = 2.25 - x + x * y**2
        r3 = 2.625 - x + x * y**3
        return x, y, r1, r2, r3

    def func(self, point: np.ndarray) -> float:
        self.func_calls += 1
        _, _, r1, r2, r3 = self._residuals(point)
        return float(r1**2 + r2**2 + r3**2)

    def grad(self, point: np.ndarray) -> np.ndarray:
        self.grad_calls += 1
        x, y, r1, r2, r3 = self._residuals(point)

        gx = 2.0 * (
            r1 * (y - 1.0)
            + r2 * (y**2 - 1.0)
            + r3 * (y**3 - 1.0)
        )
        gy = 2.0 * (
            x * r1
            + 2.0 * x * y * r2
            + 3.0 * x * y**2 * r3
        )
        return np.array([gx, gy], dtype=np.float64)

    def hess(self, point: np.ndarray) -> np.ndarray:
        x, y, r1, r2, r3 = self._residuals(point)

        h_xx = 2.0 * (
            (y - 1.0) ** 2
            + (y**2 - 1.0) ** 2
            + (y**3 - 1.0) ** 2
        )

        h_xy = 2.0 * (
            x * (y - 1.0) + r1
            + 2.0 * x * y * (y**2 - 1.0) + 2.0 * y * r2
            + 3.0 * x * y**2 * (y**3 - 1.0) + 3.0 * y**2 * r3
        )

        h_yy = 2.0 * (
            x**2
            + 2.0 * x * r2
            + 4.0 * x**2 * y**2
            + 6.0 * x * y * r3
            + 9.0 * x**2 * y**4
        )

        return np.array([[h_xx, h_xy], [h_xy, h_yy]], dtype=np.float64)


BealeOracle = NonConvexOracle


class PoissonL2Oracle(BaseSmoothOracle):
    r"""Oracle for Poisson regression with L2 regularization."""

    def __init__(self, A, b, regcoef: float, exp_clip: float = EXP_CLIP):
        super().__init__()
        self.A = A
        self.b = _as_1d_array(b)
        self.regcoef = float(regcoef)
        self.exp_clip = float(exp_clip)

        if self.A.shape[0] != self.b.size:
            raise ValueError("A and b have incompatible shapes.")

        self.m = self.A.shape[0]
        self.n = self.A.shape[1]

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        return _as_1d_array(self.A.dot(x))

    def _rmatvec(self, v: np.ndarray) -> np.ndarray:
        return _as_1d_array(self.A.T.dot(v))

    def _safe_exp(self, z: np.ndarray) -> np.ndarray:
        return np.exp(np.minimum(z, self.exp_clip))

    def _weighted_gram(self, weights: np.ndarray) -> np.ndarray:
        if sp.issparse(self.A):
            scaled = self.A.multiply(weights[:, None])
            gram = self.A.T @ scaled
            return gram.toarray()
        return self.A.T @ (weights[:, None] * self.A)

    def func(self, x: np.ndarray) -> float:
        self.func_calls += 1
        x = _as_1d_array(x)
        z = self._matvec(x)
        exp_z = self._safe_exp(z)
        value = (exp_z.sum() - self.b.dot(z)) / self.m + 0.5 * self.regcoef * x.dot(x)
        return float(value)

    def grad(self, x: np.ndarray) -> np.ndarray:
        self.grad_calls += 1
        x = _as_1d_array(x)
        z = self._matvec(x)
        exp_z = self._safe_exp(z)
        return self._rmatvec(exp_z - self.b) / self.m + self.regcoef * x

    def hess(self, x: np.ndarray) -> np.ndarray:
        x = _as_1d_array(x)
        z = self._matvec(x)
        exp_z = self._safe_exp(z)
        hess = self._weighted_gram(exp_z) / self.m
        hess.flat[:: self.n + 1] += self.regcoef
        return np.asarray(hess, dtype=np.float64)

    def hess_vec(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        self.hess_vec_calls += 1
        x = _as_1d_array(x)
        v = _as_1d_array(v)
        z = self._matvec(x)
        exp_z = self._safe_exp(z)
        Av = self._matvec(v)
        return self._rmatvec(exp_z * Av) / self.m + self.regcoef * v


PoissonRegressionOracle = PoissonL2Oracle


class LogisticL2Oracle(BaseSmoothOracle):
    r"""Oracle for logistic regression with L2 regularization."""

    def __init__(self, A, b, regcoef: float):
        super().__init__()
        self.A = A
        self.b = _as_1d_array(b)
        self.regcoef = float(regcoef)

        if self.A.shape[0] != self.b.size:
            raise ValueError("A and b have incompatible shapes.")

        unique = np.unique(self.b)
        if not np.all(np.isin(unique, [-1.0, 1.0])):
            raise ValueError("For logistic regression labels must belong to {-1, 1}.")

        self.m = self.A.shape[0]
        self.n = self.A.shape[1]

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        return _as_1d_array(self.A.dot(x))

    def _rmatvec(self, v: np.ndarray) -> np.ndarray:
        return _as_1d_array(self.A.T.dot(v))

    def _weighted_gram(self, weights: np.ndarray) -> np.ndarray:
        if sp.issparse(self.A):
            scaled = self.A.multiply(weights[:, None])
            gram = self.A.T @ scaled
            return gram.toarray()
        return self.A.T @ (weights[:, None] * self.A)

    def func(self, x: np.ndarray) -> float:
        self.func_calls += 1
        x = _as_1d_array(x)
        margins = self.b * self._matvec(x)
        value = np.logaddexp(0.0, -margins).mean() + 0.5 * self.regcoef * x.dot(x)
        return float(value)

    def grad(self, x: np.ndarray) -> np.ndarray:
        self.grad_calls += 1
        x = _as_1d_array(x)
        margins = self.b * self._matvec(x)
        probs = expit(-margins)
        return -self._rmatvec(self.b * probs) / self.m + self.regcoef * x

    def hess(self, x: np.ndarray) -> np.ndarray:
        x = _as_1d_array(x)
        margins = self.b * self._matvec(x)
        probs = expit(-margins)
        weights = probs * (1.0 - probs)
        hess = self._weighted_gram(weights) / self.m
        hess.flat[:: self.n + 1] += self.regcoef
        return np.asarray(hess, dtype=np.float64)

    def hess_vec(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        self.hess_vec_calls += 1
        x = _as_1d_array(x)
        v = _as_1d_array(v)
        margins = self.b * self._matvec(x)
        probs = expit(-margins)
        weights = probs * (1.0 - probs)
        Av = self._matvec(v)
        return self._rmatvec(weights * Av) / self.m + self.regcoef * v


LogisticRegressionOracle = LogisticL2Oracle


class QuadraticOracle(BaseSmoothOracle):
    def __init__(self, A, b):
        super().__init__()
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)

    def func(self, x):
        self.func_calls += 1
        return 0.5 * x.dot(self.A.dot(x)) - self.b.dot(x)

    def grad(self, x):
        self.grad_calls += 1
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A


class DiagonalQuadraticOracle(BaseSmoothOracle):
    def __init__(self, diag):
        super().__init__()
        self.diag = np.array(diag, dtype=float)

    def func(self, x):
        self.func_calls += 1
        return 0.5 * np.sum(self.diag * (x**2))

    def grad(self, x):
        self.grad_calls += 1
        return self.diag * x

    def hess(self, x):
        return np.diag(self.diag)



def grad_finite_diff(func, x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = _as_1d_array(x)
    n = x.size
    eye = np.eye(n, dtype=np.float64)
    points = x[None, :] + eps * eye
    f0 = float(func(x))
    values = np.apply_along_axis(func, 1, points)
    return (values - f0) / eps



def hess_finite_diff(func, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    x = _as_1d_array(x)
    n = x.size
    eye = np.eye(n, dtype=np.float64)

    single_points = x[None, :] + eps * eye
    f_single = np.apply_along_axis(func, 1, single_points)
    f0 = float(func(x))

    pairwise_shifts = eye[:, None, :] + eye[None, :, :]
    pairwise_points = x[None, None, :] + eps * pairwise_shifts
    pairwise_values = np.apply_along_axis(func, 2, pairwise_points)

    hess = (pairwise_values - f_single[:, None] - f_single[None, :] + f0) / (eps ** 2)
    return 0.5 * (hess + hess.T)



def hess_vec_finite_diff(func, x, v, eps=None):
    """
    Returns approximation of the matrix product 'Hessian times vector'
    using finite differences.
    """
    x = _as_1d_array(x)
    v = _as_1d_array(v)
    n = x.size
    if v.size != n:
        raise ValueError('x and v must have the same dimension.')

    if eps is None:
        eps = np.cbrt(np.finfo(np.float64).eps)

    hv = np.empty(n, dtype=np.float64)
    f_x = float(func(x))
    f_x_eps_v = float(func(x + eps * v))

    for i in range(n):
        e_i = np.zeros(n, dtype=np.float64)
        e_i[i] = 1.0
        hv[i] = (
            func(x + eps * v + eps * e_i)
            - f_x_eps_v
            - func(x + eps * e_i)
            + f_x
        ) / (eps ** 2)

    return hv


CLASS_MODEL_NAMEL2Oracle = LogisticL2Oracle

__all__ = [
    'BaseSmoothOracle',
    'NonConvexOracle',
    'BealeOracle',
    'PoissonL2Oracle',
    'PoissonRegressionOracle',
    'LogisticL2Oracle',
    'LogisticRegressionOracle',
    'QuadraticOracle',
    'DiagonalQuadraticOracle',
    'grad_finite_diff',
    'hess_finite_diff',
    'hess_vec_finite_diff',
]
