import numpy as np
from scipy.optimize._linesearch import scalar_search_wolfe2


class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
            - 'Best' -- optimal step size inferred via analytical minimization.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
            fallback_to_armijo : bool
                Whether to fall back to Armijo backtracking if Wolfe search fails.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """

    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
            self.fallback_to_armijo = kwargs.get('fallback_to_armijo', True)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        elif self._method == 'Best':
            pass
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).
        """
        if self._method == 'Constant':
            return self.c

        if self._method == 'Best':
            if hasattr(oracle, 'minimize_directional'):
                return oracle.minimize_directional(x_k, d_k)
            return None

        phi = lambda alpha: oracle.func_directional(x_k, d_k, alpha)
        derphi = lambda alpha: oracle.grad_directional(x_k, d_k, alpha)

        phi0 = phi(0.0)
        derphi0 = derphi(0.0)

        if phi0 is None or derphi0 is None:
            return None
        if not np.isfinite(phi0) or not np.isfinite(derphi0):
            return None
        if derphi0 >= 0:
            return None

        def armijo_backtracking(alpha_start):
            alpha = alpha_start
            if alpha is None or alpha <= 0 or not np.isfinite(alpha):
                alpha = getattr(self, 'alpha_0', 1.0)

            while alpha > 1e-16:
                phi_alpha = phi(alpha)
                if (
                    phi_alpha is not None
                    and np.isfinite(phi_alpha)
                    and phi_alpha <= phi0 + self.c1 * alpha * derphi0
                ):
                    return alpha
                alpha *= 0.5
            return None

        if self._method == 'Armijo':
            alpha_start = previous_alpha if previous_alpha is not None else self.alpha_0
            return armijo_backtracking(alpha_start)

        if self._method == 'Wolfe':
            alpha = scalar_search_wolfe2(
                phi=phi,
                derphi=derphi,
                phi0=phi0,
                derphi0=derphi0,
                c1=self.c1,
                c2=self.c2,
            )[0]
            if alpha is not None and np.isfinite(alpha):
                return alpha

            if getattr(self, 'fallback_to_armijo', True):
                alpha_start = previous_alpha if previous_alpha is not None else self.alpha_0
                return armijo_backtracking(alpha_start)
            return None

        return None


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        return LineSearchTool.from_dict(line_search_options)
    return LineSearchTool()
