import time
from collections import defaultdict, deque

import numpy as np
import scipy
from numpy.linalg import LinAlgError

from utils import get_line_search_tool


EPS = 1e-12


def _is_finite_scalar(x):
    return x is not None and np.isfinite(x)



def _is_finite_array(x):
    return x is not None and np.all(np.isfinite(x))



def _push_history_linear(history, start_time, x, residual):
    if history is None:
        return
    history['time'].append(time.perf_counter() - start_time)
    history['residual_norm'].append(float(np.linalg.norm(residual)))
    if x.size <= 2:
        history['x'].append(np.copy(x))



def _push_history_smooth(history, start_time, x, f_val, grad):
    if history is None:
        return
    history['time'].append(time.perf_counter() - start_time)
    history['func'].append(float(f_val))
    history['grad_norm'].append(float(np.linalg.norm(grad)))
    if x.size <= 2:
        history['x'].append(np.copy(x))


def _apply_callback(history, callback, x_k, **kwargs):
    if history is None or callback is None:
        return
    result = callback(x_k, kwargs)
    if not result:
        return
    for key, val in result.items():
        history[key].append(val)


def _init_micro_timing(history, trace_micro):
    if history is None or not trace_micro:
        return
    history['time_oracle'].append(0.0)
    history['time_linesearch'].append(0.0)
    history['time_linalg'].append(0.0)


def _append_micro_timing(history, trace_micro, t_oracle, t_ls, t_linalg):
    if history is None or not trace_micro:
        return
    history['time_oracle'].append(float(t_oracle))
    history['time_linesearch'].append(float(t_ls))
    history['time_linalg'].append(float(t_linalg))



def linear_conjugate_gradients(matvec, b, x_0, tolerance=1e-4, max_iter=None, trace=False, display=False,
                               callback=None):
    """
    Solves system Ax=b using Conjugate Gradients method.
    """
    history = defaultdict(list) if (trace or callback is not None) else None
    x_k = np.copy(x_0)
    b = np.asarray(b, dtype=np.float64)

    if max_iter is None:
        max_iter = b.size

    start_time = time.perf_counter()

    g_k = matvec(x_k) - b
    if not _is_finite_array(g_k):
        return x_k, 'computational_error', history

    _push_history_linear(history, start_time, x_k, g_k)
    _apply_callback(history, callback, x_k, iteration=0)

    b_norm = np.linalg.norm(b)
    rhs = tolerance * b_norm
    if np.linalg.norm(g_k) <= rhs:
        return x_k, 'success', history

    d_k = -g_k
    Ad_k = matvec(d_k)
    if not _is_finite_array(Ad_k):
        return x_k, 'computational_error', history

    for k in range(max_iter):
        gg = float(g_k.dot(g_k))
        denom = float(d_k.dot(Ad_k))

        if not np.isfinite(denom) or denom <= 0:
            return x_k, 'computational_error', history

        alpha_k = gg / denom
        x_k = x_k + alpha_k * d_k
        g_next = g_k + alpha_k * Ad_k

        if not _is_finite_array(x_k) or not _is_finite_array(g_next):
            return x_k, 'computational_error', history

        _push_history_linear(history, start_time, x_k, g_next)
        _apply_callback(history, callback, x_k, iteration=k + 1)

        residual_norm = np.linalg.norm(g_next)
        if display:
            print('iter = {}, ||r|| = {:.6e}'.format(k + 1, residual_norm))

        if residual_norm <= rhs:
            return x_k, 'success', history

        beta_k = float(g_next.dot(g_next)) / gg
        d_k = -g_next + beta_k * d_k
        Ad_k = matvec(d_k)
        if not _is_finite_array(Ad_k):
            return x_k, 'computational_error', history
        g_k = g_next

    return x_k, 'iterations_exceeded', history



def nonlinear_conjugate_gradients(oracle, x_0, tolerance=1e-4, max_iter=500,
                                  line_search_options=None, display=False, trace=False, callback=None):
    """
    Nonlinear Conjugate Gradients method for optimization.
    """
    history = defaultdict(list) if (trace or callback is not None) else None
    trace_micro = bool(trace)
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    start_time = time.perf_counter()

    f_k = oracle.func(x_k)
    g_k = oracle.grad(x_k)
    if not _is_finite_scalar(f_k) or not _is_finite_array(g_k):
        return x_k, 'computational_error', history

    _push_history_smooth(history, start_time, x_k, f_k, g_k)
    _init_micro_timing(history, trace_micro)
    _apply_callback(history, callback, x_k, iteration=0)
    grad0_norm_sq = float(g_k.dot(g_k))
    if grad0_norm_sq <= tolerance * grad0_norm_sq:
        return x_k, 'success', history

    d_k = -g_k
    alpha_k = None

    for k in range(max_iter):
        t_lin_0 = time.perf_counter()
        if g_k.dot(d_k) >= 0:
            d_k = -g_k
        t_lin_1 = time.perf_counter()

        previous_alpha = alpha_k if alpha_k is not None else None
        t_ls_0 = time.perf_counter()
        alpha_k = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=previous_alpha)
        t_ls_1 = time.perf_counter()
        if alpha_k is None or not np.isfinite(alpha_k):
            return x_k, 'computational_error', history

        t_mid = time.perf_counter()
        x_next = x_k + alpha_k * d_k
        t_ora_0 = time.perf_counter()
        f_next = oracle.func(x_next)
        g_next = oracle.grad(x_next)
        t_ora_1 = time.perf_counter()

        if not _is_finite_scalar(f_next) or not _is_finite_array(g_next):
            return x_k, 'computational_error', history

        time_linalg = (t_lin_1 - t_lin_0) + (t_ora_0 - t_mid)
        time_linesearch = t_ls_1 - t_ls_0
        time_oracle = t_ora_1 - t_ora_0

        _push_history_smooth(history, start_time, x_next, f_next, g_next)
        _append_micro_timing(history, trace_micro, time_oracle, time_linesearch, time_linalg)
        _apply_callback(history, callback, x_next, iteration=k + 1)

        grad_norm_sq = float(g_next.dot(g_next))
        if display:
            print('iter = {}, f = {:.6e}, ||g|| = {:.6e}, alpha = {:.6e}'.format(
                k + 1, f_next, np.sqrt(grad_norm_sq), alpha_k
            ))

        if grad_norm_sq <= tolerance * grad0_norm_sq:
            return x_next, 'success', history

        y_k = g_next - g_k
        beta_k = float(g_next.dot(y_k)) / max(float(g_k.dot(g_k)), EPS)
        if beta_k < 0:
            beta_k = 0.0

        d_next = -g_next + beta_k * d_k
        if g_next.dot(d_next) >= 0:
            d_next = -g_next

        x_k, f_k, g_k, d_k = x_next, f_next, g_next, d_next

    return x_k, 'iterations_exceeded', history



def _two_loop_recursion(grad, s_history, y_history):
    q = np.copy(grad)
    if len(s_history) == 0:
        return q

    alpha_list = []
    rho_list = []
    pairs = list(zip(s_history, y_history))

    for s_k, y_k in reversed(pairs):
        rho_k = 1.0 / max(float(y_k.dot(s_k)), EPS)
        alpha_k = rho_k * float(s_k.dot(q))
        q = q - alpha_k * y_k
        alpha_list.append(alpha_k)
        rho_list.append(rho_k)

    s_last, y_last = pairs[-1]
    gamma_k = float(y_last.dot(s_last)) / max(float(y_last.dot(y_last)), EPS)
    r = gamma_k * q

    for (s_k, y_k), alpha_k, rho_k in zip(pairs, reversed(alpha_list), reversed(rho_list)):
        beta_k = rho_k * float(y_k.dot(r))
        r = r + s_k * (alpha_k - beta_k)

    return r



def lbfgs(oracle, x_0, tolerance=1e-4, max_iter=500, memory_size=10,
          line_search_options=None, display=False, trace=False, callback=None):
    """
    Limited-memory BFGS method for optimization.
    """
    history = defaultdict(list) if (trace or callback is not None) else None
    trace_micro = bool(trace)
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    start_time = time.perf_counter()

    f_k = oracle.func(x_k)
    g_k = oracle.grad(x_k)
    if not _is_finite_scalar(f_k) or not _is_finite_array(g_k):
        return x_k, 'computational_error', history

    _push_history_smooth(history, start_time, x_k, f_k, g_k)
    _init_micro_timing(history, trace_micro)
    _apply_callback(history, callback, x_k, iteration=0)
    grad0_norm_sq = float(g_k.dot(g_k))
    if grad0_norm_sq <= tolerance * grad0_norm_sq:
        return x_k, 'success', history

    s_history = deque(maxlen=max(memory_size, 1))
    y_history = deque(maxlen=max(memory_size, 1))
    alpha_k = None

    for k in range(max_iter):
        t_lin_0 = time.perf_counter()
        if memory_size == 0:
            d_k = -g_k
        else:
            d_k = -_two_loop_recursion(g_k, s_history, y_history)
            if g_k.dot(d_k) >= 0:
                d_k = -g_k
        t_lin_1 = time.perf_counter()

        previous_alpha = alpha_k if alpha_k is not None else None
        t_ls_0 = time.perf_counter()
        alpha_k = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=previous_alpha)
        t_ls_1 = time.perf_counter()
        if alpha_k is None or not np.isfinite(alpha_k):
            return x_k, 'computational_error', history

        t_mid = time.perf_counter()
        x_next = x_k + alpha_k * d_k
        t_ora_0 = time.perf_counter()
        f_next = oracle.func(x_next)
        g_next = oracle.grad(x_next)
        t_ora_1 = time.perf_counter()
        if not _is_finite_scalar(f_next) or not _is_finite_array(g_next):
            return x_k, 'computational_error', history

        t_sy_0 = time.perf_counter()
        s_k = x_next - x_k
        y_k = g_next - g_k
        sy = float(s_k.dot(y_k))

        if memory_size > 0 and sy > 1e-12:
            s_history.append(np.copy(s_k))
            y_history.append(np.copy(y_k))
        t_sy_1 = time.perf_counter()

        time_linalg = (t_lin_1 - t_lin_0) + (t_ora_0 - t_mid) + (t_sy_1 - t_sy_0)
        time_linesearch = t_ls_1 - t_ls_0
        time_oracle = t_ora_1 - t_ora_0

        _push_history_smooth(history, start_time, x_next, f_next, g_next)
        _append_micro_timing(history, trace_micro, time_oracle, time_linesearch, time_linalg)
        _apply_callback(history, callback, x_next, iteration=k + 1)

        grad_norm_sq = float(g_next.dot(g_next))
        if display:
            print('iter = {}, f = {:.6e}, ||g|| = {:.6e}, alpha = {:.6e}'.format(
                k + 1, f_next, np.sqrt(grad_norm_sq), alpha_k
            ))

        if grad_norm_sq <= tolerance * grad0_norm_sq:
            return x_next, 'success', history

        x_k, f_k, g_k = x_next, f_next, g_next

    return x_k, 'iterations_exceeded', history



def cautious_lbfgs(oracle, x_0, tolerance=1e-4, max_iter=500, memory_size=10,
                   cautious_eps=1e-4, cautious_alpha=1.0,
                   line_search_options=None, display=False, trace=False, callback=None):
    """
    Cautious L-BFGS for track 4.

    The step is found by Armijo backtracking, and the pair (s_k, y_k) is added
    to memory only if the cautious curvature condition holds:

        <s_k, y_k> / ||s_k||^2 > cautious_eps * ||grad_k||^cautious_alpha.
    """
    if line_search_options is None:
        line_search_options = {'method': 'Armijo', 'c1': 1e-4, 'alpha_0': 1.0}

    history = defaultdict(list) if (trace or callback is not None) else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    start_time = time.perf_counter()

    f_k = oracle.func(x_k)
    g_k = oracle.grad(x_k)
    if not _is_finite_scalar(f_k) or not _is_finite_array(g_k):
        return x_k, 'computational_error', history

    _push_history_smooth(history, start_time, x_k, f_k, g_k)
    grad0_norm_sq = float(g_k.dot(g_k))
    if history is not None:
        history['skipped_updates'].append(0)
        history['accepted_updates'].append(0)
    _apply_callback(history, callback, x_k, iteration=0)
    if grad0_norm_sq <= tolerance * grad0_norm_sq:
        return x_k, 'success', history

    s_history = deque(maxlen=max(memory_size, 1))
    y_history = deque(maxlen=max(memory_size, 1))
    alpha_k = None
    skipped = 0
    accepted = 0

    for k in range(max_iter):
        if memory_size == 0:
            d_k = -g_k
        else:
            d_k = -_two_loop_recursion(g_k, s_history, y_history)
            if g_k.dot(d_k) >= 0:
                d_k = -g_k

        previous_alpha = alpha_k if alpha_k is not None else None
        alpha_k = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=previous_alpha)
        if alpha_k is None or not np.isfinite(alpha_k):
            return x_k, 'computational_error', history

        x_next = x_k + alpha_k * d_k
        f_next = oracle.func(x_next)
        g_next = oracle.grad(x_next)
        if not _is_finite_scalar(f_next) or not _is_finite_array(g_next):
            return x_k, 'computational_error', history

        s_k = x_next - x_k
        y_k = g_next - g_k
        s_norm_sq = float(s_k.dot(s_k))
        lhs = float(s_k.dot(y_k)) / max(s_norm_sq, EPS)
        rhs = cautious_eps * (np.linalg.norm(g_k) ** cautious_alpha)

        if memory_size > 0 and lhs > rhs and float(s_k.dot(y_k)) > 1e-12:
            s_history.append(np.copy(s_k))
            y_history.append(np.copy(y_k))
            accepted += 1
        else:
            skipped += 1

        _push_history_smooth(history, start_time, x_next, f_next, g_next)
        if history is not None:
            history['skipped_updates'].append(skipped)
            history['accepted_updates'].append(accepted)
        _apply_callback(history, callback, x_next, iteration=k + 1)

        grad_norm_sq = float(g_next.dot(g_next))
        if display:
            print('iter = {}, f = {:.6e}, ||g|| = {:.6e}, alpha = {:.6e}, skipped = {}'.format(
                k + 1, f_next, np.sqrt(grad_norm_sq), alpha_k, skipped
            ))

        if grad_norm_sq <= tolerance * grad0_norm_sq:
            return x_next, 'success', history

        x_k, f_k, g_k = x_next, f_next, g_next

    return x_k, 'iterations_exceeded', history



def hessian_free_newton(oracle, x_0, tolerance=1e-4, max_iter=500,
                        line_search_options=None, display=False, trace=False, callback=None):
    """
    Hessian-Free Newton method for optimization.
    """
    history = defaultdict(list) if (trace or callback is not None) else None
    trace_micro = bool(trace)
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    start_time = time.perf_counter()

    f_k = oracle.func(x_k)
    g_k = oracle.grad(x_k)
    if not _is_finite_scalar(f_k) or not _is_finite_array(g_k):
        return x_k, 'computational_error', history

    _push_history_smooth(history, start_time, x_k, f_k, g_k)
    _init_micro_timing(history, trace_micro)
    _apply_callback(history, callback, x_k, iteration=0)
    if history is not None:
        history['inner_iterations'].append(0)
    grad0_norm_sq = float(g_k.dot(g_k))
    if grad0_norm_sq <= tolerance * grad0_norm_sq:
        return x_k, 'success', history

    for k in range(max_iter):
        time_oracle = 0.0
        time_linalg = 0.0
        time_linesearch = 0.0

        t_prep = time.perf_counter()
        grad_norm = np.linalg.norm(g_k)
        eta_k = min(0.5, np.sqrt(grad_norm))
        d_start = -g_k
        time_linalg += time.perf_counter() - t_prep

        total_inner_iters = 0
        d_k = None

        while True:
            hess_time = 0.0

            def hess_matvec(v):
                nonlocal hess_time
                t_h = time.perf_counter()
                out = oracle.hess_vec(x_k, v)
                hess_time += time.perf_counter() - t_h
                return out

            max_cg_iter = x_k.size
            t_cg0 = time.perf_counter()
            d_k, msg, cg_history = linear_conjugate_gradients(
                hess_matvec,
                -g_k,
                d_start,
                tolerance=eta_k,
                max_iter=max_cg_iter,
                trace=trace,
                display=False,
            )
            t_cg1 = time.perf_counter()
            cg_wall = t_cg1 - t_cg0
            time_oracle += hess_time
            time_linalg += max(cg_wall - hess_time, 0.0)

            if cg_history is not None:
                total_inner_iters += max(len(cg_history['residual_norm']) - 1, 0)
            else:
                total_inner_iters += max_cg_iter

            if _is_finite_array(d_k) and float(g_k.dot(d_k)) < 0:
                break

            t_adj = time.perf_counter()
            eta_k *= 0.1
            d_start = d_k
            if eta_k < 1e-12:
                d_k = -g_k
                time_linalg += time.perf_counter() - t_adj
                break
            time_linalg += time.perf_counter() - t_adj

        t_ls0 = time.perf_counter()
        alpha_k = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=1.0)
        time_linesearch += time.perf_counter() - t_ls0
        if alpha_k is None or not np.isfinite(alpha_k):
            return x_k, 'computational_error', history

        t_step = time.perf_counter()
        x_next = x_k + alpha_k * d_k
        time_linalg += time.perf_counter() - t_step

        t_ora = time.perf_counter()
        f_next = oracle.func(x_next)
        g_next = oracle.grad(x_next)
        time_oracle += time.perf_counter() - t_ora
        if not _is_finite_scalar(f_next) or not _is_finite_array(g_next):
            return x_k, 'computational_error', history

        _push_history_smooth(history, start_time, x_next, f_next, g_next)
        _append_micro_timing(history, trace_micro, time_oracle, time_linesearch, time_linalg)
        _apply_callback(history, callback, x_next, iteration=k + 1)
        if history is not None:
            history['inner_iterations'].append(total_inner_iters)

        grad_norm_sq = float(g_next.dot(g_next))
        if display:
            print('iter = {}, f = {:.6e}, ||g|| = {:.6e}, alpha = {:.6e}, inner_cg = {}'.format(
                k + 1, f_next, np.sqrt(grad_norm_sq), alpha_k, total_inner_iters
            ))

        if grad_norm_sq <= tolerance * grad0_norm_sq:
            return x_next, 'success', history

        x_k, f_k, g_k = x_next, f_next, g_next

    return x_k, 'iterations_exceeded', history



def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False, callback=None):
    history = defaultdict(list) if (trace or callback is not None) else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    start_time = time.perf_counter()

    f_k = oracle.func(x_k)
    grad_k = oracle.grad(x_k)
    if not _is_finite_scalar(f_k) or not _is_finite_array(grad_k):
        return x_k, 'computational_error', history

    grad0_norm_sq = float(grad_k.dot(grad_k))
    _push_history_smooth(history, start_time, x_k, f_k, grad_k)
    _apply_callback(history, callback, x_k, iteration=0)

    if grad0_norm_sq <= tolerance * grad0_norm_sq:
        return x_k, 'success', history

    alpha_k = None

    for k in range(max_iter):
        d_k = -grad_k
        previous_alpha = 2 * alpha_k if alpha_k is not None else None
        alpha_k = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=previous_alpha)

        if alpha_k is None or not np.isfinite(alpha_k):
            return x_k, 'computational_error', history

        x_k = x_k + alpha_k * d_k
        f_k = oracle.func(x_k)
        grad_k = oracle.grad(x_k)

        if not _is_finite_scalar(f_k) or not _is_finite_array(grad_k):
            return x_k, 'computational_error', history

        _push_history_smooth(history, start_time, x_k, f_k, grad_k)
        _apply_callback(history, callback, x_k, iteration=k + 1)
        grad_norm_sq = float(grad_k.dot(grad_k))

        if display:
            print('iter = {}, f = {:.6e}, ||g|| = {:.6e}, alpha = {:.6e}'.format(
                k + 1, f_k, np.sqrt(grad_norm_sq), alpha_k
            ))

        if grad_norm_sq <= tolerance * grad0_norm_sq:
            return x_k, 'success', history

    return x_k, 'iterations_exceeded', history



def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False, callback=None):
    history = defaultdict(list) if (trace or callback is not None) else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    start_time = time.perf_counter()

    f_k = oracle.func(x_k)
    grad_k = oracle.grad(x_k)
    if not _is_finite_scalar(f_k) or not _is_finite_array(grad_k):
        return x_k, 'computational_error', history

    grad0_norm_sq = float(grad_k.dot(grad_k))
    _push_history_smooth(history, start_time, x_k, f_k, grad_k)
    _apply_callback(history, callback, x_k, iteration=0)

    if grad0_norm_sq <= tolerance * grad0_norm_sq:
        return x_k, 'success', history

    for k in range(max_iter):
        hess_k = oracle.hess(x_k)

        try:
            c_factor, lower = scipy.linalg.cho_factor(hess_k)
            d_k = scipy.linalg.cho_solve((c_factor, lower), -grad_k)
        except LinAlgError:
            return x_k, 'newton_direction_error', history

        if not _is_finite_array(d_k):
            return x_k, 'computational_error', history

        alpha_k = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=1.0)
        if alpha_k is None or not np.isfinite(alpha_k):
            return x_k, 'computational_error', history

        x_k = x_k + alpha_k * d_k
        f_k = oracle.func(x_k)
        grad_k = oracle.grad(x_k)

        if not _is_finite_scalar(f_k) or not _is_finite_array(grad_k):
            return x_k, 'computational_error', history

        _push_history_smooth(history, start_time, x_k, f_k, grad_k)
        _apply_callback(history, callback, x_k, iteration=k + 1)
        grad_norm_sq = float(grad_k.dot(grad_k))

        if display:
            print('iter = {}, f = {:.6e}, ||g|| = {:.6e}, alpha = {:.6e}'.format(
                k + 1, f_k, np.sqrt(grad_norm_sq), alpha_k
            ))

        if grad_norm_sq <= tolerance * grad0_norm_sq:
            return x_k, 'success', history

    return x_k, 'iterations_exceeded', history
