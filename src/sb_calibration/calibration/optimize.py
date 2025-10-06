import numpy as np
from typing import Tuple
import os
from . import objective
from typing import Dict, Any, Optional, Tuple as Tup
import time
import sys
from scipy.optimize import minimize, differential_evolution


def make_internal_transform(p0_real: np.ndarray, bounds_real: list):
    p0 = np.asarray(p0_real, dtype=float)
    s = np.maximum(p0, 1e-6)
    lb = np.array([lo for (lo, hi) in bounds_real], dtype=float)
    ub = np.array([hi for (lo, hi) in bounds_real], dtype=float)
    z_lb = np.log(np.maximum(lb / s, 1e-12))
    z_ub = np.log(ub / s)

    def real_from_z(z):
        return s * np.exp(np.asarray(z, dtype=float))

    def z_from_real(p):
        return np.log(np.asarray(p, dtype=float) / s)

    return real_from_z, z_from_real, list(zip(z_lb, z_ub)), s


class Progress:
    def __init__(self, name="OPT", verbose: bool = True):
        self.name = name
        self.t0 = time.time()
        self.eval_count = 0
        self.best_sse = float("inf")
        self.verbose = verbose

    def mark_eval(self, sse, every=50):
        self.eval_count += 1
        if sse < self.best_sse:
            self.best_sse = float(sse)
        if self.verbose and (self.eval_count % every) == 0:
            dt = time.time() - self.t0
            print(f"[{self.name}] eval={self.eval_count:6d}  best_SSE={self.best_sse:.4e}  t={dt:6.1f}s")
            sys.stdout.flush()


def calibrate_full(mats: Dict[str, Any],
                    p0_real: np.ndarray,
                    bounds_real: list,
                    simulate_fn,
                    mode: str = "multistart",
                    pulses_by_assay: Optional[Dict[str, list]] = None,
                    x0_by_assay: Optional[Dict[str, np.ndarray]] = None,
                    weights: Optional[Dict[str, float]] = None,
                    n_starts: int = 8,
                    local_maxiter: int = 200,
                    patience_starts: int = 6,
                    patience_evals: int = 2000,
                    min_improvement_rel: float = 1e-3,
                    out_path: str = "mats/pbest_checkpoint.npz",
                    verbose: bool = True,
                    eval_print_every: int = 50) -> Tup[np.ndarray, float, Dict[str, Any]]:
    """Port of the legacy calibrator with limited defaults for fast unit tests.

    The function expects a callable `simulate_fn(p_real, t_meas, temp_segments, pulses, x0)`.
    """
    real_from_z, z_from_real, z_bounds, s = make_internal_transform(p0_real, bounds_real)
    z0 = np.clip(z_from_real(p0_real), [b[0] for b in z_bounds], [b[1] for b in z_bounds])

    stds = objective.compute_global_stds(mats)
    prog = Progress(name=f"OPT-{mode.upper()}", verbose=verbose)
    best_sse_seen = np.inf
    last_improve_eval = 0

    # resume if existing checkpoint available
    best_sse_seen = np.inf
    best_z_ckpt = None
    if out_path and os.path.exists(out_path):
        try:
            data = np.load(out_path)
            pbest_prev = data.get("pbest")
            score_prev = float(data.get("score")) if "score" in data else np.inf
            if pbest_prev is not None:
                z_prev = z_from_real(pbest_prev)
                if np.isfinite(score_prev):
                    best_sse_seen = score_prev
                    best_z_ckpt = z_prev
                    if verbose:
                        print(f"[RESUME] loaded previous best SSE={best_sse_seen:.4e}")
        except Exception:
            pass

    def save_checkpoint_if_better(z_vec, sse_val):
        nonlocal best_sse_seen, best_z_ckpt
        if sse_val < (1.0 - min_improvement_rel) * best_sse_seen:
            best_sse_seen = float(sse_val)
            best_z_ckpt = np.asarray(z_vec, dtype=float).copy()
            p_best_real = real_from_z(best_z_ckpt)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.savez(out_path, pbest=p_best_real, score=float(best_sse_seen))
            if verbose:
                dt = time.time() - prog.t0
                print(f"[CKPT] best SSE improved to {best_sse_seen:.4e}  eval={prog.eval_count}  t={dt:6.1f}s")

    def obj_z(z):
        nonlocal best_sse_seen, last_improve_eval
        p = real_from_z(z)
        sse = objective.sse_for_experiments_real(p, mats, pulses_by_assay, x0_by_assay, weights, stds, verbose=False, simulate_fn=simulate_fn)
        prog.mark_eval(sse, every=eval_print_every)
        if sse < (1.0 - min_improvement_rel) * best_sse_seen:
            best_sse_seen = sse
            last_improve_eval = prog.eval_count
            save_checkpoint_if_better(z, sse)
        if (prog.eval_count - last_improve_eval) >= patience_evals:
            raise RuntimeError("EARLY_STOP_EVALS")
        return sse

    z_best = None
    sse_best = np.inf
    result = {}

    if mode == "multistart":
        try:
            from scipy.stats.qmc import Sobol
            qmc = Sobol(d=len(z_bounds), scramble=True)
            U = qmc.random_base2(int(np.ceil(np.log2(n_starts))))
            U = U[:n_starts]
        except Exception:
            U = np.random.default_rng(123).uniform(size=(n_starts, len(z_bounds)))
        z_lo = np.array([b[0] for b in z_bounds]); z_hi = np.array([b[1] for b in z_bounds])
        Z = z_lo + U * (z_hi - z_lo)
        Z[0, :] = z0
        if best_z_ckpt is not None:
            Z[0, :] = best_z_ckpt

        local_runs = []
        no_improve_starts = 0
        for i, zi in enumerate(Z, start=1):
            if verbose:
                print(f"[MS] start {i}/{len(Z)}  (maxiter={local_maxiter})")
            try:
                loc = minimize(obj_z, zi, method="L-BFGS-B", bounds=z_bounds, options=dict(maxiter=local_maxiter, ftol=1e-9))
            except RuntimeError as e:
                if "EARLY_STOP_EVALS" in str(e):
                    if verbose:
                        print("[MS] :: early stop by evals ::")
                    break
                else:
                    raise
            local_runs.append(loc)
            if verbose:
                print(f"[MS] end   {i}/{len(Z)}  nit={getattr(loc, 'nit', '-') }  f={loc.fun:.4e}  success={loc.success}")
            improved = loc.success and (loc.fun < (1.0 - min_improvement_rel) * sse_best)
            if improved:
                sse_best = float(loc.fun)
                z_best = loc.x.copy()
                no_improve_starts = 0
                save_checkpoint_if_better(z_best, sse_best)
            else:
                no_improve_starts += 1
                if no_improve_starts >= patience_starts:
                    if verbose:
                        print(f"[MS] :: stopping after {no_improve_starts} starts without improvement ::")
                    break
        result = {"multistart": local_runs}

    elif mode == "de":
        def obj_z_de(z):
            p = real_from_z(z)
            sse = objective.sse_for_experiments_real(p, mats, pulses_by_assay, x0_by_assay, weights, stds, verbose=False, simulate_fn=simulate_fn)
            prog.mark_eval(sse, every=eval_print_every)
            return sse

        def cb_de(xk, convergence):
            if verbose:
                dt = time.time() - prog.t0
                print(f"[DE] conv={convergence:.3e}  best_SSE={prog.best_sse:.4e}  t={dt:6.1f}s")
                sys.stdout.flush()
            return False

        de_res = differential_evolution(
            obj_z_de, bounds=z_bounds, maxiter=10, popsize=6, mutation=(0.5, 1.0),
            recombination=0.7, tol=1e-6, polish=False, updating='deferred', workers=1, disp=False, callback=cb_de
        )
        loc = minimize(obj_z, de_res.x, method="L-BFGS-B", bounds=z_bounds, options=dict(maxiter=local_maxiter, ftol=1e-9))
        if verbose:
            print(f"[DE->LBFGS] nit={getattr(loc, 'nit', '-') }  f={loc.fun:.4e}  success={loc.success}")
        z_best = (loc.x if (loc.success and loc.fun < de_res.fun) else de_res.x).copy()
        sse_best = float(min(loc.fun, de_res.fun))
        save_checkpoint_if_better(z_best, sse_best)
        result = {"de": de_res, "local": loc}
    else:
        raise ValueError("mode debe ser 'de' o 'multistart'")

    if z_best is None:
        # fallback: pick best from local_runs
        if "multistart" in result and len(result["multistart"])>0:
            j = int(np.argmin([r.fun for r in result["multistart"]]))
            z_best = result["multistart"][j].x.copy(); sse_best = float(result["multistart"][j].fun)
        else:
            z_best = z0.copy(); sse_best = float(obj_z(z0))

    p_best_real = real_from_z(z_best)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, pbest=p_best_real, score=float(sse_best))
    return p_best_real, float(sse_best), result


def calibrate_dummy(y_meas: np.ndarray, initial_guess: np.ndarray = None, out_path: str = "mats/pbest_checkpoint.npz") -> Tuple[np.ndarray, float]:
    """A tiny 'optimizer' that returns zeros or initial guess and writes a checkpoint.

    This is a placeholder for the real optimization loop (DE, multistart, etc.).
    It writes a numpy .npz file with keys 'pbest' and 'score'.
    """
    if initial_guess is None:
        pbest = np.zeros(5, dtype=float)
    else:
        pbest = np.array(initial_guess, dtype=float)

    # Dummy score: variance of measurement as 'score' to have a float
    score = float(np.nanvar(y_meas)) if y_meas is not None else 0.0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, pbest=pbest, score=score)
    return pbest, score


def calibrate_simple(mats: Dict[str, Any],
                     p0_real: np.ndarray,
                     simulate_fn,
                     bounds_real=None,
                     weights: Optional[Dict[str, float]] = None,
                     out_path: str = "mats/pbest_checkpoint.npz") -> Tup[np.ndarray, float, Dict[str, Any]]:
    """A simplified calibrator that evaluates p0_real and returns it as 'best'.

    This is intentionally conservative: it computes stds, evaluates the SSE at p0_real
    using the provided simulate_fn, writes a checkpoint and returns the tuple
    (p_best, score, metadata). Replace with full optimizer later.
    """
    if bounds_real is None:
        bounds_real = []
    stds = objective.compute_global_stds(mats)
    sse = objective.sse_for_experiments_real(p0_real, mats, weights=weights, stds=stds, simulate_fn=simulate_fn)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, pbest=np.asarray(p0_real, dtype=float), score=float(sse))
    meta = {"method": "simple_eval", "stds": stds}
    return np.asarray(p0_real, dtype=float), float(sse), meta
