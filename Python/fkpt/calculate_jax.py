"""JAX-accelerated implementation of k-functions calculation.

This module provides a JAX/JIT-compiled version of the calculate() function that is
compatible with calculate_numpy.py but uses jax.numpy for automatic differentiation
and GPU acceleration.

Key differences from calculate_numpy.py:
- Uses jax.numpy instead of numpy
- All operations are functional (no in-place modifications)
- Critical functions are JIT-compiled for performance
- Accepts numpy arrays as input, converts internally, returns numpy arrays
"""

# Enable 64-bit precision in JAX to match NumPy
from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit
import numpy as np
from functools import partial

from fkpt.types import KFunctionsInitData, KFunctionsOut, Float64NDArray, AbsCalculator


@jit
def calc_2nd_derivs_jax(x, y):
    """JAX-compatible cubic spline initialization.

    Computes second derivatives for cubic spline interpolation.
    This is a JAX port of util.init_cubic_spline that can be JIT-compiled.

    Optimized version using lax.scan for better performance.

    Args:
        x: Knot positions (1D array, shape: (n_knots,))
        y: Function values at knots (shape: (n_features, n_knots))

    Returns:
        y2: Second derivatives (shape: (n_features, n_knots))
    """
    from jax import lax

    n = len(x)
    n_features = y.shape[0]

    # Pre-compute interval widths to avoid redundant calculations
    h = jnp.diff(x)  # x[i+1] - x[i] for all i
    h_span = x[2:] - x[:-2]  # x[i+1] - x[i-1] for i=1..n-2

    # Initialize arrays
    y2 = jnp.zeros_like(y)
    u = jnp.zeros_like(y)

    # Forward sweep using lax.scan (more efficient than fori_loop for sequential operations)
    def forward_sweep_scan(carry, i):
        y2, u = carry
        sig = h[i-1] / h_span[i-1]
        p = sig * y2[:, i-1] + 2.0
        y2_i = (sig - 1.0) / p
        udiff = (y[:, i+1] - y[:, i]) / h[i] - (y[:, i] - y[:, i-1]) / h[i-1]
        u_i = (6.0 * udiff / h_span[i-1] - sig * u[:, i-1]) / p

        # Update y2 and u at index i
        y2 = y2.at[:, i].set(y2_i)
        u = u.at[:, i].set(u_i)
        return (y2, u), None

    (y2, u), _ = lax.scan(forward_sweep_scan, (y2, u), jnp.arange(1, n-1))

    # Back substitution using lax.scan
    def back_sub_scan(y2, k):
        # k iterates from n-2 down to 0
        y2_k = y2[:, k] * y2[:, k+1] + u[:, k]
        return y2.at[:, k].set(y2_k), None

    y2, _ = lax.scan(back_sub_scan, y2, jnp.arange(n-2, -1, -1))

    return y2


def init_cubic_spline_jax(xa, x):
    """Pre-compute cubic spline interpolation coefficients for given xa and x.

    Args:
        xa: Knot positions (1D array, must be increasing)
        x: Evaluation points (any shape)

    Returns:
        Dictionary containing precomputed coefficients needed for evaluation
    """
    # Remember original shape as a concrete Python tuple (not a traced value)
    x_shape = tuple(int(s) for s in x.shape)
    x_flat = x.ravel()

    # Get flat indices
    idx_hi_flat = jnp.searchsorted(xa, x_flat, side='right')
    idx_hi_flat = jnp.clip(idx_hi_flat, 1, xa.size - 1)
    idx_lo_flat = idx_hi_flat - 1

    # Compute interpolation coefficients
    h_flat = xa[idx_hi_flat] - xa[idx_lo_flat]
    a_flat = (xa[idx_hi_flat] - x_flat) / h_flat
    b_flat = (x_flat - xa[idx_lo_flat]) / h_flat

    # Power operations
    a2_flat = a_flat * a_flat
    a3_flat = a2_flat * a_flat
    b2_flat = b_flat * b_flat
    b3_flat = b2_flat * b_flat
    h2_flat = h_flat * h_flat

    return {
        'x_shape': x_shape,
        'idx_lo_flat': idx_lo_flat,
        'idx_hi_flat': idx_hi_flat,
        'a_flat': a_flat,
        'b_flat': b_flat,
        'a3_flat': a3_flat,
        'b3_flat': b3_flat,
        'h2_flat': h2_flat
    }


def eval_cubic_spline_jax(ya, y2a, x_shape, idx_lo_flat, idx_hi_flat, a_flat, b_flat, a3_flat, b3_flat, h2_flat):
    """JAX-compatible cubic spline evaluation using precomputed coefficients.

    Args:
        ya: Function values at knots (shape: (n_features, len(xa)))
        y2a: Second derivatives at knots (shape: (n_features, len(xa)))
        x_shape: Original shape of evaluation points (as Python tuple)
        idx_lo_flat, idx_hi_flat: Precomputed indices
        a_flat, b_flat, a3_flat, b3_flat, h2_flat: Precomputed interpolation coefficients

    Returns:
        Interpolated values (shape: (*x_shape, n_features))
    """
    # Index into ya and y2a (ya has shape (n_features, n_knots))
    # We want result shape (n_features, n_eval_points)
    ya_lo = ya[:, idx_lo_flat]  # (n_features, n_eval_points)
    ya_hi = ya[:, idx_hi_flat]
    y2a_lo = y2a[:, idx_lo_flat]
    y2a_hi = y2a[:, idx_hi_flat]

    # Compute interpolated values (broadcasting a_flat etc to match)
    result_flat = (
        a_flat[None, :] * ya_lo + b_flat[None, :] * ya_hi +
        ((a3_flat[None, :] - a_flat[None, :]) * y2a_lo +
         (b3_flat[None, :] - b_flat[None, :]) * y2a_hi) * h2_flat[None, :] / 6.0
    )

    # Reshape to (*x_shape, n_features)
    n_features = ya.shape[0]
    result = result_flat.T.reshape(*x_shape, n_features)

    return result


@partial(jit, static_argnums=(2, 10, 18))
def _calculate_jax_core(
        Y, Y2,
        # Spline coefficients for logk_grid
        spline_logk_shape, spline_logk_idx_lo, spline_logk_idx_hi,
        spline_logk_a, spline_logk_b, spline_logk_a3, spline_logk_b3, spline_logk_h2,
        # Spline coefficients for kk_grid
        spline_kk_shape, spline_kk_idx_lo, spline_kk_idx_hi,
        spline_kk_a, spline_kk_b, spline_kk_a3, spline_kk_b3, spline_kk_h2,
        # Spline coefficients for y
        spline_y_shape, spline_y_idx_lo, spline_y_idx_hi,
        spline_y_a, spline_y_b, spline_y_a3, spline_y_b3, spline_y_h2,
        logk_grid2, dkk, dkk_reshaped, scale_Q,
        r, r2, x, w, x2, y2, y,
        r_r, r2_r, x_r, w_r, x2_r, y2_r, AngleEvR, AngleEvR2, dkk_r,
        kk_grid, logk_grid,
        A, ApOverf0, CFD3, CFD3p, sigma2v
    ):
    """Core JAX calculation - all arrays are JAX arrays.

    This is the JIT-compiled inner function that operates entirely on JAX arrays.
    All precomputed quantities are passed as parameters.

    Note: Y2 should be pre-computed and passed in as a JAX array for best performance.
    """

    # Interpolate onto output grid using precomputed coefficients
    Pout, Pout_nw, fout = eval_cubic_spline_jax(
        Y, Y2, spline_logk_shape, spline_logk_idx_lo, spline_logk_idx_hi,
        spline_logk_a, spline_logk_b, spline_logk_a3, spline_logk_b3, spline_logk_h2
    ).T

    # Interpolate onto quadrature grid using precomputed coefficients
    Pkk, Pkk_nw, fkk = eval_cubic_spline_jax(
        Y, Y2, spline_kk_shape, spline_kk_idx_lo, spline_kk_idx_hi,
        spline_kk_a, spline_kk_b, spline_kk_a3, spline_kk_b3, spline_kk_h2
    ).T

    # ============================================================================
    # Q-FUNCTIONS: Vectorized over ALL dimensions
    # ============================================================================

    # Use precomputed Q-function quantities
    # (r, r2, x, w, x2, y2, y are passed as parameters)

    # Loop over quadrature k values (line 378 in C)
    fp = fkk[1:].reshape(-1, 1, 1)

    # Interpolate power spectra at (ki * y) points using precomputed coefficients
    interp_result = eval_cubic_spline_jax(
        Y, Y2, spline_y_shape, spline_y_idx_lo, spline_y_idx_hi,
        spline_y_a, spline_y_b, spline_y_a3, spline_y_b3, spline_y_h2
    )
    psl_w, psl_nw, fkmp = interp_result.T  # Unpack gives (120, 1, 299, 10) each

    # Transpose back to (NQ, nquadSteps-1, 1, Nk)
    psl_w = psl_w.T  # (10, 299, 1, 120)
    psl_nw = psl_nw.T  # (10, 299, 1, 120)
    fkmp = fkmp.T  # (10, 299, 1, 120) - already has the correct shape!

    # Concatenate wiggle and no-wiggle components (matches numpy version)
    psl = jnp.concatenate([psl_w, psl_nw], axis=2)  # shape (NQ, nquadSteps-1, 2, Nk) -> (10, 299, 2, 120)
    # fkmp already has shape (10, 299, 1, 120) which is what we need

    # Compute SPT kernels F2evQ and G2evQ
    AngleEvQ = (x - r) / y
    AngleEvQ2 = AngleEvQ * AngleEvQ
    fsum = fp + fkmp

    S2evQ = AngleEvQ ** 2 - 1.0/3.0
    F2evQ = (1.0/2.0 + 3.0/14.0 * A + (1.0/2.0 - 3.0/14.0 * A) * AngleEvQ2 +
             AngleEvQ / 2.0 * (y/r + r/y))
    G2evQ = (3.0/14.0 * A * fsum + 3.0/14.0 * ApOverf0 +
             (1.0/2.0 * fsum - 3.0/14.0 * A * fsum - 3.0/14.0 * ApOverf0) * AngleEvQ2 +
             AngleEvQ / 2.0 * (fkmp * y/r + fp * r/y))

    # Precompute some temporary expressions that are used multiple times
    wpsl = w * psl
    fkmpr2 = fkmp * r2
    rx = r * x
    y4 = y2 * y2

    # P22 kernels
    P22dd_B = jnp.sum(wpsl * (2.0 * r2 * F2evQ**2), axis=0)
    P22du_B = jnp.sum(wpsl * (2.0 * r2 * F2evQ * G2evQ), axis=0)
    P22uu_B = jnp.sum(wpsl * (2.0 * r2 * G2evQ**2), axis=0)

    # ========== 5 THREE-POINT CORRELATION FUNCTION KERNELS (Q-part) ==========

    # I1udd1tA
    I1udd1tA_B = jnp.sum(wpsl * (
        2.0 * (fp * rx + fkmpr2 * (1.0 - rx) / y2) * F2evQ
        ), axis=0)

    # I2uud1tA
    I2uud1tA_B = jnp.sum(wpsl * (-fp * fkmpr2 * (1.0 - x2) / y2 * F2evQ), axis=0)

    # I2uud2tA
    I2uud2tA_B = jnp.sum(wpsl * (
        2.0 * (fp * rx + fkmpr2 * (1.0 - rx) / y2) * G2evQ
        + fp * fkmp * (r2 * (1.0 - 3.0 * x2) + 2.0 * rx) / y2 * F2evQ
        ), axis=0)

    # I3uuu2tA
    I3uuu2tA_B = jnp.sum(wpsl * (fp * fkmpr2 * (x2 - 1.0) / y2 * G2evQ), axis=0)

    # I3uuu3tA
    I3uuu3tA_B = jnp.sum(wpsl * (
        fp * fkmp * (r2 * (1.0 - 3.0 * x2) + 2.0 * rx) / y2 * G2evQ
        ), axis=0)

    # ========== 7 BpC TERM KERNELS (Q-part, will become D-terms) ==========

    # I2uudd1BpC
    I2uudd1BpC_B = jnp.sum(wpsl * (
        1.0 / 4.0 * (1.0 - x2) * (fp * fp + fkmpr2 * fkmpr2 / y4)
        + fp * fkmpr2 * (-1.0 + x2) / y2 / 2.0
        ), axis=0)

    # I2uudd2BpC
    I2uudd2BpC_B = jnp.sum(wpsl * (
        (
            fp * fp * (-1.0 + 3.0 * x2)
            + 2.0 * fkmp * fp * r * (r + 2.0 * x - 3.0 * r * x2) / y2
            + fkmp * fkmpr2 * (2.0 - 4.0 * rx + r2 * (-1.0 + 3.0 * x2)) / y4
        )
        / 4.0
        ), axis=0)

    # I3uuud2BpC
    I3uuud2BpC_B = jnp.sum(wpsl * (
        -(
            fkmp * fp * (
                fkmp * (-2.0 + 3.0 * rx) * r2
                - fp * (-1.0 + 3.0 * rx) * (1.0 - 2.0 * rx + r2)
            )
            * (-1.0 + x2)
        )
        / (2.0 * y2 * y2)
        ), axis=0)

    # I3uuud3BpC
    I3uuud3BpC_B = jnp.sum(wpsl * (
        (
            fkmp * fp * (
                -(
                    fp
                    * (1.0 - 2.0 * rx + r2)
                    * (1.0 - 3.0 * x2 + rx * (-3.0 + 5.0 * x2))
                )
                + fkmp * r * (2.0 * x + r * (2.0 - 6.0 * x2 + rx * (-3.0 + 5.0 * x2)))
            )
        )
        / (2.0 * y4)
        ), axis=0)

    # I4uuuu2BpC
    I4uuuu2BpC_B = jnp.sum(wpsl * (
        3.0 * fkmp**2 * fp**2 * r2 * (-1.0 + x2) ** 2 / (16.0 * y4)
        ), axis=0)

    # I4uuuu3BpC
    I4uuuu3BpC_B = jnp.sum(wpsl * (
        -(
            fkmp**2 * fp**2 * (-1.0 + x2) * (2.0 + 3.0 * r * (-4.0 * x + r * (-1.0 + 5.0 * x2)))
        )
        / (8.0 * y2 * y2)
        ), axis=0)

    # I4uuuu4BpC
    I4uuuu4BpC_B = jnp.sum(wpsl * (
        (
            fkmp**2 * fp**2 * (
                -4.0
                + 8.0 * rx * (3.0 - 5.0 * x2)
                + 12.0 * x2
                + r2 * (3.0 - 30.0 * x2 + 35.0 * x2 ** 2)
            )
        )
        / (16.0 * y4)
        ), axis=0)

    # Left endpoints for power spectra
    PSLB = jnp.stack([Pkk[1:], Pkk_nw[1:]], axis=1)[:, :, None]  # shape (nQuadSteps-1, 2, 1)

    # Use precomputed dkk_reshaped and scale_Q

    # Bias terms
    Pb1b2_B = jnp.sum(wpsl * (r2 * F2evQ), axis=0)
    Pb1bs2_B = jnp.sum(wpsl * (r2 * F2evQ * S2evQ), axis=0)
    Pratio = PSLB / psl
    PratioInv = psl / PSLB
    Pb22_B = jnp.sum(wpsl * (
        1.0 / 2.0 * r2 * (1.0 / 2.0 * (1.0 - Pratio) + 1.0 / 2.0 * (1.0 - PratioInv))
        ), axis=0)
    Pb2s2_B = jnp.sum(wpsl * (
        1.0 / 2.0 * r2 * (
            1.0 / 2.0 * (S2evQ - 2.0 / 3.0 * Pratio)
            + 1.0 / 2.0 * (S2evQ - 2.0 / 3.0 * PratioInv)
        )
        ), axis=0)
    Ps22_B = jnp.sum(wpsl * (
        1.0 / 2.0 * r2
        * (
            1.0 / 2.0 * (S2evQ ** 2 - 4.0 / 9.0 * Pratio)
            + 1.0 / 2.0 * (S2evQ ** 2 - 4.0 / 9.0 * PratioInv)
        )
        ), axis=0)
    Pb2theta_B = jnp.sum(wpsl * (r2 * G2evQ), axis=0)
    Pbs2theta_B = jnp.sum(wpsl * (r2 * S2evQ * G2evQ), axis=0)

    # Apply trapezoidal rule (functional version for JAX)
    def trapsumQ(B):
        """Functional trapezoidal integration for Q-functions.

        Matches numpy version which computes:
        sum_i ( (B[i] + B[i-1]) * dk[i] ) * scale_Q * PSLB
        The first element uses dk[0] with B[0] only (no previous element).
        """
        B = B * scale_Q * PSLB
        # Compute trapezoidal sum: (B[i-1] + B[i]) * dk[i] for i >= 1, plus B[0] * dk[0]
        return jnp.sum((B[:-1] + B[1:]) * dkk_reshaped[1:], axis=0) + B[0] * dkk_reshaped[0]

    P22dd = trapsumQ(P22dd_B)
    P22du = trapsumQ(P22du_B)
    P22uu = trapsumQ(P22uu_B)

    I1udd1tA = trapsumQ(I1udd1tA_B)
    I2uud1tA = trapsumQ(I2uud1tA_B)
    I2uud2tA = trapsumQ(I2uud2tA_B)
    I3uuu2tA = trapsumQ(I3uuu2tA_B)
    I3uuu3tA = trapsumQ(I3uuu3tA_B)

    I2uudd1BpC = trapsumQ(I2uudd1BpC_B)
    I2uudd2BpC = trapsumQ(I2uudd2BpC_B)
    I3uuud2BpC = trapsumQ(I3uuud2BpC_B)
    I3uuud3BpC = trapsumQ(I3uuud3BpC_B)
    I4uuuu2BpC = trapsumQ(I4uuuu2BpC_B)
    I4uuuu3BpC = trapsumQ(I4uuuu3BpC_B)
    I4uuuu4BpC = trapsumQ(I4uuuu4BpC_B)

    # Bias terms
    Pb1b2 = trapsumQ(Pb1b2_B)
    Pb1bs2 = trapsumQ(Pb1bs2_B)
    Pb22 = trapsumQ(Pb22_B)
    Pb2s2 = trapsumQ(Pb2s2_B)
    Ps22 = trapsumQ(Ps22_B)
    Pb2theta = trapsumQ(Pb2theta_B)
    Pbs2theta = trapsumQ(Pbs2theta_B)

    # ============================================================================
    # R-FUNCTIONS: Also fully vectorized
    # ============================================================================

    # Get f(k) at output k values
    fk = fout

    # Use precomputed R-function quantities
    # (r_r, r2_r, x_r, w_r, x2_r, y2_r, AngleEvR, AngleEvR2 are passed as parameters)

    # R-function uses fp from kk[1:-1] and psl from Pkk[1:-1]
    fp_r = fkk[1:-1].reshape(-1, 1, 1)
    psl_r = jnp.stack((Pkk[1:-1], Pkk_nw[1:-1]), axis=1)[:,:,None] # shape (nquadSteps-2, 2, 1)

    F2evR = (1.0/2.0 + 3.0/14.0 * A + (1.0/2.0 - 3.0/14.0 * A) * AngleEvR2 +
             AngleEvR / 2.0 * (1.0/r_r + r_r))
    G2evR = (3.0/14.0 * A * (fp_r + fk) + 3.0/14.0 * ApOverf0 +
             (1.0/2.0 * (fp_r + fk) - 3.0/14.0 * A * (fp_r + fk) -
              3.0/14.0 * ApOverf0) * AngleEvR2 +
             AngleEvR / 2.0 * (fk/r_r + fp_r * r_r))

    # ========== 5 THREE-POINT CORRELATION FUNCTION KERNELS (R-part) ==========
    wpsl_r = w_r * psl_r

    Gamma2evR = A * (1.0 - x2_r)
    Gamma2fevR = A * (1.0 - x2_r) * (fk + fp_r) / 2.0 + 1.0/2.0 * ApOverf0 * (1.0 - x2_r)
    C3Gamma3 = 2.0 * 5.0/21.0 * CFD3 * (1.0 - x2_r) * (1.0 - x2_r) / y2_r
    C3Gamma3f = 2.0 * 5.0/21.0 * CFD3p * (1.0 - x2_r) * (1.0 - x2_r) / y2_r * (fk + 2.0 * fp_r) / 3.0
    G3K = (
        C3Gamma3f / 2.0 + (2.0 * Gamma2fevR * x_r) / (7.0 * r_r) - (fk * x2_r) / (6.0 * r2_r)
        + fp_r * Gamma2evR * (1.0 - r_r * x_r) / (7.0 * y2_r)
        - 1.0/7.0 * (fp_r * Gamma2evR + 2.0 * Gamma2fevR) * (1.0 - x2_r) / y2_r)
    F3K = C3Gamma3 / 6.0 - x2_r / (6.0 * r2_r) + (Gamma2evR * x_r * (1.0 - r_r * x_r)) / (7.0 * r_r * y2_r)

    P13dd_B = jnp.sum(wpsl_r * (6.0 * r2_r * F3K), axis=0)
    P13du_B = jnp.sum(wpsl_r * (3.0 * r2_r * G3K + 3.0 * r2_r * F3K * fk), axis=0)
    P13uu_B = jnp.sum(wpsl_r * (6.0 * r2_r * G3K * fk), axis=0)

    sigma32PSL_B = jnp.sum(wpsl_r * (
        (5.0 * r2_r * (7.0 - 2.0*r2_r + 4.0*r_r*x_r + 6.0*(-2.0 + r2_r)*x2_r - 12.0*r_r*x2_r*x_r + 9.0*x2_r*x2_r))
        / (24.0 * y2_r)
    ), axis=0)

    # I1udd1a
    I1udd1a_B = jnp.sum(wpsl_r * (
        2.0 * r2_r * (1.0 - r_r * x_r) / y2_r * G2evR + 2.0 * fp_r * r_r * x_r * F2evR
    ), axis=0)

    # I2uud1a
    I2uud1a_B = jnp.sum(wpsl_r * (
        -fp_r * r2_r * (1.0 - x2_r) / y2_r * G2evR
    ), axis=0)

    # I2uud2a
    I2uud2a_B = jnp.sum(wpsl_r * (
        ((r2_r * (1.0 - 3.0 * x2_r) + 2.0 * r_r * x_r) / y2_r * fp_r +
                fk * 2.0 * r2_r * (1.0 - r_r * x_r) / y2_r) * G2evR + 2.0 * x_r * r_r * fp_r * fk * F2evR
    ), axis=0)

    # I3uuu2a
    I3uuu2a_B = jnp.sum(wpsl_r * (
        -fp_r * r2_r * (1.0 - x2_r) / y2_r * G2evR * fk
    ), axis=0)

    # I3uuu3a
    I3uuu3a_B = jnp.sum(wpsl_r * (
        (r2_r * (1.0 - 3.0 * x2_r) + 2.0 * r_r * x_r) / y2_r * fp_r * fk * G2evR
    ), axis=0)

    # Calculate scaling for R-functions
    pkl_k = jnp.stack([Pout, Pout_nw], axis=0)  # shape (2, Nk)
    scale_R = logk_grid2 / (8.0 * jnp.pi ** 2) * pkl_k

    # Use precomputed dkk_r for trapezoidal integration

    def trapsumR(B):
        """Functional trapezoidal integration for R-functions.

        Matches numpy version computation pattern.
        """
        B = B * scale_R
        # Compute trapezoidal sum: (B[i-1] + B[i]) * dk[i] for i >= 1, plus B[0] * dk[0]
        return jnp.sum((B[:-1] + B[1:]) * dkk_r[1:], axis=0) + B[0] * dkk_r[0]

    I1udd1a = trapsumR(I1udd1a_B)
    I2uud1a = trapsumR(I2uud1a_B)
    I2uud2a = trapsumR(I2uud2a_B)
    I3uuu2a = trapsumR(I3uuu2a_B)
    I3uuu3a = trapsumR(I3uuu3a_B)

    P13uu = trapsumR(P13uu_B)
    P13du = trapsumR(P13du_B)
    P13dd = trapsumR(P13dd_B)

    sigma32PSL = trapsumR(sigma32PSL_B)

    # ============================================================================
    # Combine Q and R functions
    # ============================================================================
    I1udd1A = I1udd1tA + 2.0 * I1udd1a
    I2uud1A = I2uud1tA + 2.0 * I2uud1a
    I2uud2A = I2uud2tA + 2.0 * I2uud2a
    I3uuu2A = I3uuu2tA + 2.0 * I3uuu2a
    I3uuu3A = I3uuu3tA + 2.0 * I3uuu3a

    # ============================================================================
    # D-TERMS (B + C - G corrections)
    # ============================================================================
    fk_grid = fk

    # I2uudd1D: subtract k^2 * sigma2v * P_L(k)
    I2uudd1BpC = I2uudd1BpC - logk_grid2 * sigma2v * pkl_k

    # I3uuud2D: subtract 2 * k^2 * sigma2v * f(k) * P_L(k)
    I3uuud2BpC = I3uuud2BpC - 2.0 * logk_grid2 * sigma2v * fk_grid * pkl_k

    # I4uuuu3D: subtract k^2 * sigma2v * f(k)^2 * P_L(k)
    I4uuuu3BpC = I4uuuu3BpC - logk_grid2 * sigma2v * fk_grid ** 2 * pkl_k

    return (
        P22dd, P22du, P22uu,
        I1udd1A, I2uud1A, I2uud2A,
        I3uuu2A, I3uuu3A,
        I2uudd1BpC, I2uudd2BpC,
        I3uuud2BpC, I3uuud3BpC,
        I4uuuu2BpC, I4uuuu3BpC, I4uuuu4BpC,
        Pb1b2, Pb1bs2, Pb22, Pb2s2, Ps22,
        Pb2theta, Pbs2theta,
        P13dd, P13du, P13uu,
        sigma32PSL,
        pkl_k
    )


class JaxCalculator(AbsCalculator):
    """JAX-accelerated k-functions calculator implementing the AbsCalculator interface.

    This calculator uses JAX for JIT compilation and automatic differentiation.
    Grid data is converted to JAX arrays once during initialization and reused.
    """

    def __init__(self):
        """Initialize an empty calculator. Call initialize() before evaluate()."""
        self.k_in_jax = None
        self.logk_grid_jax = None
        self.kk_grid_jax = None
        self.xxQ_jax = None
        self.wwQ_jax = None
        self.xxR_jax = None
        self.wwR_jax = None
        # Precomputed spline interpolation coefficients
        self.spline_logk = None
        self.spline_kk = None
        self.spline_y = None
        # Precomputed Q-function quantities
        self.logk_grid2_jax = None
        self.dkk_jax = None
        self.dkk_reshaped_jax = None
        self.scale_Q_jax = None
        self.r_jax = None
        self.r2_jax = None
        self.x_jax = None
        self.w_jax = None
        self.x2_jax = None
        self.y2_jax = None
        self.y_jax = None
        # Precomputed R-function quantities
        self.r_r_jax = None
        self.r2_r_jax = None
        self.x_r_jax = None
        self.w_r_jax = None
        self.x2_r_jax = None
        self.y2_r_jax = None
        self.AngleEvR_jax = None
        self.AngleEvR2_jax = None
        self.dkk_r_jax = None

    def initialize(self, data: KFunctionsInitData) -> None:
        """Initialize the calculator with grid data and quadrature points.

        Converts numpy arrays to JAX arrays and stores them for reuse.
        Pre-computes all fixed quantities that don't depend on input power spectra.

        Args:
            data: Initialization data containing k-grid, quadrature points, etc.
        """
        self.k_in_jax = jnp.asarray(data.k_in, dtype=jnp.float64)
        self.logk_grid_jax = jnp.asarray(data.logk_grid, dtype=jnp.float64)
        self.kk_grid_jax = jnp.asarray(data.kk_grid, dtype=jnp.float64)
        self.xxQ_jax = jnp.asarray(data.xxQ, dtype=jnp.float64)
        self.wwQ_jax = jnp.asarray(data.wwQ, dtype=jnp.float64)
        self.xxR_jax = jnp.asarray(data.xxR, dtype=jnp.float64)
        self.wwR_jax = jnp.asarray(data.wwR, dtype=jnp.float64)

        # Pre-compute commonly used grid quantities
        self.logk_grid2_jax = self.logk_grid_jax * self.logk_grid_jax
        self.dkk_jax = jnp.diff(self.kk_grid_jax)
        self.dkk_reshaped_jax = self.dkk_jax.reshape(-1, 1, 1)
        self.scale_Q_jax = 0.25 * self.logk_grid2_jax / jnp.pi ** 2

        # Pre-compute spline coefficients for fixed interpolation grids
        # 1. Coefficients for interpolating onto logk_grid
        self.spline_logk = init_cubic_spline_jax(self.k_in_jax, self.logk_grid_jax)

        # 2. Coefficients for interpolating onto kk_grid
        self.spline_kk = init_cubic_spline_jax(self.k_in_jax, self.kk_grid_jax)

        # 3. Pre-compute Q-function quantities and spline coefficients
        # Compute variable integration limits for mu (local variables only needed here)
        rmax = self.k_in_jax[-1] / self.logk_grid_jax
        rmin = self.k_in_jax[0] / self.logk_grid_jax
        rmax2 = rmax * rmax
        rmin2 = rmin * rmin

        self.r_jax = self.kk_grid_jax[1:].reshape(-1, 1, 1) / self.logk_grid_jax
        self.r2_jax = self.r_jax * self.r_jax

        mumin = jnp.maximum(-1.0, (1.0 + self.r2_jax - rmax2) / (2.0 * self.r_jax))
        mumax = jnp.minimum(1.0, (1.0 + self.r2_jax - rmin2) / (2.0 * self.r_jax))
        mumax = jnp.where(self.r_jax >= 0.5, 0.5 / self.r_jax, mumax)

        # Scale Gauss-Legendre nodes and weights to [mumin, mumax]
        dmu = mumax - mumin
        xGL = 0.5 * (dmu * self.xxQ_jax.reshape(-1, 1, 1, 1) + (mumax + mumin))
        wGL = 0.5 * dmu * self.wwQ_jax.reshape(-1, 1, 1, 1)

        # Compute x, w, x2, y2, y values for Q-function integration
        self.x_jax = xGL
        self.w_jax = wGL
        self.x2_jax = self.x_jax * self.x_jax
        self.y2_jax = 1.0 + self.r2_jax - 2.0 * self.r_jax * self.x_jax
        self.y_jax = jnp.sqrt(self.y2_jax)

        # Pre-compute coefficients for interpolating at logk_grid * y
        self.spline_y = init_cubic_spline_jax(self.k_in_jax, self.logk_grid_jax * self.y_jax)

        # Pre-compute R-function quantities
        # R-function uses r from kk[1:-1] (indices 1 to nquadSteps-2)
        self.r_r_jax = self.kk_grid_jax[1:-1].reshape(-1, 1, 1) / self.logk_grid_jax
        self.r2_r_jax = self.r_r_jax * self.r_r_jax

        # Gauss-Legendre points in [-1, 1] (fixed limits for R-functions)
        self.x_r_jax = self.xxR_jax.reshape(-1, 1, 1, 1)
        self.w_r_jax = self.wwR_jax.reshape(-1, 1, 1, 1)
        self.x2_r_jax = self.x_r_jax * self.x_r_jax
        self.y2_r_jax = 1.0 + self.r2_r_jax - 2.0 * self.r_r_jax * self.x_r_jax

        # R-function angles (independent of input parameters)
        self.AngleEvR_jax = -self.x_r_jax
        self.AngleEvR2_jax = self.AngleEvR_jax * self.AngleEvR_jax

        # R-function trapezoidal integration spacing
        self.dkk_r_jax = self.dkk_reshaped_jax[:-1].reshape(-1, 1, 1)

    def evaluate(self, Pk_in: Float64NDArray, Pk_nw_in: Float64NDArray,
                 fk_in: Float64NDArray, A: float, ApOverf0: float, CFD3: float,
                 CFD3p: float, sigma2v: float, f0: float) -> KFunctionsOut:
        """Evaluate k-functions given input power spectra.

        Args:
            Pk_in: Linear power spectrum values at k_in grid points
            Pk_nw_in: No-wiggle linear power spectrum values at k_in grid points
            fk_in: Growth rate f(k) values at k_in grid points
            A: Cosmological parameter A
            ApOverf0: Cosmological parameter Ap/f0
            CFD3: Cosmological parameter CFD3
            CFD3p: Cosmological parameter CFD3p
            sigma2v: Velocity dispersion parameter
            f0: Reference growth rate

        Returns:
            KFunctionsOut containing all computed k-functions (as numpy arrays)
        """
        # Stack input power spectra and normalize fk by f0
        Y = np.stack([Pk_in, Pk_nw_in, fk_in / f0], axis=0)
        Y_jax = jnp.asarray(Y, dtype=jnp.float64)

        # Compute second derivatives for cubic spline
        Y2_jax = calc_2nd_derivs_jax(self.k_in_jax, Y_jax)

        # Run JIT-compiled calculation with all precomputed values
        # Unpack spline coefficient dictionaries
        results = _calculate_jax_core(
            Y_jax, Y2_jax,
            # Spline coefficients for logk_grid
            self.spline_logk['x_shape'], self.spline_logk['idx_lo_flat'], self.spline_logk['idx_hi_flat'],
            self.spline_logk['a_flat'], self.spline_logk['b_flat'], self.spline_logk['a3_flat'],
            self.spline_logk['b3_flat'], self.spline_logk['h2_flat'],
            # Spline coefficients for kk_grid
            self.spline_kk['x_shape'], self.spline_kk['idx_lo_flat'], self.spline_kk['idx_hi_flat'],
            self.spline_kk['a_flat'], self.spline_kk['b_flat'], self.spline_kk['a3_flat'],
            self.spline_kk['b3_flat'], self.spline_kk['h2_flat'],
            # Spline coefficients for y
            self.spline_y['x_shape'], self.spline_y['idx_lo_flat'], self.spline_y['idx_hi_flat'],
            self.spline_y['a_flat'], self.spline_y['b_flat'], self.spline_y['a3_flat'],
            self.spline_y['b3_flat'], self.spline_y['h2_flat'],
            self.logk_grid2_jax, self.dkk_jax, self.dkk_reshaped_jax, self.scale_Q_jax,
            self.r_jax, self.r2_jax, self.x_jax, self.w_jax, self.x2_jax, self.y2_jax, self.y_jax,
            self.r_r_jax, self.r2_r_jax, self.x_r_jax, self.w_r_jax, self.x2_r_jax, self.y2_r_jax,
            self.AngleEvR_jax, self.AngleEvR2_jax, self.dkk_r_jax,
            self.kk_grid_jax, self.logk_grid_jax,
            A, ApOverf0, CFD3, CFD3p, sigma2v
        )

        # Convert JAX arrays back to numpy arrays
        results_np = tuple(np.asarray(r) for r in results)

        return KFunctionsOut(*results_np)
