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

from fkpt.types import KFunctionsIn, KFunctionsOut, Float64NDArray


@jit
def eval_cubic_spline_jax(xa, ya, y2a, x):
    """JAX-compatible cubic spline evaluation.

    Evaluates the cubic spline at given x values assuming xa is increasing.
    This is a JAX port of util.eval_cubic_spline that can be JIT-compiled.

    Args:
        xa: Knot positions (1D array)
        ya: Function values at knots (shape: (n_features, len(xa)))
        y2a: Second derivatives at knots (shape: (n_features, len(xa)))
        x: Evaluation points (any shape)

    Returns:
        Interpolated values (shape: (*x.shape, n_features))
    """
    # Remember original shape
    x_shape = x.shape
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


@jit
def _calculate_jax_core(
        k_in, logk_grid, kk_grid, Y, Y2,
        xxQ, wwQ, xxR, wwR,
        A, ApOverf0, CFD3, CFD3p, sigma2v
    ):
    """Core JAX calculation - all arrays are JAX arrays.

    This is the JIT-compiled inner function that operates entirely on JAX arrays.
    """

    # Define interpolator using JAX cubic spline
    def interpolator(x):
        return eval_cubic_spline_jax(k_in, Y, Y2, x)

    # Interpolate onto output grid
    Pout, Pout_nw, fout = interpolator(logk_grid).T

    # Interpolate onto quadrature grid
    Pkk, Pkk_nw, fkk = interpolator(kk_grid).T

    dkk = jnp.diff(kk_grid)

    # ============================================================================
    # Q-FUNCTIONS: Vectorized over ALL dimensions
    # ============================================================================

    # Compute variable integration limits for mu
    rmax = k_in[-1] / logk_grid  # shape (Nk,)
    rmin = k_in[0] / logk_grid  # shape (Nk,)
    rmax2 = rmax * rmax
    rmin2 = rmin * rmin

    # Loop over quadrature k values (line 378 in C)
    fp = fkk[1:].reshape(-1, 1, 1)

    # r shape: (nquadSteps-1, 1, 1)
    r = kk_grid[1:].reshape(-1, 1, 1) / logk_grid
    r2 = r * r

    # mumin, mumax: shape (nquadSteps-1, 1, Nk)
    mumin = jnp.maximum(-1.0, (1.0 + r2 - rmax2) / (2.0 * r))
    mumax = jnp.minimum(1.0, (1.0 + r2 - rmin2) / (2.0 * r))

    # Line 389-390 in C: if r >= 0.5, mumax = 0.5/r
    mumax = jnp.where(r >= 0.5, 0.5 / r, mumax)

    # Scale Gauss-Legendre nodes and weights to [mumin, mumax]
    # Shape: (NQ, nquadSteps-1, 1, Nk)
    dmu = mumax - mumin
    xGL = 0.5 * (dmu * xxQ.reshape(-1, 1, 1, 1) + (mumax + mumin))
    wGL = 0.5 * dmu * wwQ.reshape(-1, 1, 1, 1)

    # Perform Gauss-Legendre quadrature over mu
    x = xGL
    w = wGL
    x2 = x * x
    y2 = 1.0 + r2 - 2.0 * r * x
    y = jnp.sqrt(y2)

    # Interpolate power spectra at (ki * y) points
    # numpy does: psl_w, psl_nw, fkmp = interpolator(logk_grid * y).T
    # which unpacks the first dimension after transpose
    interp_result = interpolator(logk_grid * y)  # shape (10, 299, 1, 120, 3)
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
    dkk_reshaped = dkk.reshape(-1, 1, 1)  # shape (nQuadSteps-1, 1, 1)

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

    # Calculate scaling for Q-functions
    logk_grid2 = logk_grid * logk_grid  # Cache this computation
    scale_Q = 0.25 * logk_grid2 / jnp.pi ** 2

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

    # R-function uses r from kk[2] to kk[nquadSteps-1]
    fp_r = fkk[1:-1].reshape(-1, 1, 1)
    r_r = kk_grid[1:-1].reshape(-1, 1, 1) / logk_grid
    r2_r = r_r * r_r
    #psl_r = Pkk[1:-1].reshape(-1, 1, 1)
    psl_r = jnp.stack((Pkk[1:-1], Pkk_nw[1:-1]), axis=1)[:,:,None] # shape (nquadSteps-2, 2, 1)

    # Gauss-Legendre points in [-1, 1] (fixed limits for R-functions)
    x_r = xxR.reshape(-1, 1, 1, 1)
    w_r = wwR.reshape(-1, 1, 1, 1)
    x2_r = x_r * x_r
    y2_r = 1.0 + r2_r - 2.0 * r_r * x_r

    # R-function kernels
    AngleEvR = -x_r
    AngleEvR2 = AngleEvR * AngleEvR

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

    # Trapezoidal integration (functional version)
    dkk_r = dkk_reshaped[:-1].reshape(-1, 1, 1)

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


def calculate(
        kfuncs_in: KFunctionsIn,
        A: float, ApOverf0: float, CFD3: float, CFD3p: float, sigma2v: float
    ) -> KFunctionsOut:
    """JAX-accelerated k-functions calculator.

    This function accepts numpy arrays (via KFunctionsIn), converts them to JAX arrays,
    runs the JIT-compiled calculation, and converts results back to numpy arrays.

    Args:
        kfuncs_in: Input data structure (uses numpy arrays from util.init_kfunctions)
        A, ApOverf0, CFD3, CFD3p, sigma2v: Scalar parameters

    Returns:
        KFunctionsOut containing numpy arrays (compatible with calculate_numpy output)
    """
    # Convert numpy arrays to JAX arrays with explicit float64 dtype
    # This is critical to match NumPy precision and avoid numerical differences
    k_in_jax = jnp.asarray(kfuncs_in.k_in, dtype=jnp.float64)
    logk_grid_jax = jnp.asarray(kfuncs_in.logk_grid, dtype=jnp.float64)
    kk_grid_jax = jnp.asarray(kfuncs_in.kk_grid, dtype=jnp.float64)
    Y_jax = jnp.asarray(kfuncs_in.Y, dtype=jnp.float64)
    Y2_jax = jnp.asarray(kfuncs_in.Y2, dtype=jnp.float64)
    xxQ_jax = jnp.asarray(kfuncs_in.xxQ, dtype=jnp.float64)
    wwQ_jax = jnp.asarray(kfuncs_in.wwQ, dtype=jnp.float64)
    xxR_jax = jnp.asarray(kfuncs_in.xxR, dtype=jnp.float64)
    wwR_jax = jnp.asarray(kfuncs_in.wwR, dtype=jnp.float64)

    # Run JIT-compiled calculation
    results = _calculate_jax_core(
        k_in_jax, logk_grid_jax, kk_grid_jax, Y_jax, Y2_jax,
        xxQ_jax, wwQ_jax, xxR_jax, wwR_jax,
        A, ApOverf0, CFD3, CFD3p, sigma2v
    )

    # Convert JAX arrays back to numpy arrays
    results_np = tuple(np.asarray(r) for r in results)

    return KFunctionsOut(*results_np)
