import numpy as np

from fkpt.types import KFunctionsIn, KFunctionsOut, Float64NDArray
from fkpt.util import eval_cubic_spline


def calculate(
        kfuncs_in: KFunctionsIn,
        A: float, ApOverf0: float, CFD3: float, CFD3p: float, sigma2v: float
    ) -> KFunctionsOut:

    # Unpack inputs
    (
        k_in, logk_grid, kk_grid, Y, Y2,
        xxQ, wwQ, xxR, wwR,
    )  = kfuncs_in

    def interpolator(x: Float64NDArray) -> Float64NDArray:
        return eval_cubic_spline(k_in, Y, Y2, x)

    # Interpolate onto output grid
    Pout, Pout_nw, fout = interpolator(logk_grid).T

    # Interpolate onto quadrature grid
    Pkk, Pkk_nw, fkk = interpolator(kk_grid).T

    dkk = np.diff(kk_grid)

    # ============================================================================
    # Q-FUNCTIONS: Vectorized over ALL dimensions
    # ============================================================================

    # Compute variable integration limits for mu
    rmax = k_in[-1] / logk_grid  # shape (Nk,)
    rmin = k_in[0] / logk_grid  # shape (Nk,)
    rmax2 = rmax * rmax
    rmin2 = rmin * rmin

    # Loop over quadrature k values (line 378 in C)
    # fp shape: (nquadSteps-1, 1, 1) - will broadcast to (nquadSteps-1, 1, 1Nk)
    fp = fkk[1:].reshape(-1, 1, 1)

    # r shape: (nquadSteps-1, Nk)
    r = kk_grid[1:].reshape(-1, 1, 1) / logk_grid
    r2 = r ** 2

    # mumin, mumax: shape (nquadSteps-1, Nk)
    mumin = np.maximum(-1.0, (1.0 + r2 - rmax2) / (2.0 * r))
    mumax = np.minimum(1.0, (1.0 + r2 - rmin2) / (2.0 * r))

    # Line 389-390 in C: if r >= 0.5, mumax = 0.5/r
    mumax = np.divide(0.5, r, out=mumax, where=r >= 0.5)

    # Scale Gauss-Legendre nodes and weights to [mumin, mumax]
    # Shape: (NQ, nquadSteps-1, 1, Nk)
    dmu = mumax - mumin
    xGL = 0.5 * (dmu * xxQ.reshape(-1, 1, 1, 1) + (mumax + mumin))
    wGL = 0.5 * dmu * wwQ.reshape(-1, 1, 1, 1)

    # Perform Gauss-Legendre quadrature over mu (line 392)
    # All shapes: (NQ, nquadSteps-1, Nk)
    x = xGL
    w = wGL
    x2 = x * x
    y2 = 1.0 + r2 - 2.0 * r * x
    y = np.sqrt(y2)

    # Interpolate power spectra at (ki * y) points
    psl_w, psl_nw, fkmp = interpolator(logk_grid * y).T
    psl = np.concatenate((psl_w.T, psl_nw.T), axis=2) # shape (NQ, nquadSteps-1, 2, Nk)
    fkmp = fkmp.T # shape (NQ, nquadSteps-1, 1, Nk)

    # Compute SPT kernels F2evQ and G2evQ (lines 404-411)
    AngleEvQ = (x - r) / y
    AngleEvQ2 = AngleEvQ ** 2
    fsum = fp + fkmp

    S2evQ = AngleEvQ ** 2 - 1./3.
    F2evQ = (1.0/2.0 + 3.0/14.0 * A + (1.0/2.0 - 3.0/14.0 * A) * AngleEvQ2 +
             AngleEvQ / 2.0 * (y/r + r/y))
    G2evQ = (3.0/14.0 * A * fsum + 3.0/14.0 * ApOverf0 +
             (1.0/2.0 * fsum - 3.0/14.0 * A * fsum - 3.0/14.0 * ApOverf0) * AngleEvQ2 +
             AngleEvQ / 2.0 * (fkmp * y/r + fp * r/y))

    # Accumulate over mu dimension (sum along axis 0)
    # Shapes after summation: (nquadSteps-1, Nk)

    # Precompute some temporary expressions that are used multiple times
    wpsl = w * psl
    fkmpr2 = fkmp * r2
    rx = r * x
    y4 = y2 ** 2

    # P22 kernels (lines 414-416)
    P22dd_B = np.sum(wpsl * (2.0 * r2 * F2evQ**2), axis=0)
    P22du_B = np.sum(wpsl * (2.0 * r2 * F2evQ * G2evQ), axis=0)
    P22uu_B = np.sum(wpsl * (2.0 * r2 * G2evQ**2), axis=0)

    # ========== 5 THREE-POINT CORRELATION FUNCTION KERNELS (Q-part) ==========
    # Lines 420-429 in C code

    # I1udd1tA: Line 420
    I1udd1tA_B = np.sum(wpsl * (
        2.0 * (fp * rx + fkmpr2 * (1.0 - rx) / y2) * F2evQ
        ), axis=0)

    # I2uud1tA: Line 422
    I2uud1tA_B = np.sum(wpsl * (-fp * fkmpr2 * (1.0 - x2) / y2 * F2evQ), axis=0)

    # I2uud2tA: Lines 424-425
    I2uud2tA_B = np.sum(wpsl * (
        2.0 * (fp * rx + fkmpr2 * (1.0 - rx) / y2) * G2evQ
        + fp * fkmp * (r2 * (1.0 - 3.0 * x2) + 2.0 * rx) / y2 * F2evQ
        ), axis=0)

    # I3uuu2tA: Line 427
    I3uuu2tA_B = np.sum(wpsl * (fp * fkmpr2 * (x2 - 1.0) / y2 * G2evQ), axis=0)

    # I3uuu3tA: Line 429
    I3uuu3tA_B = np.sum(wpsl * (
        fp * fkmp * (r2 * (1.0 - 3.0 * x2) + 2.0 * rx) / y2 * G2evQ
        ), axis=0)

    # ========== 7 BpC TERM KERNELS (Q-part, will become D-terms) ==========
    # Lines 435-455 in C code

    # I2uudd1BpC: Lines 435-436
    I2uudd1BpC_B = np.sum(wpsl * (
        1.0 / 4.0 * (1.0 - x2) * (fp * fp + fkmpr2 ** 2 / y4)
        + fp * fkmpr2 * (-1.0 + x2) / y2 / 2.0
        ), axis=0)

    # I2uudd2BpC: Lines 438-441
    I2uudd2BpC_B = np.sum(wpsl * (
        (
            fp * fp * (-1.0 + 3.0 * x2)
            + 2.0 * fkmp * fp * r * (r + 2.0 * x - 3.0 * r * x2) / y2
            + fkmp * fkmpr2 * (2.0 - 4.0 * rx + r2 * (-1.0 + 3.0 * x2)) / y4
        )
        / 4.0
        ), axis=0)

    # I3uuud2BpC: Lines 443-444
    I3uuud2BpC_B = np.sum(wpsl * (
        -(
            fkmp * fp * (
                fkmp * (-2.0 + 3.0 * rx) * r2
                - fp * (-1.0 + 3.0 * rx) * (1.0 - 2.0 * rx + r2)
            )
            * (-1.0 + x2)
        )
        / (2.0 * y2 * y2)
        ), axis=0)

    # I3uuud3BpC: Lines 446-447
    I3uuud3BpC_B = np.sum(wpsl * (
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

    # I4uuuu2BpC: Line 449
    I4uuuu2BpC_B = np.sum(wpsl * (
        3.0 * fkmp**2 * fp**2 * r2 * (-1.0 + x2) ** 2 / (16.0 * y4)
        ), axis=0)

    # I4uuuu3BpC: Lines 451-452
    I4uuuu3BpC_B = np.sum(wpsl * (
        -(
            fkmp**2 * fp**2 * (-1.0 + x2) * (2.0 + 3.0 * r * (-4.0 * x + r * (-1.0 + 5.0 * x2)))
        )
        / (8.0 * y2 * y2)
        ), axis=0)

    # I4uuuu4BpC: Lines 454-455
    I4uuuu4BpC_B = np.sum(wpsl * (
        (
            fkmp**2 * fp**2 * (
                -4.0
                + 8.0 * rx * (3.0 - 5.0 * x2)
                + 12.0 * x2
                + r2 * (3.0 - 30.0 * x2 + 35.0 * x ** 4)
            )
        )
        / (16.0 * y4)
        ), axis=0)

    # Left and right endpoints for power spectra
    PSLB = np.stack((Pkk[1:], Pkk_nw[1:]), axis=1)[:,:,None] # shape (nQuadSteps-1, 2, 1)
    dkk_reshaped = dkk.reshape(-1, 1, 1)  # shape (nQuadSteps-1, 1, 1)

    # Bias
    Pb1b2_B = np.sum(wpsl * (r2 * F2evQ), axis=0)
    Pb1bs2_B = np.sum(wpsl * (r2 * F2evQ * S2evQ), axis=0)
    Pratio = PSLB / psl
    PratioInv = psl / PSLB
    Pb22_B = np.sum(wpsl * (
        1.0 / 2.0 * r2 * (1.0 / 2.0 * (1.0 - Pratio) + 1.0 / 2.0 * (1.0 - PratioInv))
        ), axis=0)
    Pb2s2_B = np.sum(wpsl * (
        1.0 / 2.0 * r2 * (
            1.0 / 2.0 * (S2evQ - 2.0 / 3.0 * Pratio)
            + 1.0 / 2.0 * (S2evQ - 2.0 / 3.0 * PratioInv)
        )
        ), axis=0)
    Ps22_B = np.sum(wpsl * (
        1.0 / 2.0 * r2
        * (
            1.0 / 2.0 * (S2evQ ** 2 - 4.0 / 9.0 * Pratio)
            + 1.0 / 2.0 * (S2evQ ** 2 - 4.0 / 9.0 * PratioInv)
        )
        ), axis=0)
    Pb2theta_B = np.sum(wpsl * (r2 * G2evQ), axis=0)
    Pbs2theta_B = np.sum(wpsl * (r2 * S2evQ * G2evQ), axis=0)

    # Calculate scaling for Q-functions (line 561-563, 565-569 in C)
    scale_Q = 0.25 * (logk_grid / np.pi) ** 2

    # Apply trapezoidal rule: sum of (f_A * P_A + f_B * P_B) * dk / 2
    #def trapsumQ_orig(B):
    #    A = np.vstack([np.zeros((1, Nk)), B[:-1, :]])
    #    return np.sum(dkk_reshaped * (A * PSLA + B * PSLB) / 2.0, axis=0) * scale_Q

    # Eliminate memory allocation for A and perform all array ops in place
    def trapsumQ(B):
        B *= scale_Q
        B *= PSLB
        B[1:] += B[:-1]
        B *= dkk_reshaped
        B[0] += np.sum(B[1:], axis=0)
        return B[0]

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
    # R-FUNCTIONS: Also fully vectorized (NR, nquadSteps-2, Nk) - note nquadSteps-2!
    # ============================================================================

    # Get f(k) at output k values (line 602-603 in C)
    fk = fout  # already computed above

    # R-function uses r from kk[2] to kk[nquadSteps-1] (see line 604: i=2 to nquadSteps-1)
    fp_r = fkk[1:-1].reshape(-1, 1)
    r_r = kk_grid[1:-1].reshape(-1, 1) / logk_grid
    r2_r = r_r ** 2
    psl_r = Pkk[1:-1].reshape(-1, 1)

    # Gauss-Legendre points in [-1, 1] (fixed limits for R-functions)
    x_r = xxR.reshape(-1, 1, 1)
    w_r = wwR.reshape(-1, 1, 1)
    x2_r = x_r * x_r
    y2_r = 1.0 + r2_r - 2.0 * r_r * x_r

    # R-function kernels (lines 618-636 in C)
    AngleEvR = -x_r
    AngleEvR2 = AngleEvR ** 2

    F2evR = (1.0/2.0 + 3.0/14.0 * A + (1.0/2.0 - 3.0/14.0 * A) * AngleEvR2 +
             AngleEvR / 2.0 * (1.0/r_r + r_r))
    G2evR = (3.0/14.0 * A * (fp_r + fk) + 3.0/14.0 * ApOverf0 +
             (1.0/2.0 * (fp_r + fk) - 3.0/14.0 * A * (fp_r + fk) -
              3.0/14.0 * ApOverf0) * AngleEvR2 +
             AngleEvR / 2.0 * (fk/r_r + fp_r * r_r))

    # ========== 5 THREE-POINT CORRELATION FUNCTION KERNELS (R-part) ==========
    # Lines 644-654 in C code
    # Accumulate over mu (sum along axis 0)
    # Shapes after summation: (nquadSteps-2, Nk)
    wpsl_r = w_r * psl_r

    Gamma2evR  = A *(1. - x2_r)
    Gamma2fevR = A *(1. - x2_r)*(fk + fp_r)/2. + 1./2. * ApOverf0 *(1 - x2_r)
    C3Gamma3  = 2.*5./21. * CFD3  *(1 - x2_r)*(1 - x2_r)/y2_r
    C3Gamma3f = 2.*5./21. * CFD3p *(1 - x2_r)*(1 - x2_r)/y2_r *(fk + 2 * fp_r)/3.
    G3K = (
        C3Gamma3f/ 2. + (2 * Gamma2fevR * x_r)/(7. * r_r) - (fk  * x2_r)/(6 * r2_r)
        + fp_r * Gamma2evR*(1 - r_r * x_r)/(7 * y2_r)
        - 1./7.*(fp_r * Gamma2evR + 2 *Gamma2fevR) * (1. - x2_r)/y2_r)
    F3K = C3Gamma3/6. - x2_r/(6 * r2_r) + (Gamma2evR * x_r *(1 - r_r * x_r))/(7. *r_r *y2_r)

    P13dd_B = np.sum(wpsl_r * (6.* r2_r * F3K), axis=0)
    P13du_B = np.sum(wpsl_r * (3.* r2_r * G3K + 3.* r2_r * F3K * fk), axis=0)
    P13uu_B = np.sum(wpsl_r * (6.* r2_r * G3K * fk), axis=0)

    sigma32PSL_B = np.sum(wpsl_r * (
        ( 5.0* r2_r * (7. - 2*r2_r + 4*r_r*x_r + 6*(-2 + r2_r)*x2_r - 12*r_r*x2_r*x_r + 9*x2_r*x2_r))
        / (24.0 * y2_r)
    ), axis=0)

    # I1udd1a: Line 644
    I1udd1a_B = np.sum(wpsl_r * (
        2.0 * r2_r * (1.0 - r_r * x_r) / y2_r * G2evR + 2.0 * fp_r * r_r * x_r * F2evR
    ), axis=0)

    # I2uud1a: Line 646
    I2uud1a_B = np.sum(wpsl_r * (
        -fp_r * r2_r * (1.0 - x2_r) / y2_r * G2evR
    ), axis=0)

    # I2uud2a: Lines 648-649
    I2uud2a_B = np.sum(wpsl_r * (
        ((r2_r * (1.0 - 3.0 * x2_r) + 2.0 * r_r * x_r) / y2_r * fp_r +
                fk * 2.0 * r2_r * (1.0 - r_r * x_r) / y2_r) * G2evR + 2.0 * x_r * r_r * fp_r * fk * F2evR
    ), axis=0)

    # I3uuu2a: Line 652
    # Note this is similar to I2uud1a_B but with additional factor of fk inside the sum
    I3uuu2a_B = np.sum(wpsl_r * (
        -fp_r * r2_r * (1.0 - x2_r) / y2_r * G2evR * fk
    ), axis=0)

    # I3uuu3a: Line 654
    I3uuu3a_B = np.sum(wpsl_r * (
        (r2_r * (1.0 - 3.0 * x2_r) + 2.0 * r_r * x_r) / y2_r * fp_r * fk * G2evR
    ), axis=0)

    # Calculate scaling for R-functions (line 699-707 in C)
    pkl_k = np.vstack([Pout, Pout_nw])  # shape (2, Nk)
    scale_R = logk_grid ** 2 / (8.0 * np.pi ** 2) * pkl_k

    # Trapezoidal integration (line 679-683 in C)
    dkk_r = dkk_reshaped[:-1].reshape(-1, 1, 1)

    def trapsumR(B):
        B = B[:,None,:] * scale_R
        B[1:] += B[:-1]
        B = B * dkk_r
        B[0] += np.sum(B[1:], axis=0)
        return B[0]

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
    # Combine Q and R functions (line 717-721 in C)
    # ============================================================================
    I1udd1A = I1udd1tA + 2.0 * I1udd1a
    I2uud1A = I2uud1tA + 2.0 * I2uud1a
    I2uud2A = I2uud2tA + 2.0 * I2uud2a
    I3uuu2A = I3uuu2tA + 2.0 * I3uuu2a
    I3uuu3A = I3uuu3tA + 2.0 * I3uuu3a

    # ============================================================================
    # D-TERMS (B + C - G corrections)
    # Lines 723-735 in C code
    # ============================================================================
    # Note: BpC terms already calculated from Q-functions above
    # Now apply G-corrections (sigma2v damping) to specific terms
    fk_grid = fk  # Already normalized by f0

    # I2uudd1D: Line 723-724 (subtract k^2 * sigma2v * P_L(k))
    I2uudd1BpC = I2uudd1BpC - logk_grid**2 * sigma2v * pkl_k

    # I3uuud2D: Line 727-728 (subtract 2 * k^2 * sigma2v * f(k) * P_L(k))
    I3uuud2BpC = I3uuud2BpC - 2.0 * logk_grid**2 * sigma2v * fk_grid * pkl_k

    # I4uuuu3D: Line 732-733 (subtract k^2 * sigma2v * f(k)^2 * P_L(k))
    I4uuuu3BpC = I4uuuu3BpC - logk_grid**2 * sigma2v * fk_grid**2 * pkl_k

    return KFunctionsOut(
        P22dd, P22du, P22uu,
        I1udd1A, I2uud1A, I2uud2A,
        I3uuu2A, I3uuu3A,
        I2uudd1BpC, I2uudd2BpC,
        I3uuud2BpC, I3uuud3BpC,
        I4uuuu2BpC, I4uuuu3BpC, I4uuuu4BpC,
        Pb1b2, Pb1bs2, Pb22, Pb2s2, Ps22,
        Pb2theta, Pbs2theta,
        P13dd, P13du, P13uu,
        sigma32PSL
    )
