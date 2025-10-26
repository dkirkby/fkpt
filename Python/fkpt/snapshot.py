from dataclasses import dataclass

import h5py

from fkpt.types import Float64NDArray


@dataclass
class CosmologyParams:
    """Cosmological parameters"""
    Om: float
    h: float
    zout: float
    f0: float      # Growth rate at k→0
    Dplus: float   # Growth factor D+(z)/D+(0)

@dataclass
class ModelParams:
    """Model parameters"""
    mgmodel: str
    fR0: float

@dataclass
class KGridParams:
    """Output k-grid parameters"""
    kmin: float
    kmax: float
    Nk: int

@dataclass
class NumericalParams:
    """Numerical integration parameters"""
    nquadSteps: int

@dataclass
class KernelConstants:
    """LCDM kernel constants"""
    KA_LCDM: float
    KAp_LCDM: float
    KB_LCDM: float
    KR1_LCDM: float
    KR1p_LCDM: float

@dataclass
class SigmaValues:
    """Variance and damping integrals"""
    sigma8: float
    sigma2L: float
    sigma2v: float
    Sigma2: float
    deltaSigma2: float

@dataclass
class LinearPowerSpectrum:
    """Input linear power spectrum"""
    k: Float64NDArray      # k values [h/Mpc]
    P: Float64NDArray      # P(k) [(Mpc/h)³]
    Ppp: Float64NDArray    # P''(k) second derivative
    f: Float64NDArray      # f(k) growth rate

@dataclass
class KFunctions:
    """Output k-functions (27 arrays)"""
    k: Float64NDArray
    # P22 components
    P22dd: Float64NDArray
    P22du: Float64NDArray
    P22uu: Float64NDArray
    # P13 components
    P13dd: Float64NDArray
    P13du: Float64NDArray
    P13uu: Float64NDArray
    # RSD A-terms
    I1udd1A: Float64NDArray
    I2uud1A: Float64NDArray
    I2uud2A: Float64NDArray
    I3uuu2A: Float64NDArray
    I3uuu3A: Float64NDArray
    # RSD D-terms (B+C-G)
    I2uudd1BpC: Float64NDArray
    I2uudd2BpC: Float64NDArray
    I3uuud2BpC: Float64NDArray
    I3uuud3BpC: Float64NDArray
    I4uuuu2BpC: Float64NDArray
    I4uuuu3BpC: Float64NDArray
    I4uuuu4BpC: Float64NDArray
    # Bias terms
    Pb1b2: Float64NDArray
    Pb1bs2: Float64NDArray
    Pb22: Float64NDArray
    Pb2s2: Float64NDArray
    Ps22: Float64NDArray
    Pb2theta: Float64NDArray
    Pbs2theta: Float64NDArray
    # Additional
    sigma32PSL: Float64NDArray
    pkl: Float64NDArray    # Linear P(k) on output grid
    fk: Float64NDArray     # f(k) on output grid

@dataclass
class KFunctionsSnapshot:
    """Complete snapshot data"""
    # Metadata
    timestamp: str
    command_line: str
    # Parameters
    cosmology: CosmologyParams
    model: ModelParams
    k_grid: KGridParams
    numerical: NumericalParams
    kernels: KernelConstants
    # Sigma values
    sigma_values: SigmaValues
    # Inputs
    ps_wiggle: LinearPowerSpectrum
    ps_nowiggle: LinearPowerSpectrum
    # Outputs
    kfuncs_wiggle: KFunctions
    kfuncs_nowiggle: KFunctions


def load_snapshot(filename: str) -> KFunctionsSnapshot:
    """Load complete k-functions snapshot from HDF5 file."""
    with h5py.File(filename, 'r') as f:
        # Metadata
        timestamp = f['/metadata'].attrs['timestamp'].decode('utf-8')
        command_line = f['/metadata'].attrs['command_line'].decode('utf-8')

        # Parameters - Cosmology
        cosmology = CosmologyParams(
            Om=f['/parameters/cosmology/Om'][0],
            h=f['/parameters/cosmology/h'][0],
            zout=f['/parameters/cosmology/zout'][0],
            f0=f['/parameters/cosmology/f0'][0],
            Dplus=f['/parameters/cosmology/Dplus'][0]
        )

        # Parameters - Model
        model = ModelParams(
            mgmodel=f['/parameters/model/mgmodel'][0].decode('utf-8'),
            fR0=f['/parameters/model/fR0'][0]
        )

        # Parameters - K-grid
        k_grid = KGridParams(
            kmin=f['/parameters/k_grid/kmin'][0],
            kmax=f['/parameters/k_grid/kmax'][0],
            Nk=f['/parameters/k_grid/Nk'][0]
        )

        # Parameters - Numerical
        numerical = NumericalParams(
            nquadSteps=f['/parameters/numerical/nquadSteps'][0]
        )

        # Parameters - Kernels
        kernels = KernelConstants(
            KA_LCDM=f['/parameters/kernels/KA_LCDM'][0],
            KAp_LCDM=f['/parameters/kernels/KAp_LCDM'][0],
            KB_LCDM=f['/parameters/kernels/KB_LCDM'][0],
            KR1_LCDM=f['/parameters/kernels/KR1_LCDM'][0],
            KR1p_LCDM=f['/parameters/kernels/KR1p_LCDM'][0]
        )

        # Sigma values
        sigma_values = SigmaValues(
            sigma8=f['/sigma_values/sigma8'][0],
            sigma2L=f['/sigma_values/sigma2L'][0],
            sigma2v=f['/sigma_values/sigma2v'][0],
            Sigma2=f['/sigma_values/Sigma2'][0],
            deltaSigma2=f['/sigma_values/deltaSigma2'][0]
        )

        # Input power spectra
        ps_wiggle_data = f['/inputs/linear_ps_wiggle'][:]
        ps_wiggle = LinearPowerSpectrum(
            k=ps_wiggle_data[:, 0],
            P=ps_wiggle_data[:, 1],
            Ppp=ps_wiggle_data[:, 2],
            f=ps_wiggle_data[:, 3]
        )

        ps_nowiggle_data = f['/inputs/linear_ps_nowiggle'][:]
        ps_nowiggle = LinearPowerSpectrum(
            k=ps_nowiggle_data[:, 0],
            P=ps_nowiggle_data[:, 1],
            Ppp=ps_nowiggle_data[:, 2],
            f=ps_nowiggle_data[:, 3]
        )

        # Helper to load k-functions
        def load_kfunctions(group_path: str) -> KFunctions:
            g = f[group_path]
            return KFunctions(
                k=g['k'][:],
                P22dd=g['P22dd'][:], P22du=g['P22du'][:], P22uu=g['P22uu'][:],
                P13dd=g['P13dd'][:], P13du=g['P13du'][:], P13uu=g['P13uu'][:],
                I1udd1A=g['I1udd1A'][:], I2uud1A=g['I2uud1A'][:], I2uud2A=g['I2uud2A'][:],
                I3uuu2A=g['I3uuu2A'][:], I3uuu3A=g['I3uuu3A'][:],
                I2uudd1BpC=g['I2uudd1BpC'][:], I2uudd2BpC=g['I2uudd2BpC'][:],
                I3uuud2BpC=g['I3uuud2BpC'][:], I3uuud3BpC=g['I3uuud3BpC'][:],
                I4uuuu2BpC=g['I4uuuu2BpC'][:], I4uuuu3BpC=g['I4uuuu3BpC'][:],
                I4uuuu4BpC=g['I4uuuu4BpC'][:],
                Pb1b2=g['Pb1b2'][:], Pb1bs2=g['Pb1bs2'][:], Pb22=g['Pb22'][:],
                Pb2s2=g['Pb2s2'][:], Ps22=g['Ps22'][:],
                Pb2theta=g['Pb2theta'][:], Pbs2theta=g['Pbs2theta'][:],
                sigma32PSL=g['sigma32PSL'][:],
                pkl=g['pkl'][:], fk=g['fk'][:]
            )

        kfuncs_wiggle = load_kfunctions('/outputs/kfunctions_wiggle')
        kfuncs_nowiggle = load_kfunctions('/outputs/kfunctions_nowiggle')

    return KFunctionsSnapshot(
        timestamp=timestamp,
        command_line=command_line,
        cosmology=cosmology,
        model=model,
        k_grid=k_grid,
        numerical=numerical,
        kernels=kernels,
        sigma_values=sigma_values,
        ps_wiggle=ps_wiggle,
        ps_nowiggle=ps_nowiggle,
        kfuncs_wiggle=kfuncs_wiggle,
        kfuncs_nowiggle=kfuncs_nowiggle
    )
