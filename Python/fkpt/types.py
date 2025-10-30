from typing import NamedTuple, Type
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

Float64NDArray = NDArray[np.float64]  # Float64NDArray[float] in python 3.10+

class KFunctionsInitData(NamedTuple):
    k_in: Float64NDArray
    logk_grid: Float64NDArray
    kk_grid: Float64NDArray
    xxQ: Float64NDArray
    wwQ: Float64NDArray
    xxR: Float64NDArray
    wwR: Float64NDArray

class KFunctionsOut(NamedTuple):
    P22dd: Float64NDArray
    P22du: Float64NDArray
    P22uu: Float64NDArray
    I1udd1A: Float64NDArray
    I2uud1A: Float64NDArray
    I2uud2A: Float64NDArray
    I3uuu2A: Float64NDArray
    I3uuu3A: Float64NDArray
    I2uudd1BpC: Float64NDArray
    I2uudd2BpC: Float64NDArray
    I3uuud2BpC: Float64NDArray
    I3uuud3BpC: Float64NDArray
    I4uuuu2BpC: Float64NDArray
    I4uuuu3BpC: Float64NDArray
    I4uuuu4BpC: Float64NDArray
    Pb1b2: Float64NDArray
    Pb1bs2: Float64NDArray
    Pb22: Float64NDArray
    Pb2s2: Float64NDArray
    Ps22: Float64NDArray
    Pb2theta: Float64NDArray
    Pbs2theta: Float64NDArray
    P13dd: Float64NDArray
    P13du: Float64NDArray
    P13uu: Float64NDArray
    sigma32PSL: Float64NDArray
    pkl: Float64NDArray

class AbsCalculator(ABC):
    """Abstract base class for k-functions calculators."""

    @abstractmethod
    def initialize(self, data: KFunctionsInitData) -> None:
        """Initialize the calculator with grid data and quadrature points.

        Args:
            data: Initialization data containing k-grid, quadrature points, etc.
        """
        pass

    @abstractmethod
    def evaluate(self, Pk_in: Float64NDArray, Pk_nw_in: Float64NDArray, fk_in: Float64NDArray,
                 A: float, ApOverf0: float, CFD3: float, CFD3p: float, sigma2v: float, f0: float) -> KFunctionsOut:
        """Evaluate k-functions given input power spectra.

        Args:
            Pk_in: Linear power spectrum values at k-grid points
            Pk_nw_in: No-wiggle linear power spectrum values at k-grid points
            fk_in: Growth rate f(k) values at k-grid points
            A: Kernel constant A
            ApOverf0: Kernel constant Ap/f0
            CFD3: Kernel constant CFD3
            CFD3p: Kernel constant CFD3'
            sigma2v: Velocity dispersion sigma^2_v
            f0: Growth rate at z=0

        Returns:
            KFunctionsOut containing all computed k-functions
        """
        pass

KFunctionsCalculator = Type[AbsCalculator]
