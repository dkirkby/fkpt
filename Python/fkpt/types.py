from typing import NamedTuple, Callable

import numpy as np
from numpy.typing import NDArray

Float64NDArray = NDArray[np.float64]  # Float64NDArray[float] in python 3.10+

KInterpolator = Callable[[Float64NDArray], Float64NDArray]

class KFunctionsIn(NamedTuple):
    k_in: Float64NDArray
    logk_grid: Float64NDArray
    kk_grid: Float64NDArray
    Y: Float64NDArray
    Y2: Float64NDArray
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

# Args are (KFunctionsIn, A, ApOverf0, CFD3, CFD3p, sigma2v) -> KFunctionsOut
KFunctionsCalculator = Callable[[KFunctionsIn, float, float, float, float, float], KFunctionsOut]
