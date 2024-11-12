from scipy.io import loadmat, savemat
from scipy.linalg import orth
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import gc

# Import Process level primitives
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort

# Import parent classes for ProcessModels
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.model.py.model import PyLoihiProcessModel

# Import ProcessModel ports, data-types
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType

# Import execution protocol and hardware resources
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.resources import CPU

# Import decorators
from lava.magma.core.decorator import implements, requires

# Import MNIST dataset
from lava.utils.dataloader.mnist import MnistDataset
np.set_printoptions(linewidth=np.inf)

from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg, Loihi2HwCfg

import math
import logging
import numpy as np
import matplotlib.pyplot as plt

from lava.magma.core.run_configs import Loihi2HwCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
# from lava.magma.compiler.subcompilers.nc.ncproc_compiler import CompilerOptions
from lava.utils.profiler import Profiler

np.set_printoptions(linewidth=110)  # Increase the line lenght of output cells.

from lava.utils.system import Loihi2
from lava.proc import io
from lava.magma.core.run_configs import Loihi2SimCfg, Loihi2HwCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc import io

from lava.proc.monitor.process import Monitor
import logging
import numpy as np
import matplotlib.pyplot as plt

from lava.magma.core.run_configs import Loihi2HwCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
# from lava.magma.compiler.subcompilers.nc.ncproc_compiler import CompilerOptions
from lava.utils.profiler import Profiler

np.set_printoptions(linewidth=110)  # Increase the line lenght of output cells.

from lava.utils.system import Loihi2
from lava.proc import io
from lava.magma.core.run_configs import Loihi2HwCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc import io

from type_neuron import NeuronType, INF as infinity

class SNN:
    def __init__(self):
        self.timesteps = None
        self.n_vec = None
        self.dims = None

    def create_network(self, timesteps, scaled_queryVector, scaled_dbVectors):
        pass

    def update_network(self, queryVector):
        pass

class TwoLayerSNN(SNN):

    lif_1 = None
    dense = None
    lif_2 = None

    def __init__(self):
        super().__init__()

    def create_network(self, timesteps, scaled_queryVector, scaled_dbVectors):

        if scaled_queryVector.shape[0] != scaled_dbVectors.shape[1]:
            assert("Dims do not match!")

        ## Initialize parameters in the Network
        self.timesteps = timesteps
        self.n_vec = scaled_dbVectors.shape[0]
        self.dims  = scaled_dbVectors.shape[1]

        
        ## Initialize LIF neuron parameters
        init_v = [ -(self.timesteps - elem - 1) for elem in scaled_queryVector]
        weights = np.array(scaled_dbVectors).reshape(self.n_vec,self.dims).astype(float)

        ## Initialize LIF neurons
        TwoLayerSNN.lif_1   = LIF(shape=(self.dims,), bias_mant=1, vth=0, v=np.round(init_v).astype(int))
        TwoLayerSNN.dense   = Dense(weights=np.round(weights).astype(int))
        TwoLayerSNN.lif_2   = LIF(shape=(self.n_vec,), vth=infinity, bias_mant=0, du=0)

        ## Connect LIF neurons
        TwoLayerSNN.lif_1.s_out.connect(TwoLayerSNN.dense.s_in)
        TwoLayerSNN.dense.a_out.connect(TwoLayerSNN.lif_2.a_in)

        return TwoLayerSNN.lif_1, TwoLayerSNN.dense, TwoLayerSNN.lif_2
        
    def update_network(self, scaled_queryVector):

        init_v = [ -(self.timesteps  - elem - 1) for elem in scaled_queryVector]
        TwoLayerSNN.lif_1.u.set(np.zeros(self.dims).astype(int))
        TwoLayerSNN.lif_1.v.set(np.round(init_v).astype(int))

        TwoLayerSNN.lif_2.u.set(np.zeros(self.n_vec).astype(int))
        TwoLayerSNN.lif_2.v.set(np.zeros(self.n_vec).astype(int))

    