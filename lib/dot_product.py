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

from type_neuron import INF as infinity
from snn import TwoLayerSNN, SNN

class ScaleDatabase:

    def __init__(self):
        self.normalizer = Normalizer()
        pass

    ## Input: dbVectors: 2D np.ndarray
    ## Output: dbVectors: 2D np.ndarray
    def mean_center_normalize(self, dbVectors: np.ndarray) -> np.ndarray:

        mu1 = np.mean(dbVectors,axis=0)

        dbVectors_centered = np.subtract(dbVectors,mu1)

        dbVectors_norm = self.normalizer.fit_transform(dbVectors_centered)

        return dbVectors_norm

    def augment_0(self, dbVectors: np.ndarray) -> np.ndarray:
        return dbVectors
    
    def augment_positive(self, positive_value: float, dbVectors: np.ndarray) -> np.ndarray:
        mean_center_norm_positive = self.mean_center_normalize(dbVectors) + positive_value
        rslt = self.normalizer.fit_transform(mean_center_norm_positive) 
        assert np.all(rslt >= 0), "Result contains negative values"
        return rslt
    


    def augment_positive_rescale_0_1_timesteps(self, positive_value: float, timesteps: int, dbVectors: np.ndarray) -> np.ndarray:
        int_vec = self.mean_center_normalize(dbVectors) + math.sqrt(positive_value)
        return int_vec/np.max(int_vec) * timesteps

class ScaleQuery:
    def __init__(self, mu1):
        self.mu1 = mu1
        self.normalizer = Normalizer()
    
    ## Input: queryVector: 1D np.ndarray
    ## Output: queryVector: 1D np.ndarray
    def mean_center_normalize(self, queryVector: np.ndarray) -> np.ndarray:

        queryVector_centered = np.subtract(queryVector, self.mu1)

        queryVector_norm = self.normalizer.fit_transform(queryVector_centered)

        return queryVector_norm
    
    def augment_0(self, queryVector: np.ndarray) -> np.ndarray:
        return queryVector
    
    def augment_positive(self, positive_value: float, queryVector: np.ndarray) -> np.ndarray:
        mean_center_norm_positive = self.mean_center_normalize(queryVector) + positive_value
        rslt = self.normalizer.fit_transform(mean_center_norm_positive)
        assert np.all(rslt >= 0), "Result contains negative values"
        return rslt
    
    def augment_positive_rescale_0_1_timesteps(self, positive_value: float, timesteps: int, queryVector: np.ndarray) -> np.ndarray:
        int_vec = self.mean_center_normalize(queryVector + positive_value)
        return int_vec/np.max(int_vec) * timesteps

class DotProduct:

    def __init__(self, dbVectors, queryVectors):
        self.dbVectors = dbVectors
        self.queryVectors = queryVectors
        self.scale_db = ScaleDatabase()
        self.scale_query = ScaleQuery(np.mean(dbVectors,axis=0))

class CPUDotProduct(DotProduct):

    def __init__(self, dbVectors, queryVectors):
        super().__init__(dbVectors, queryVectors)

    def run(self):
        scaled_dbVectors = self.scale_db.mean_center_normalize(self.dbVectors)
        scaled_queryVectors = self.scale_query.mean_center_normalize(self.queryVectors) 
        
        return cdist(scaled_queryVectors, scaled_dbVectors, 'cosine')
    
class CPUDotProductPositive(DotProduct):

    def __init__(self, dbVectors, queryVectors, positive_value: float):
        super().__init__(dbVectors, queryVectors)
        self.positive_value = positive_value

    def run(self):
        transform_dbVectors    = self.scale_db.augment_positive(self.positive_value, self.dbVectors)
        transform_queryVectors = self.scale_query.augment_positive(self.positive_value, self.queryVectors)

        return cdist(transform_queryVectors, transform_dbVectors, 'cosine')
    
class CPUDotProductPositiveMIN(DotProduct):

    def __init__(self, dbVectors, queryVectors):
        super().__init__(dbVectors, queryVectors)
        self.positive_value = abs(min(np.min(dbVectors))) # we dont know the minimum of the query vectors
        return self.positive_value

    def run(self):
        transform_dbVectors    = self.scale_db.augment_positive(self.positive_value, self.dbVectors)
        transform_queryVectors = self.scale_query.augment_positive(self.positive_value, self.queryVectors)

        return cdist(transform_queryVectors, transform_dbVectors, 'cosine')

class LoihiDotProduct(DotProduct):
    def __init__(self, dbVectors, queryVectors):
        super().__init__(dbVectors, queryVectors)
        self.run_config = None

class LoihiDotProductSimulationPositive(LoihiDotProduct):

    def __init__(self, dbVectors, queryVectors, positive_value: float, timesteps=50, INF=infinity):
        super().__init__(dbVectors, queryVectors)
        self.run_config = Loihi1SimCfg(select_tag='fixed_pt')
        # self.run_config = Loihi1SimCfg(select_tag='floating_pt')
        self.timesteps = timesteps
        self.INF = infinity
        self.network = TwoLayerSNN()
        self.positive_value = positive_value

    def run(self):

        results = []

        transform_dbVectors = self.scale_db.augment_positive_rescale_0_1_timesteps(self.positive_value, self.timesteps, self.dbVectors)
        transform_queryVectors = self.scale_query.augment_positive_rescale_0_1_timesteps(self.positive_value, self.timesteps, self.queryVectors)
        
        queryVector = transform_queryVectors[0]
        # print(queryVector[:10])
        lif_1, dense, lif_2 = self.network.create_network(self.timesteps, queryVector, transform_dbVectors)
        
        lif_2.run(condition=RunSteps(num_steps=self.timesteps), run_cfg=self.run_config)

        for i, queryVector in enumerate(tqdm(transform_queryVectors)):
            
            init_v = [ -(self.timesteps  - elem - 1) for elem in queryVector]
            # print(init_v[:5])
            self.network.update_network(queryVector)
            # print(lif_1.v.get()[:5])

            lif_2.run(condition=RunSteps(num_steps=self.timesteps), run_cfg=self.run_config)
            
            aproxDotProduct = lif_2.u.get()/( 2**6 )

            # print(lif_2.u.get()[:10])

            cosine_dist = 1 - aproxDotProduct
        
            results.append(cosine_dist)
        
        lif_2.stop()
        
        return np.array(results)
 

class LoihiDotProductHardware(LoihiDotProduct):

    def __init__(self, dbVectors, queryVector):
        # Loihi2.preferred_partition = 'oheogulch'
        Loihi2.preferred_partition = 'oheogulch_2h'
        # Loihi2.preferred_partition = 'oheogulch_20m'
        loihi2_is_available = Loihi2.is_loihi2_available

        if loihi2_is_available:
            from lava.utils import loihi2_profiler
            print(f'Running on {Loihi2.partition}')
            compression = io.encoder.Compression.DELTA_SPARSE_8
        else:
            RuntimeError("Loihi2 compiler is not available in this system. "
            "This tutorial cannot proceed further.")

        super().__init__(dbVectors, queryVector)
        self.run_config = Loihi2HwCfg()
  