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
from snn import TwoLayerSNN, ThreeLayerSNN#, ThreeLayerSNNRateEncoded

debug = False
def debug_print(args, str_lst=[]):
    global debug
    if debug:
        for arg in args:
            if str_lst != []:
                print(f'{str_lst[0]} :\n{arg}')
                str_lst.pop(0)
            else:
                print(arg)

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
        # int_vec = self.mean_center_normalize(dbVectors) + (positive_value)
        # return int_vec/np.max(int_vec) * 10
        mean_center_norm_positive = self.mean_center_normalize(dbVectors) + positive_value
        rslt = self.normalizer.fit_transform(mean_center_norm_positive) 
        assert np.all(rslt >= 0), "Result contains negative values"
        return rslt * timesteps
    
    def augment_positive_based_on_q0(self, positive_value: float, timesteps: int, queryVector: np.ndarray) -> np.ndarray:
        mean_center_norm_positive = self.mean_center_normalize(queryVector) + positive_value
        rslt = self.normalizer.fit_transform(mean_center_norm_positive)
        assert np.all(rslt >= 0), "Result contains negative values"
        max_q = np.max(rslt)
        min_q = np.min(rslt)
        return (max_q - rslt) * timesteps / (max_q - min_q)
    
class ScaleQuery:
    def __init__(self, mu1):
        self.mu1 = mu1
        self.normalizer = Normalizer()
    
    ## Input: queryVector: 1D np.ndarray
    ## Output: queryVector: 1D np.ndarray
    def mean_center_normalize(self, queryVector: np.ndarray) -> np.ndarray:
        debug_print([queryVector], ['queryVector'])
        debug_print([self.mu1], ['self.mu1'])
        queryVector_centered = np.subtract(queryVector, self.mu1)
        debug_print([queryVector_centered], ['queryVector_centered'])
        queryVector_norm = self.normalizer.fit_transform(queryVector_centered)
        debug_print([queryVector_norm], ['queryVector_norm'])
        return queryVector_norm
    
    def augment_0(self, queryVector: np.ndarray) -> np.ndarray:
        return queryVector
    
    def augment_positive(self, positive_value: float, queryVector: np.ndarray) -> np.ndarray:
        mean_center_norm_positive = self.mean_center_normalize(queryVector) + positive_value
        rslt = self.normalizer.fit_transform(mean_center_norm_positive)
        assert np.all(rslt >= 0), "Result contains negative values"
        return rslt
    
    def augment_positive_rescale_0_1_timesteps(self, positive_value: float, timesteps: int, queryVector: np.ndarray) -> np.ndarray:
        # int_vec = self.mean_center_normalize(queryVector) + positive_value
        # return int_vec/np.max(int_vec) * timesteps
        mean_center_norm_positive = self.mean_center_normalize(queryVector) + positive_value
        rslt = self.normalizer.fit_transform(mean_center_norm_positive)
        assert np.all(rslt >= 0), "Result contains negative values"
        debug_print([rslt], ['rslt'])
        debug_print([rslt*timesteps], ['rslt*timesteps'])
        return rslt*timesteps
    
    def augment_positive_based_on_q0(self, positive_value: float, timesteps: int, queryVector: np.ndarray) -> np.ndarray:
        # int_vec = self.mean_center_normalize(queryVector) + positive_value
        # return int_vec/np.max(int_vec) * timesteps
        debug_print([queryVector], ['queryVector'])
        debug_print([queryVector], ['queryVector'])
        debug_print([positive_value], ['positive_value'])
        mean_center_norm_positive = self.mean_center_normalize(queryVector) + positive_value
        debug_print([mean_center_norm_positive], ['mean_center_norm_positive'])
        rslt = self.normalizer.fit_transform(mean_center_norm_positive)
        assert np.all(rslt >= 0), "Result contains negative values"
        debug_print([rslt], ['rslt'])
        max_q0 = np.max(rslt[0])
        min_q0 = np.min(rslt[0])
        debug_print([max_q0, min_q0], ['max_q0', 'min_q0'])
        return (max_q0 - rslt) * timesteps / (max_q0 - min_q0)

class DotProduct:

    def __init__(self, dbVectors, queryVectors):
        self.dbVectors = dbVectors
        self.queryVectors = queryVectors
        self.scale_db = ScaleDatabase()
        self.scale_query = ScaleQuery(np.mean(dbVectors,axis=0))

    def run(self):
        pass

    def compute_similarity(self, D_db: np.ndarray, D_query: np.ndarray, **kwargs) -> np.ndarray:
        return self.run(D_db, D_query, **kwargs)

class CPUDotProduct(DotProduct):

    def __init__(self, dbVectors, queryVectors):
        super().__init__(dbVectors, queryVectors)

    def run(self, dbVectors=None, queryVectors=None):
        if dbVectors is not None:
            self.dbVectors = dbVectors
        if queryVectors is not None:
            self.queryVectors = queryVectors
        scaled_dbVectors = self.scale_db.mean_center_normalize(self.dbVectors)
        scaled_queryVectors = self.scale_query.mean_center_normalize(self.queryVectors) 
        debug_print([scaled_dbVectors, scaled_queryVectors], ['scaled_dbVectors', 'scaled_queryVectors'])
        return cdist(scaled_queryVectors, scaled_dbVectors, 'cosine')
    
class CPUDotProductPositive(DotProduct):

    def __init__(self, dbVectors, queryVectors, positive_value: float):
        super().__init__(dbVectors, queryVectors)
        self.positive_value = positive_value

    def run(self, dbVectors=None, queryVectors=None):
        if dbVectors is not None:
            self.dbVectors = dbVectors
        if queryVectors is not None:
            self.queryVectors = queryVectors
        transform_dbVectors    = self.scale_db.augment_positive(self.positive_value, self.dbVectors)
        transform_queryVectors = self.scale_query.augment_positive(self.positive_value, self.queryVectors)
        debug_print([transform_dbVectors, transform_queryVectors], ['transform_dbVectors', 'transform_queryVectors'])
        return -cdist(transform_queryVectors, transform_dbVectors, 'cosine').T
    
class CPUDotProductPositiveForLoop(DotProduct):

    def __init__(self, dbVectors, queryVectors, positive_value: float):
        super().__init__(dbVectors, queryVectors)
        self.positive_value = positive_value
        self.transform_dbVectors = None
        self.transform_queryVectors = None

    def pre_run(self, dbVectors=None, queryVectors=None):
        if dbVectors is not None:
            self.dbVectors = dbVectors
        if queryVectors is not None:
            self.queryVectors = queryVectors
        self.transform_dbVectors    = self.scale_db.augment_positive(self.positive_value, self.dbVectors)
        self.transform_queryVectors = self.scale_query.augment_positive(self.positive_value, self.queryVectors)
        debug_print([self.transform_dbVectors, self.transform_queryVectors], ['transform_dbVectors', 'transform_queryVectors'])

    def run(self, dbVectors=None, queryVectors=None):
        
        for j in range(len(self.transform_dbVectors)):
            qVector  = self.transform_queryVectors[0]
            dbVector = self.transform_dbVectors[j]
            # dotProduct = np.dot(self.transform_queryVectors[0], dbVector)
            dot_prod = 0
            for k in range(len(dbVector)):
                dot_prod += qVector[k] * dbVector[k]
    
    
class CPUDotProductPositiveMIN(DotProduct):

    def __init__(self, dbVectors, queryVectors):
        super().__init__(dbVectors, queryVectors)
        self.positive_value = abs(min(np.min(dbVectors))) # we dont know the minimum of the query vectors
        return self.positive_value

    def run(self, dbVectors=None, queryVectors=None):
        if dbVectors is not None:
            self.dbVectors = dbVectors
        if queryVectors is not None:
            self.queryVectors = queryVectors

        transform_dbVectors    = self.scale_db.augment_positive(self.positive_value, self.dbVectors)
        transform_queryVectors = self.scale_query.augment_positive(self.positive_value, self.queryVectors)
        debug_print([transform_dbVectors, transform_queryVectors], ['transform_dbVectors', 'transform_queryVectors'])
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

    def run(self, dbVectors=None, queryVectors=None):
        if dbVectors is not None:
            self.dbVectors = dbVectors
        if queryVectors is not None:
            self.queryVectors = queryVectors

        results = []

        transform_dbVectors = self.scale_db.augment_positive_based_on_q0(self.positive_value, self.timesteps, self.dbVectors)
        transform_queryVectors = self.scale_query.augment_positive_based_on_q0(self.positive_value, self.timesteps, self.queryVectors)
        debug_print([transform_dbVectors, transform_queryVectors], ['transform_dbVectors', 'transform_queryVectors'])
        queryVector = transform_queryVectors[0]
        # print(queryVector[:10])
        lif_1, dense, lif_2 = self.network.create_network(self.timesteps, queryVector, transform_dbVectors)
        
        lif_2.run(condition=RunSteps(num_steps=self.timesteps), run_cfg=self.run_config)

        for i, queryVector in enumerate(tqdm(transform_queryVectors)):
            
            init_v = [ -(self.timesteps  - elem - 1) for elem in queryVector]
            # debug_print([init_v], ['init_v'])
            # print(init_v[:5])
            self.network.update_network(queryVector)
            # print(lif_1.v.get()[:5])

            lif_2.run(condition=RunSteps(num_steps=self.timesteps), run_cfg=self.run_config)
            
            aproxDotProduct = lif_2.u.get()/( 2**6 )

            # print(lif_2.u.get()[:10])

            # cosine_dist = 1 - aproxDotProduct
        
            results.append(aproxDotProduct)
        
        lif_2.stop()
        
        return np.array(results), transform_queryVectors, transform_dbVectors
    
class LoihiDotProductSimulationPositiveDBScale(LoihiDotProduct):

    def __init__(self, dbVectors, queryVectors, positive_value: float, timesteps=50, dbScale=50, INF=infinity):
        super().__init__(dbVectors, queryVectors)
        self.run_config = Loihi1SimCfg(select_tag='fixed_pt')
        # self.run_config = Loihi1SimCfg(select_tag='floating_pt')
        self.timesteps = timesteps
        self.dbScale = dbScale
        self.INF = infinity
        self.network = TwoLayerSNN()
        self.positive_value = positive_value

    def run(self, dbVectors=None, queryVectors=None, monitor=False):
        if dbVectors is not None:
            self.dbVectors = dbVectors
            self.network = TwoLayerSNN()
            self.scale_query = ScaleQuery(np.mean(dbVectors,axis=0))
        if queryVectors is not None:
            self.queryVectors = queryVectors
            self.network = TwoLayerSNN()

        results = []

        transform_dbVectors = self.scale_db.augment_positive_based_on_q0(self.positive_value, self.dbScale, self.dbVectors)
        transform_queryVectors = self.scale_query.augment_positive_based_on_q0(self.positive_value, self.timesteps, self.queryVectors)
        debug_print([transform_dbVectors, transform_queryVectors], ['transform_dbVectors', 'transform_queryVectors'])
        queryVector = transform_queryVectors[0]
        # debug_print([queryVector[:10]], ['queryVector[:10]'])
        # debug_print([transform_dbVectors[0][:10]], ['transform_dbVectors[0][:10]'])
        lif_1, dense, lif_2 = self.network.create_network(self.timesteps, queryVector, transform_dbVectors)

        if monitor:
            mon_lif_1_v = Monitor()
            mon_lif_2_u = Monitor()
            mon_lif_2_v = Monitor()
            mon_spike_1 = Monitor()
            mon_spike_2 = Monitor()

            mon_lif_1_v.probe(lif_1.v,   self.timesteps)
            mon_lif_2_u.probe(lif_2.u,   self.timesteps)
            mon_lif_2_v.probe(lif_2.v,   self.timesteps)
            mon_spike_1.probe(lif_1.s_out, self.timesteps)
            mon_spike_2.probe(lif_2.s_out, self.timesteps)

            mon_lif_1_v_process = list(mon_lif_1_v.get_data())[0]
            mon_lif_2_u_process = list(mon_lif_2_u.get_data())[0]
            mon_lif_2_v_process = list(mon_lif_2_v.get_data())[0]
            mon_spike_1_process = list(mon_spike_1.get_data())[0]
            mon_spike_2_process = list(mon_spike_2.get_data())[0]
        
        lif_2.run(condition=RunSteps(num_steps=self.timesteps), run_cfg=self.run_config)

        if monitor:
            monitors = {
                'lif1_voltage': mon_lif_1_v.get_data()[mon_lif_1_v_process]["v"],
                'lif2_current': mon_lif_2_u.get_data()[mon_lif_2_u_process]["u"],
                'lif2_voltage': mon_lif_2_v.get_data()[mon_lif_2_v_process]["v"],
                'lif1_spikes': mon_spike_1.get_data()[mon_spike_1_process]["s_out"],
                'lif2_spikes': mon_spike_2.get_data()[mon_spike_2_process]["s_out"]
            }

            return transform_queryVectors, transform_dbVectors, monitors

        for i, queryVector in enumerate(tqdm(transform_queryVectors)):
            
            init_v = [ -(self.timesteps  - elem - 1) for elem in queryVector]
            # debug_print([init_v], ['init_v'])
            # print(init_v[:5])
            self.network.update_network(queryVector)
            # print(lif_1.v.get()[:5])

            lif_2.run(condition=RunSteps(num_steps=self.timesteps), run_cfg=self.run_config)
            
            aproxDotProduct = lif_2.u.get()

            # print(lif_2.u.get()[:10])

            # cosine_dist = 1 - aproxDotProduct
        
            results.append(aproxDotProduct)
        
        lif_2.stop()
        
        return np.array(results).T#, transform_queryVectors, transform_dbVectors
 
class LoihiDotProductSimulationPositive3Layer(LoihiDotProduct):

    def __init__(self, dbVectors, queryVectors, positive_value: float, timesteps=50, INF=infinity):
        super().__init__(dbVectors, queryVectors)
        self.run_config = Loihi1SimCfg(select_tag='fixed_pt')
        # self.run_config = Loihi1SimCfg(select_tag='floating_pt')
        self.timesteps = timesteps
        self.INF = infinity
        self.network = ThreeLayerSNN()
        self.positive_value = positive_value

    def run(self, dbVectors=None, queryVectors=None):
        if dbVectors is not None:
            self.dbVectors = dbVectors
        if queryVectors is not None:
            self.queryVectors = queryVectors

        results = []

        transform_dbVectors = self.scale_db.augment_positive_based_on_q0(self.positive_value, self.timesteps, self.dbVectors)
        transform_queryVectors = self.scale_query.augment_positive_based_on_q0(self.positive_value, self.timesteps, self.queryVectors)
        debug_print([transform_dbVectors, transform_queryVectors], ['transform_dbVectors', 'transform_queryVectors'])
        queryVector = transform_queryVectors[0]
        # print(queryVector[:10])
        lif_1, dense, lif_2, lif_3 = self.network.create_network(self.timesteps, queryVector, transform_dbVectors)

        # mon_lif_1_v = Monitor()
        # mon_lif_2_u = Monitor()
        # mon_lif_2_v = Monitor()
        # mon_lif_3_u = Monitor()
        # mon_spike_1 = Monitor()
        # mon_spike_2 = Monitor()

        # mon_lif_1_v.probe(lif_1.v,   self.timesteps)
        # mon_lif_2_u.probe(lif_2.u,   self.timesteps)
        # mon_lif_2_v.probe(lif_2.v,   self.timesteps)
        # mon_lif_3_u.probe(lif_3.u,   self.timesteps)
        # mon_spike_1.probe(lif_1.s_out, self.timesteps)
        # mon_spike_2.probe(lif_2.s_out, self.timesteps)

        # mon_lif_1_v_process = list(mon_lif_1_v.get_data())[0]
        # mon_lif_2_u_process = list(mon_lif_2_u.get_data())[0]
        # mon_lif_2_v_process = list(mon_lif_2_v.get_data())[0]
        # mon_lif_3_u_process = list(mon_lif_3_u.get_data())[0]
        # mon_spike_1_process = list(mon_spike_1.get_data())[0]
        # mon_spike_2_process = list(mon_spike_2.get_data())[0]


        lif_3.run(condition=RunSteps(num_steps=self.timesteps), run_cfg=self.run_config)

        # monitors = [mon_lif_1_v.get_data()[mon_lif_1_v_process]["v"], 
        #             mon_lif_2_u.get_data()[mon_lif_2_u_process]["u"], 
        #             mon_lif_2_v.get_data()[mon_lif_2_v_process]["v"], 
        #             mon_lif_3_u.get_data()[mon_lif_3_u_process]["u"], 
        #             mon_spike_1.get_data()[mon_spike_1_process]["s_out"], 
        #             mon_spike_2.get_data()[mon_spike_2_process]["s_out"]]
        
        # lif_3.stop()

        # return monitors

        for i, queryVector in enumerate(tqdm(transform_queryVectors)):
            
            init_v = [ -(self.timesteps  - elem - 1) for elem in queryVector]
            # debug_print([init_v], ['init_v'])
            # print(init_v[:5])
            self.network.update_network(queryVector)
            # print(lif_1.v.get()[:5])

            lif_3.run(condition=RunSteps(num_steps=self.timesteps), run_cfg=self.run_config)
            
            aproxDotProduct_v = lif_2.v.get()/( 2**6 )
            aproxDotProduct_u = lif_2.u.get()/( 2**6 )
            overflow_count    = lif_3.u.get()/( 2**6 )

            # print(lif_2.u.get()[:10])

            # cosine_dist = 1 - aproxDotProduct
        
            # results.append([aproxDotProduct_v, aproxDotProduct_u, overflow_count])
            results.append(overflow_count)
        
        lif_3.stop()

        
        return np.array(results)
    

# class LoihiDotProductSimulationPositive3LayerRateEncoded(LoihiDotProduct):

#     def __init__(self, dbVectors, queryVectors, positive_value: float, timesteps=50, INF=infinity):
#         super().__init__(dbVectors, queryVectors)
#         self.run_config = Loihi1SimCfg(select_tag='fixed_pt')
#         # self.run_config = Loihi1SimCfg(select_tag='floating_pt')
#         self.timesteps = timesteps
#         self.INF = infinity
#         self.network = ThreeLayerSNNRateEncoded()
#         self.positive_value = positive_value

#     def run(self):

#         results = []

#         transform_dbVectors = self.scale_db.augment_positive_based_on_q0(self.positive_value, self.timesteps, self.dbVectors)
#         transform_queryVectors = self.scale_query.augment_positive_based_on_q0(self.positive_value, self.timesteps, self.queryVectors)
#         debug_print([transform_dbVectors, transform_queryVectors], ['transform_dbVectors', 'transform_queryVectors'])
#         queryVector = transform_queryVectors[0]
#         # print(queryVector[:10])
#         lif_1, dense, lif_2, lif_3 = self.network.create_network(self.timesteps, queryVector, transform_dbVectors)


#         debug_print([lif_1.bias_mant.get()[:3]], ['lif_1.bias_mant.get()[:3]'])

#         mon_lif_1_v = Monitor()
#         mon_lif_2_u = Monitor()
#         mon_lif_2_v = Monitor()
#         mon_lif_3_u = Monitor()
#         mon_spike_1 = Monitor()
#         mon_spike_2 = Monitor()

#         mon_lif_1_v.probe(lif_1.v,   self.timesteps)
#         mon_lif_2_u.probe(lif_2.u,   self.timesteps)
#         mon_lif_2_v.probe(lif_2.v,   self.timesteps)
#         mon_lif_3_u.probe(lif_3.u,   self.timesteps)
#         mon_spike_1.probe(lif_1.s_out, self.timesteps)
#         mon_spike_2.probe(lif_2.s_out, self.timesteps)

#         mon_lif_1_v_process = list(mon_lif_1_v.get_data())[0]
#         mon_lif_2_u_process = list(mon_lif_2_u.get_data())[0]
#         mon_lif_2_v_process = list(mon_lif_2_v.get_data())[0]
#         mon_lif_3_u_process = list(mon_lif_3_u.get_data())[0]
#         mon_spike_1_process = list(mon_spike_1.get_data())[0]
#         mon_spike_2_process = list(mon_spike_2.get_data())[0]


#         lif_3.run(condition=RunSteps(num_steps=self.timesteps), run_cfg=self.run_config)

#         monitors = {
#             'lif1_voltage': mon_lif_1_v.get_data()[mon_lif_1_v_process]["v"],
#             'lif2_current': mon_lif_2_u.get_data()[mon_lif_2_u_process]["u"],
#             'lif2_voltage': mon_lif_2_v.get_data()[mon_lif_2_v_process]["v"],
#             'lif3_current': mon_lif_3_u.get_data()[mon_lif_3_u_process]["u"],
#             'lif1_spikes': mon_spike_1.get_data()[mon_spike_1_process]["s_out"],
#             'lif2_spikes': mon_spike_2.get_data()[mon_spike_2_process]["s_out"]
#         }
        
#         lif_3.stop()

#         return monitors

#         for i, queryVector in enumerate(tqdm(transform_queryVectors)):
            
#             init_v = [ -(self.timesteps  - elem - 1) for elem in queryVector]
#             # debug_print([init_v], ['init_v'])
#             # print(init_v[:5])
#             self.network.update_network(queryVector)
#             # print(lif_1.v.get()[:5])

#             lif_3.run(condition=RunSteps(num_steps=self.timesteps), run_cfg=self.run_config)
            
#             aproxDotProduct_v = lif_2.v.get()/( 2**6 )
#             aproxDotProduct_u = lif_2.u.get()/( 2**6 )
#             overflow_count    = lif_3.u.get()/( 2**6 )

#             # print(lif_2.u.get()[:10])

#             # cosine_dist = 1 - aproxDotProduct
        
#             # results.append([aproxDotProduct_v, aproxDotProduct_u, overflow_count])
#             results.append(overflow_count)
        
#         lif_3.stop()

        
#         return np.array(results)

class LoihiDotProductHardware(LoihiDotProduct):

    def __init__(self, dbVectors, queryVectors, positive_value: float, timesteps=50, INF=infinity, hw_profiler=False):
        # Loihi2.preferred_partition = 'oheogulch'
        Loihi2.preferred_partition = 'oheogulch_2h'
        # Loihi2.preferred_partition = 'oheogulch_20m'
        loihi2_is_available = Loihi2.is_loihi2_available
        self.hw_profiler = hw_profiler
        self.profiler = None
        
        if loihi2_is_available:
            from lava.utils import loihi2_profiler
            print(f'Running on {Loihi2.partition}')
            compression = io.encoder.Compression.DELTA_SPARSE_8
        else:
            RuntimeError("Loihi2 compiler is not available in this system. "
            "This tutorial cannot proceed further.")

        super().__init__(dbVectors, queryVectors)
        self.run_config = Loihi2HwCfg()
        self.timesteps = timesteps

        if self.hw_profiler:
            self.profiler = Profiler.init(self.run_config)

            self.profiler.execution_time_probe(num_steps=self.timesteps, buffer_size=2048, dt=1)
            self.profiler.energy_probe(num_steps=self.timesteps)
            self.profiler.memory_probe()
        
        self.INF = infinity
        self.network = TwoLayerSNN()
        self.positive_value = positive_value

    def run(self, dbVectors=None, queryVectors=None):
        if dbVectors is not None:
            self.dbVectors = dbVectors
        if queryVectors is not None:
            self.queryVectors = queryVectors

        results = []

        transform_dbVectors = self.scale_db.augment_positive_based_on_q0(self.positive_value, self.timesteps, self.dbVectors)
        transform_queryVectors = self.scale_query.augment_positive_based_on_q0(self.positive_value, self.timesteps, self.queryVectors)
        debug_print([transform_dbVectors, transform_queryVectors], ['transform_dbVectors', 'transform_queryVectors'])
        queryVector = transform_queryVectors[0]
        # print(queryVector[:10])
        lif_1, dense, lif_2 = self.network.create_network(self.timesteps, queryVector, transform_dbVectors)
        
        lif_2.run(condition=RunSteps(num_steps=self.timesteps), run_cfg=self.run_config)

        for i, queryVector in enumerate(tqdm(transform_queryVectors)):
            
            init_v = [ -(self.timesteps  - elem - 1) for elem in queryVector]
            # debug_print([init_v], ['init_v'])
            # print(init_v[:5])
            self.network.update_network(queryVector)
            # print(lif_1.v.get()[:5])

            lif_2.run(condition=RunSteps(num_steps=self.timesteps), run_cfg=self.run_config)
            
            aproxDotProduct = lif_2.u.get()/( 2**6 )

            # print(lif_2.u.get()[:10])

            # cosine_dist = 1 - aproxDotProduct
        
            results.append(aproxDotProduct)
        
        lif_2.stop()
        
        return np.array(results)
  