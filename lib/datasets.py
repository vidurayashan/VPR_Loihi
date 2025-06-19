import time
from scipy.io import loadmat, savemat
from scipy.linalg import orth
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import gc
import peer_functions as peer

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

import os
import numpy as np
from scipy.io import loadmat
from typing import Dict, Any, List
import h5py
import scipy.io

class ExperimentRunner:
    def __init__(self, model):
        """
        Initializes the experiment runner.

        Args:
            model: A trained model or object that can compute similarity.
        """
        self.model = model

        # Define dataset configurations (example structure)
        # Each entry: dataset_name: [(DB_path, Query_path, GT_path), ...]
        self.dataset_configs = {
        "GardensPointWalking": [
            (
                "../datasets/descriptors/GardensPointWalking/day_left/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
                "../datasets/descriptors/GardensPointWalking/night_right/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
                "../datasets/ground_truth/GardensPointWalking/day_left--night_right/gt.mat"
            ),
            (
                "../datasets/descriptors/GardensPointWalking/day_right/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
                "../datasets/descriptors/GardensPointWalking/night_right/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
                "../datasets/ground_truth/GardensPointWalking/day_right--night_right/gt.mat"
            ),
            (
                "../datasets/descriptors/GardensPointWalking/day_right/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
                "../datasets/descriptors/GardensPointWalking/day_left/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
                "../datasets/ground_truth/GardensPointWalking/day_right--day_left/gt.mat"
            ),
        ],
        # "CMU": [
        #     (
        #         "../datasets/descriptors/CMU/20110421/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
        #         "../datasets/descriptors/CMU/20100901/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
        #         "../datasets/ground_truth/CMU/20110421--20100901/gt.mat"
        #     ),
        #     (
        #         "../datasets/descriptors/CMU/20110421/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
        #         "../datasets/descriptors/CMU/20100915/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
        #         "../datasets/ground_truth/CMU/20110421--20100915/gt.mat"
        #     ),
        #     (
        #         "../datasets/descriptors/CMU/20110421/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
        #         "../datasets/descriptors/CMU/20101221/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
        #         "../datasets/ground_truth/CMU/20110421--20101221/gt.mat"
        #     ),
        #     (
        #         "../datasets/descriptors/CMU/20110421/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
        #         "../datasets/descriptors/CMU/20110202/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
        #         "../datasets/ground_truth/CMU/20110421--20110202/gt.mat"
        #     ),
        # ],
        # "OxfordRobotCar": [
        #     # (
        #     #     "../datasets/descriptors/OxfordRobotCar/2014-12-09-13-21-02/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
        #     #     "../datasets/descriptors/OxfordRobotCar/2015-05-19-14-06-38/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
        #     #     "../datasets/ground_truth/OxfordRobotCar/2014-12-09-13-21-02--2015-05-19-14-06-38/gt.mat"
        #     # ),
        #     (
        #         "../datasets/descriptors/OxfordRobotCar/2014-12-09-13-21-02/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
        #         "../datasets/descriptors/OxfordRobotCar/2015-08-28-09-50-22/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
        #         "../datasets/ground_truth/OxfordRobotCar/2014-12-09-13-21-02--2015-08-28-09-50-22/gt.mat"
        #     ),
        #     (
        #         "../datasets/descriptors/OxfordRobotCar/2014-12-09-13-21-02/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
        #         "../datasets/descriptors/OxfordRobotCar/2014-11-25-09-18-32/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
        #         "../datasets/ground_truth/OxfordRobotCar/2014-12-09-13-21-02--2014-11-25-09-18-32/gt.mat"
        #     ),
        #     (
        #         "../datasets/descriptors/OxfordRobotCar/2014-12-09-13-21-02/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
        #         "../datasets/descriptors/OxfordRobotCar/2014-12-16-18-44-24/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
        #         "../datasets/ground_truth/OxfordRobotCar/2014-12-09-13-21-02--2014-12-16-18-44-24/gt.mat"
        #     ),
        #     (
        #         "../datasets/descriptors/OxfordRobotCar/2015-05-19-14-06-38/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
        #         "../datasets/descriptors/OxfordRobotCar/2015-02-03-08-45-10/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
        #         "../datasets/ground_truth/OxfordRobotCar/2015-05-19-14-06-38--2015-02-03-08-45-10/gt.mat"
        #     ),
        #     (
        #         "../datasets/descriptors/OxfordRobotCar/2015-08-28-09-50-22/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
        #         "../datasets/descriptors/OxfordRobotCar/2014-11-25-09-18-32/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
        #         "../datasets/ground_truth/OxfordRobotCar/2015-08-28-09-50-22--2014-11-25-09-18-32/gt.mat"
        #     ),
        # ],
        "SFUMountain": [
            (
                "../datasets/descriptors/SFUMountain/dry/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
                "../datasets/descriptors/SFUMountain/dusk/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
                "../datasets/ground_truth/SFUMountain/dry--dusk/gt.mat"
            ),
            (
                "../datasets/descriptors/SFUMountain/dry/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
                "../datasets/descriptors/SFUMountain/jan/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
                "../datasets/ground_truth/SFUMountain/dry--jan/gt.mat"
            ),
            (
                "../datasets/descriptors/SFUMountain/dry/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
                "../datasets/descriptors/SFUMountain/wet/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
                "../datasets/ground_truth/SFUMountain/dry--wet/gt.mat"
            ),
        ],
        "Nordland1000": [
            (
                "../datasets/descriptors/Nordland1000/spring/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
                "../datasets/descriptors/Nordland1000/winter/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
                "../datasets/ground_truth/Nordland1000/spring--winter/gt.mat"
            ),
            (
                "../datasets/descriptors/Nordland1000/spring/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
                "../datasets/descriptors/Nordland1000/summer/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
                "../datasets/ground_truth/Nordland1000/spring--summer/gt.mat"
            ),
            (
                "../datasets/descriptors/Nordland1000/summer/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
                "../datasets/descriptors/Nordland1000/winter/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
                "../datasets/ground_truth/Nordland1000/summer--winter/gt.mat"
            ),
            (
                "../datasets/descriptors/Nordland1000/summer/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
                "../datasets/descriptors/Nordland1000/fall/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
                "../datasets/ground_truth/Nordland1000/summer--fall/gt.mat"
            ),
        ],
        "StLucia": [
            (
                "../datasets/descriptors/StLucia/100909_0845/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
                "../datasets/descriptors/StLucia/180809_1545/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
                "../datasets/ground_truth/StLucia/100909_0845--180809_1545/gt.mat"
            ),
            (
                "../datasets/descriptors/StLucia/100909_1000/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
                "../datasets/descriptors/StLucia/190809_1410/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
                "../datasets/ground_truth/StLucia/100909_1000--190809_1410/gt.mat"
            ),
            (
                "../datasets/descriptors/StLucia/100909_1210/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
                "../datasets/descriptors/StLucia/210809_1210/delf_hdc_4096_ortho_sigma_nx5ny7.mat",
                "../datasets/ground_truth/StLucia/100909_1210--210809_1210/gt.mat"
            ),
        ],
    }
    def top_k_sparsify(self, vector, k):
        """Keep top-k absolute values; set the rest to 0."""
        threshold = np.partition(np.abs(vector), -k)[-k]
        sparse_vector = np.where(np.abs(vector) >= threshold, vector, 0)
        return sparse_vector

    def sparsify_by_top_k(self, vector, density):
        """Zero out values to achieve the given density."""
        k = int(len(vector) * density)
        return self.top_k_sparsify(vector, k)

    def sparsify_by_bin_max(self, vector, density):
        """
        Sparsify the vector by dividing into bins and keeping only the max in each bin.
        
        Parameters:
            vector (np.ndarray): Input dense vector.
            density (float): Fraction of non-zero elements to keep (0 < density <= 1).
            
        Returns:
            np.ndarray: Sparse vector with only one non-zero per bin.
        """
        dim = len(vector)
        n_bins = max(1, int(dim * density))
        bin_size = int(np.ceil(dim / n_bins))

        sparse_vector = np.zeros_like(vector)

        for i in range(n_bins):
            start = i * bin_size
            end = min((i + 1) * bin_size, dim)
            bin_slice = vector[start:end]

            if bin_slice.size > 0:
                max_idx = np.argmax(np.abs(bin_slice))
                sparse_vector[start + max_idx] = bin_slice[max_idx]

        return sparse_vector

    def sparsify_by_bin_max_fast(vector, density):
        """
        Efficiently sparsify a vector by binning and keeping only the max (by abs) in each bin.
        
        Parameters:
            vector (np.ndarray): Input dense vector.
            density (float): Fraction of non-zero elements to keep.
        
        Returns:
            np.ndarray: Sparse vector with one non-zero per bin.
        """
        dim = len(vector)
        n_bins = max(1, int(dim * density))
        bin_size = int(np.ceil(dim / n_bins))

        # Pad vector if not divisible by bin_size
        pad_len = (n_bins * bin_size) - dim
        if pad_len > 0:
            vector = np.pad(vector, (0, pad_len), mode='constant')

        # Reshape into bins
        bins = vector.reshape(n_bins, bin_size)

        # Find max indices per bin (by absolute value)
        max_indices = np.argmax(np.abs(bins), axis=1)
        
        # Calculate global indices of max values
        flat_indices = np.arange(n_bins) * bin_size + max_indices

        # Build sparse vector
        sparse_vector = np.zeros_like(vector)
        sparse_vector[flat_indices] = vector[flat_indices]

        # Remove padding if any
        return sparse_vector[:dim]   
    
    def run_all_experiments(self, **kwargs) -> Dict[str, Any]:
        """
        Runs all experiments on all configured datasets and returns a dictionary of metrics.

        Returns:
            A dictionary where keys are dataset names and values are lists of results for each comparison.
        """
        results = {}
        for dataset_name, comparisons in self.dataset_configs.items():
            print(f'========== Evaluating {dataset_name} ============')
            dataset_results = []
            for db_path, query_path, gt_path in comparisons:
                # Load data
                D_db, D_query, g_truth = self._load_data(db_path, query_path, gt_path, sparsify=kwargs.get('sparsify', False))

                if hasattr(self.model, 'pre_run'):
                    self.model.pre_run(D_db, D_query)

                start_time = time.time()
                # Compute similarity matrix
                similarity_matrix = self._compute_similarity(D_db, D_query, **kwargs)
                # print(similarity_matrix.shape)
                end_time = time.time()
                execution_time_ms = (end_time - start_time) * 1000 / D_query.shape[0]
                print(f"Similarity computation time: {execution_time_ms:.2f} ms")
                # Calculate evaluation metrics
                metrics = self._calculate_evaluation_metrics(-similarity_matrix, g_truth)
                sub_db = db_path.split("/")[-2]
                sub_query = query_path.split("/")[-2]
                print(f'{dataset_name} - {sub_db} - {sub_query} : {metrics["auc"]} - {execution_time_ms:.2f} ms - db_size : {D_db.shape[0]}')
                dataset_results.append({
                    'db': sub_db,
                    'query': sub_query,
                    'auc': metrics,
                    'DD': similarity_matrix
                })
            
            results[dataset_name] = dataset_results
        return results

    def _load_data(self, db_path: str, query_path: str, gt_path: str, sparsify: bool = False):
        """
        Load the descriptors and ground truth data for a given comparison.

        Args:
            db_path: Path to the database descriptors .mat file
            query_path: Path to the query descriptors .mat file
            gt_path: Path to the ground truth .mat file

        Returns:
            D_db: Database descriptors
            D_query: Query descriptors
            g_truth: Ground truth dictionary
        """
        
        def load_mat_file(file_path):
            """Try loading a .mat file using scipy.io.loadmat, fallback to h5py if necessary."""
            try:
                return scipy.io.loadmat(file_path)
            except Exception:
                with h5py.File(file_path, 'r') as f:
                    return {key: f[key][:] for key in f.keys()}  # Convert all datasets to numpy arrays

        D_db = load_mat_file(db_path)['Y']
        D_query = load_mat_file(query_path)['Y']

        if sparsify:
            D_db = self.sparsify_by_top_k(D_db, 0.2)
            D_query = self.sparsify_by_top_k(D_query, 0.2)

        g_truth = load_mat_file(gt_path)

        return D_db, D_query, g_truth


    def _compute_similarity(self, D_db: np.ndarray, D_query: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute similarity matrix using the provided model.

        Args:
            D_db: Nxd database descriptor matrix
            D_query: Mxd query descriptor matrix

        Returns:
            similarity_matrix: MxN matrix of similarity scores
        """
        # Example: Use the model's function. Adapt as needed.
        # For demonstration, assume model has a method model.compute_similarity(D_db, D_query)
        similarity_matrix = self.model.run(D_db, D_query, **kwargs)
        return similarity_matrix

    def _calculate_evaluation_metrics(self, similarity_matrix: np.ndarray, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate evaluation metrics for place recognition results
        
        Args:
            similarity_matrix: Matrix of similarity scores between queries and database
            ground_truth: Ground truth data containing GThard and GTsoft
            
        Returns:
            dict: Dictionary containing recalls, precision, recall curves and AUC
        """
        # # Get candidate selections
        # ids_pos = peer.directCandSel(similarity_matrix)
        
        # # Calculate recall values
        # recalls = peer.getRecallAtKVector(ids_pos, ground_truth["GT"])
        
        hard = ground_truth['GT']["GThard"][0][0]
        soft = ground_truth['GT']["GTsoft"][0][0]
        # Calculate precision-recall curves and AUC
        [R, P] = peer.createPRNew(-similarity_matrix, 
                                hard, 
                                soft)
        auc = np.trapz(P, R)
        
        return {
            'precision': P,
            'recall': R,
            'auc': auc
        }




