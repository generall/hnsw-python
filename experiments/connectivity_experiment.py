import pickle

import numpy as np


from cat_hnsw.benchmark.runner import BaseExperiment
from cat_hnsw.hnsw import HNSW
from cat_hnsw.hnsw_cat import HNSWCat


class ConnectivityExperiment(BaseExperiment):

    def __init__(self, experiment_name):
        super().__init__(experiment_name, num_elements=10000)

    def generate_index_calss(self, param):
        return HNSW('cosine', m0=param, ef=self.ef)

    def load_index(self):
        with open(self.index_path, 'rb') as fr:
            hnsw_n: HNSW = pickle.load(fr)

        hnsw_n2 = HNSWCat('cosine').init_from_existing(hnsw_n, None)
        return hnsw_n2

    def get_mask(self, index, experiment_param, variable_param):
        all_mask = np.arange(0, index.data.shape[0]) % 100 > variable_param
        return all_mask


if __name__ == "__main__":
    experiment = ConnectivityExperiment(
        "connectivity_m0",
    )

    experiment.run_accuracy_test(
        'm0_8',
        experiment_param=8,
        variable_params=list(range(50, 99)),
        attempts_per_value=100
    )

    experiment.run_accuracy_test(
        'm0_16',
        experiment_param=16,
        variable_params=list(range(50, 99)),
        attempts_per_value=100
    )

    experiment.run_accuracy_test(
        'm0_24',
        experiment_param=24,
        variable_params=list(range(50, 99)),
        attempts_per_value=100
    )

    experiment.run_accuracy_test(
        'm0_32',
        experiment_param=32,
        variable_params=list(range(50, 99)),
        attempts_per_value=100
    )
