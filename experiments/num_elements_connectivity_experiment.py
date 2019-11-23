import pickle

import numpy as np


from cat_hnsw.benchmark.runner import BaseExperiment
from cat_hnsw.hnsw import HNSW
from cat_hnsw.hnsw_cat import HNSWCat


class ConnectivityNumElementsExperiment(BaseExperiment):

    def generate_data(self, param):
        return np.random.rand(param, self.dim)

    def load_index(self):
        with open(self.index_path, 'rb') as fr:
            hnsw_n: HNSW = pickle.load(fr)

        hnsw_n2 = HNSWCat('cosine').init_from_existing(hnsw_n)
        return hnsw_n2

    def get_mask(self, index, experiment_param, variable_param):
        all_mask = np.arange(0, index.data.shape[0]) % 100 > variable_param
        return all_mask


if __name__ == "__main__":
    experiment = ConnectivityNumElementsExperiment(
        "connectivity_num_elements",
    )

    experiment.run_accuracy_test(
        'num_10k',
        experiment_param=10_000,
        variable_params=list(range(50, 99)),
        attempts_per_value=100
    )

    experiment.run_accuracy_test(
        'num_20k',
        experiment_param=20_000,
        variable_params=list(range(50, 99)),
        attempts_per_value=100
    )

    experiment.run_accuracy_test(
        'num_30k',
        experiment_param=30_000,
        variable_params=list(range(50, 99)),
        attempts_per_value=100
    )
