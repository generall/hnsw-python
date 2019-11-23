import os
import pickle
import random

import numpy as np
from gensim.models import KeyedVectors

from cat_hnsw.benchmark.runner import BaseExperiment
from cat_hnsw.hnsw import HNSW
from cat_hnsw.hnsw_cat import HNSWCat
from cat_hnsw.settings import DATA_PATH
from experiments.num_elements_connectivity_experiment import ConnectivityNumElementsExperiment


class ConnectivityGloveNumElementsExperiment(ConnectivityNumElementsExperiment):

    def __init__(self, experiment_name):
        super().__init__(experiment_name)
        model = KeyedVectors.load_word2vec_format(os.path.join(DATA_PATH, 'glove_50k_50.txt'))

        self.glove_train = model.vectors[:40000]
        self.glove_test = model.vectors[40000:]

    def generate_data(self, param):
        return self.glove_train[:param]

    def get_random_vector(self):
        num_test = self.glove_test.shape[0]
        vect_id = random.randint(0, num_test - 1)
        return self.glove_test[vect_id:vect_id + 1]


if __name__ == "__main__":
    experiment = ConnectivityGloveNumElementsExperiment(
        "connectivity_glove_num_elements",
    )

    experiment.run_accuracy_test(
        'num_10k',
        experiment_param=10_000,
        variable_params=list(range(30, 99)),
        attempts_per_value=100
    )

    experiment.run_accuracy_test(
        'num_20k',
        experiment_param=20_000,
        variable_params=list(range(30, 99)),
        attempts_per_value=100
    )

    experiment.run_accuracy_test(
        'num_30k',
        experiment_param=30_000,
        variable_params=list(range(30, 99)),
        attempts_per_value=100
    )
