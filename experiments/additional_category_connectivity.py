import os
import pickle
import random
from collections import defaultdict

import numpy as np
from gensim.models import KeyedVectors

from cat_hnsw.benchmark.runner import BaseExperiment
from cat_hnsw.hnsw import HNSW
from cat_hnsw.hnsw_cat import HNSWCat
from cat_hnsw.hnsw_consistent_build import HNSWConsistentBuild
from cat_hnsw.settings import DATA_PATH


class CategorySizeConnectivityExperiment(BaseExperiment):

    def __init__(self, experiment_name, m=16, ef=128, dim=50, num_elements=10000):
        super().__init__(experiment_name, m, ef, dim, num_elements)
        model = KeyedVectors.load_word2vec_format(os.path.join(DATA_PATH, 'glove_50k_50.txt'))

        self.glove_train = model.vectors[:self.num_elements]
        self.glove_test = model.vectors[self.num_elements:]

    def add_batch(self, index: HNSWConsistentBuild, data, param):
        categories = {}

        for i in range(param):
            categories[i % param] = range(i, data.shape[0], param)

        index.add_batch(data, categories=categories)

    def generate_index_class(self, param):
        return HNSWConsistentBuild('cosine', m=self.m, ef=self.ef)

    def generate_data(self, param):
        return self.glove_train

    def get_random_vector(self):
        num_test = self.glove_test.shape[0]
        vect_id = random.randint(0, num_test - 1)
        return self.glove_test[vect_id:vect_id + 1]

    def get_mask(self, index, experiment_param, variable_param):
        selected_groups = random.sample(range(experiment_param), variable_param)

        all_mask = np.zeros(index.data.shape[0], dtype=bool)

        for group_idx in selected_groups:
            all_mask |= np.arange(0, index.data.shape[0]) % experiment_param == group_idx

        return all_mask


if __name__ == "__main__":
    experiment = CategorySizeConnectivityExperiment(
        "categorical_connectivity_group_size",
        m=8,
        num_elements=10000
    )

    experiment.run_accuracy_test(
        'group_100',
        experiment_param=100,
        variable_params=list(range(1, 20)),
        attempts_per_value=500
    )

    # experiment.run_accuracy_test(
    #     'group_50',
    #     experiment_param=50,
    #     variable_params=list(range(1, 10)),
    #     attempts_per_value=500
    # )
