import random

import numpy as np

from cat_hnsw.benchmark.runner import BaseExperiment
from cat_hnsw.hnsw_consistent_build import HNSWConsistentBuild


class CategorySizeConnectivityRandomExperiment(BaseExperiment):

    def add_batch(self, index: HNSWConsistentBuild, data, param):
        categories = {}

        for i in range(param):
            categories[i % param] = range(i, data.shape[0], param)

        index.add_batch(data, categories=categories)

    def generate_index_class(self, param):
        return HNSWConsistentBuild('cosine', m=self.m, ef=self.ef)

    @classmethod
    def select_groups(cls, experiment_param, variable_param):
        """

        :param experiment_param:
        :param variable_param:
        :return:
        """
        selected_group = random.choice(range(experiment_param - variable_param))

        selected_groups = []
        for i in range(variable_param + 1):
            selected_groups.append(selected_group + i)

        return selected_groups

    def get_mask(self, index, experiment_param, variable_param):
        selected_groups = self.select_groups(experiment_param, variable_param)

        all_mask = np.zeros(index.data.shape[0], dtype=bool)

        for group_idx in selected_groups:
            all_mask |= np.arange(0, index.data.shape[0]) % experiment_param == group_idx

        return all_mask


if __name__ == "__main__":
    experiment = CategorySizeConnectivityRandomExperiment(
        "categorical_connectivity_group_size",
        m=8,
        num_elements=200_000
    )

    experiment.run_accuracy_test(
        'group_100',
        experiment_param=2000,
        variable_params=list(range(1, 40)),
        attempts_per_value=100
    )

    # experiment.run_accuracy_test(
    #     'group_50',
    #     experiment_param=50,
    #     variable_params=list(range(1, 10)),
    #     attempts_per_value=500
    # )
