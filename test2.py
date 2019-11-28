import os

from experiments.additional_category_connectivity import CategorySizeConnectivityExperiment
from experiments.connectivity_experiment_glove import ConnectivityExperimentGlove


if __name__ == '__main__':
    experiment = CategorySizeConnectivityExperiment("categorical_connectivity_group_size")

    # experiment.run_build(param=16)

    index = experiment.load_index()

    if index._enter_point not in index._graphs[-1]:
        index._enter_point = list(index._graphs[-1].keys())[0]

    experiment.run_accuracy_test('random_group_count', 1000, list(range(1, 15)), 100, index=index, mask_attempts=70)
