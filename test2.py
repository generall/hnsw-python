import os

from experiments.additional_category_connectivity import CategorySizeConnectivityExperiment
from experiments.connectivity_experiment_glove import ConnectivityExperimentGlove


if __name__ == '__main__':
    experiment = CategorySizeConnectivityExperiment("categorical_connectivity_group_size")

    # experiment.run_build(param=16)

    index = experiment.load_index()

    if index._enter_point not in index._graphs[-1]:
        index._enter_point = list(index._graphs[-1].keys())[0]

    for i in range(50):
        results = experiment.test_accuracy(
            index.data,
            mask=experiment.get_mask(index, 397, i),
            index=index,
            attempts=10
        )

        print(i / 397, results)
