import os

from experiments.additional_category_connectivity import CategorySizeConnectivityExperiment
from experiments.connectivity_experiment_glove import ConnectivityExperimentGlove


if __name__ == '__main__':
    experiment = CategorySizeConnectivityExperiment("categorical_connectivity_group_size")

    # experiment.run_build(param=16)

    index = experiment.load_index()

    results = experiment.test_accuracy(
        index.data,
        mask=None,  # experiment.get_mask(index, 100, 2),
        index=index,
        attempts=10
    )

    print(results)
