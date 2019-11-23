import os

from gensim.models import KeyedVectors

from cat_hnsw.settings import DATA_PATH
from experiments.connectivity_experiment import ConnectivityExperiment


class ConnectivityExperimentGlove(ConnectivityExperiment):

    def generate_data(self, param):
        model = KeyedVectors.load_word2vec_format(os.path.join(DATA_PATH, 'glove_50k_50.txt'))
        return model.vectors[:self.num_elements]


if __name__ == "__main__":
    experiment = ConnectivityExperimentGlove(
        "connectivity_glove_m0",
    )

    experiment.run_accuracy_test(
        'm0_8',
        experiment_param=8,
        variable_params=list(range(1, 50)),
        attempts_per_value=100
    )

    experiment.run_accuracy_test(
        'm0_16',
        experiment_param=16,
        variable_params=list(range(1, 50)),
        attempts_per_value=100
    )

    experiment.run_accuracy_test(
        'm0_24',
        experiment_param=24,
        variable_params=list(range(1, 50)),
        attempts_per_value=100
    )

    experiment.run_accuracy_test(
        'm0_32',
        experiment_param=32,
        variable_params=list(range(1, 50)),
        attempts_per_value=100
    )
