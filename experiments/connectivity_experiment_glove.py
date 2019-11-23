import os
import random

from gensim.models import KeyedVectors

from cat_hnsw.settings import DATA_PATH
from experiments.connectivity_experiment import ConnectivityExperiment


class ConnectivityExperimentGlove(ConnectivityExperiment):

    def __init__(self, experiment_name):
        super().__init__(experiment_name)

        model = KeyedVectors.load_word2vec_format(os.path.join(DATA_PATH, 'glove_50k_50.txt'))

        self.glove_train = model.vectors[:self.num_elements]
        self.glove_test = model.vectors[self.num_elements:]

    def generate_data(self, param):
        return self.glove_train

    def get_random_vector(self):
        num_test = self.glove_test.shape[0]
        vect_id = random.randint(0, num_test - 1)
        return self.glove_test[vect_id:vect_id + 1]


if __name__ == "__main__":
    experiment = ConnectivityExperimentGlove(
        "connectivity_glove_m0",
    )

    experiment.run_accuracy_test(
        'm0_8',
        experiment_param=8,
        variable_params=list(range(30, 99)),
        attempts_per_value=100
    )

    experiment.run_accuracy_test(
        'm0_16',
        experiment_param=16,
        variable_params=list(range(30, 99)),
        attempts_per_value=100
    )

    experiment.run_accuracy_test(
        'm0_24',
        experiment_param=24,
        variable_params=list(range(30, 99)),
        attempts_per_value=100
    )

    experiment.run_accuracy_test(
        'm0_32',
        experiment_param=32,
        variable_params=list(range(30, 99)),
        attempts_per_value=100
    )
