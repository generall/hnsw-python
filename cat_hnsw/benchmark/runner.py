import json
import os
import pickle
import timeit
from typing import List

import numpy as np

from cat_hnsw.hnsw import HNSW
from cat_hnsw.settings import DATA_PATH

from statsmodels.stats.proportion import proportion_confint


def cosine_similarity(vector, matrix):
    return (np.sum(vector * matrix, axis=1) / (
            np.sqrt(np.sum(matrix ** 2, axis=1)) * np.sqrt(np.sum(vector ** 2))))


def calc_precision_at(found_pos: List[int], limit):
    hits = np.array(found_pos) < limit
    return np.mean(hits), proportion_confint(sum(hits), len(found_pos))


class BaseExperiment:
    """
    Class for running benchmarks
    """

    def __init__(
            self,
            experiment_name,
            m0=16,
            ef=128,
            dim=50,
            num_elements=10000
    ):
        self.num_elements = num_elements
        self.dim = dim
        self.ef = ef
        self.m0 = m0
        self.experiment_name = experiment_name

        self.experiment_dir = os.path.join(DATA_PATH, 'experiments', f'exp_{self.experiment_name}')
        os.mkdir(self.experiment_dir)

        self.index_path = os.path.join(self.experiment_dir, 'index.idx')

    def generate_data(self, param):
        return np.random.rand(self.num_elements, self.dim)

    def generate_index_calss(self, param):
        return HNSW('cosine', m0=self.m0, ef=self.ef)

    def run_build(self,
                  param,
                  ):
        data = self.generate_data(param)
        index = self.generate_index_calss(param)
        index.add_batch(data)

        # save index
        with open(self.index_path, 'wb') as f:
            pickle.dump(index, f, pickle.HIGHEST_PROTOCOL)

    def get_random_vector(self):
        return np.float32(np.random.random((1, self.dim)))

    def test_search_time(self, index, topn=10):
        runs, total_time = timeit.Timer(lambda: index.search(self.get_random_vector(), topn)).autorange()
        return total_time / runs

    def test_accuracy(self, data, mask, index, attempts=10):
        found_top = []

        for _ in range(attempts):
            target = self.get_random_vector()

            true_distance = cosine_similarity(target, data)

            np.putmask(true_distance, ~mask, 1000_000)

            closest = list(np.argsort(true_distance))

            # closest_dist = true_distance[closest[:3]]

            approx_closest = index.search(target, 1, condition=lambda point: mask[point])

            approx_closest_idx, approx_closest_dist = approx_closest[0]

            found_top.append(closest.index(approx_closest_idx))

        return found_top

    def save_metrics(self, fd, found_top, experiment_param, variable_param):
        fd.write(json.dumps({
            'experiment_param': experiment_param,
            'variable_param': variable_param,
            'precision@10': calc_precision_at(found_top, 10),
        }))
        fd.write('\n')
        fd.flush()

    def load_index(self):
        with open(self.index_path, 'rb') as fr:
            hnsw_n: HNSW = pickle.load(fr)

        return hnsw_n

    def get_mask(self, index, experiment_param, variable_param):
        all_mask = np.ones(index.data.shape[0], dtype=bool)
        return all_mask

    def run_accuracy_test(self, iteration_name, experiment_param, variable_params: list, attempts_per_value=100):
        self.run_build(experiment_param)
        index = self.load_index()

        with open(os.path.join(self.experiment_dir, iteration_name + '.jsonl')) as logs_out:
            for variable_param in variable_params:
                mask = self.get_mask(index, experiment_param, variable_param)
                found_top = self.test_accuracy(index.data, mask=mask, index=index, attempts=attempts_per_value)
                self.save_metrics(logs_out, found_top, experiment_param, variable_param)
