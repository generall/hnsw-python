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


class BaseRunner:
    """
    Class for running benchmarks
    """

    def __init__(
            self,
            experiment_name,
            param,
            m0=16,
            ef=128,
            dim=50,
            num_elements=10000
    ):
        self.param = param
        self.experiment_name = experiment_name
        self.num_elements = num_elements
        self.dim = dim
        self.ef = ef
        self.m0 = m0

        self.experiment_dir = os.path.join(DATA_PATH, 'experiments', f'exp_{self.experiment_name}')
        os.mkdir(self.experiment_dir)

        self.index_path = os.path.join(self.experiment_dir, 'index.idn')

    def run_build(self):
        path_to_save = os.path.join(self.experiment_dir, 'index.idn')

        data = np.random.rand(self.num_elements, self.dim)

        hnsw = HNSW('cosine', m0=self.m0, ef=self.ef)
        hnsw.add_batch(data)

        # save index
        with open(self.index_path, 'wb') as f:
            pickle.dump(hnsw, f, pickle.HIGHEST_PROTOCOL)

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

    def save_metrics(self, fd, found_top, param):
        fd.write(json.dumps({
            'm0': self.m0,
            'param': param,
            'precision@10': calc_precision_at(found_top, 10),
        }))
        fd.write('\n')
        fd.flush()

    def load_index(self):
        with open(self.index_path, 'rb') as fr:
            hnsw_n: HNSW = pickle.load(fr)

        return hnsw_n

    def get_mask(self, index, param):
        all_mask = np.ones(index.data.shape[0], dtype=bool)
        return all_mask

    def run(self, iteration_name, param_vals: list, attempts_per_value=100):
        self.run_build()
        index = self.load_index()

        with open(os.path.join(self.experiment_dir, iteration_name + '.jsonl')) as logs_out:
            for param_val in param_vals:
                mask = self.get_mask(index, param_val)
                found_top = self.test_accuracy(index.data, mask=mask, index=index, attempts=attempts_per_value)
                self.save_metrics(logs_out, found_top, param_val)
