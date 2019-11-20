import time
import pickle
from collections import defaultdict

import numpy as np
import tqdm as tqdm

from cat_hnsw.hnsw import HNSW

dim = 50
num_elements = 10000
import timeit


def run_build(path_to_save='random.ind'):
    data = np.random.rand(num_elements, dim)

    hnsw = HNSW('cosine', m0=16, ef=128)

    hnsw.add_batch(data)

    # save index
    with open(path_to_save, 'wb') as f:
        pickle.dump(hnsw, f, pickle.HIGHEST_PROTOCOL)


def get_random_vector():
    return np.float32(np.random.random((1, dim)))


def test_search_time(index, topn=10):
    runs, total_time = timeit.Timer(lambda: index.search(get_random_vector(), topn)).autorange()
    print("loop: ", total_time / runs)


def cosine_similarity(vector, matrix):
    return (np.sum(vector * matrix, axis=1) / (
                np.sqrt(np.sum(matrix ** 2, axis=1)) * np.sqrt(np.sum(vector ** 2))))


def test_accuracy(data, index, attempts=10, top=20):

    average_position = defaultdict(list)

    for _ in tqdm.tqdm(range(attempts)):
        target = get_random_vector()

        true_distance = cosine_similarity(target, data)

        closest = list(np.argsort(true_distance))

        # closest_dist = true_distance[closest]

        approx_closest = index.search(target, top)

        for top_appx, (idx, dict) in enumerate(approx_closest):
            average_position[top_appx].append(closest.index(idx))

    print("")
    for top, positions in average_position.items():
        print("Found pos:", top, ' avg real', np.mean(positions))


def run_search(path_to_index='random.ind'):
    with open(path_to_index, 'rb') as fr:
        hnsw_n: HNSW = pickle.load(fr)

    # test_search_time(hnsw_n)

    test_accuracy(hnsw_n.data, hnsw_n, attempts=100, top=3)


if __name__ == "__main__":
    # run_build()
    run_search()
