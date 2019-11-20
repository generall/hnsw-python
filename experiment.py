import json
import time
import pickle
from collections import defaultdict

import numpy as np
import tqdm as tqdm

import timeit

from statsmodels.stats.proportion import proportion_confint

from cat_hnsw.hnsw import HNSW
from cat_hnsw.hnsw_cat import HNSWCat

m0 = 24
ef = 128
dim = 50
num_elements = 10000


def run_build(path_to_save='random.ind'):
    data = np.random.rand(num_elements, dim)

    hnsw = HNSW('cosine', m0=m0, ef=ef)

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


def test_accuracy(data, mask, index, attempts=10, top=20):
    in_top_3 = []
    in_top_10 = []
    in_top_25 = []

    for _ in range(attempts):
        target = get_random_vector()

        true_distance = cosine_similarity(target, data)

        np.putmask(true_distance, ~mask, 1000_000)

        closest = list(np.argsort(true_distance))

        closest_dist = true_distance[closest[:3]]

        approx_closest = index.search(target, top, condition=lambda point: mask[point])

        approx_closest_idx, approx_closest_dist = approx_closest[0]

        in_top_3.append(int(closest.index(approx_closest_idx) < 3))
        in_top_10.append(int(closest.index(approx_closest_idx) < 10))
        in_top_25.append(int(closest.index(approx_closest_idx) < 25))

    return [
        (np.mean(in_top_3), proportion_confint(sum(in_top_3), attempts, alpha=0.10)),
        (np.mean(in_top_10), proportion_confint(sum(in_top_10), attempts, alpha=0.10)),
        (np.mean(in_top_25), proportion_confint(sum(in_top_25), attempts, alpha=0.10)),
    ]


def run_search(path_to_index='random.ind'):
    with open(path_to_index, 'rb') as fr:
        hnsw_n: HNSW = pickle.load(fr)

    hnsw_n2 = HNSWCat('cosine').init_from_existing(hnsw_n, None)

    all_mask = np.ones(hnsw_n2.data.shape[0], dtype=bool)
    # Found pos: 0  avg real 3.032 presition@10 0.929
    # Found pos: 1  avg real 7.752 presition@10 0.708
    # Found pos: 2  avg real 13.049 presition@10 0.409

    half = np.arange(0, hnsw_n2.data.shape[0]) % 2 == 0
    # Found pos: 0  avg real 2.9198782961460448 presition@10 0.912
    # Found pos: 1  avg real 7.421906693711968 presition@10 0.723
    # Found pos: 2  avg real 12.609533468559837 presition@10 0.44

    third = np.arange(0, hnsw_n2.data.shape[0]) % 3 == 0
    # Found pos: 0  avg real 2.3448275862068964 presition@10 0.89
    # Found pos: 1  avg real 6.608836206896552 presition@10 0.715
    # Found pos: 2  avg real 11.428879310344827 presition@10 0.451

    forth = np.arange(0, hnsw_n2.data.shape[0]) % 4 == 0
    # Found pos: 0  avg real 17.538829151732376 presition@10 0.55
    # Found pos: 1  avg real 73.38709677419355 presition@10 0.35
    # Found pos: 2  avg real 188.689366786141 presition@10 0.211

    data = []

    with open(f'./data/experiments/m0_{m0}_precision.jsonl', 'w') as fd:
        for i in range(50, 99):
            forth = np.arange(0, hnsw_n2.data.shape[0]) % 100 > i

            precision_3, precision_10, precision_25 = test_accuracy(
                data=hnsw_n2.data,
                mask=forth,
                index=hnsw_n2,
                attempts=100,
                top=3
            )

            fd.write(json.dumps({
                'm0': m0,
                'frac': i / 100,
                'precision@3': precision_3,
                'precision@10': precision_10,
                'precision@25': precision_25
            }))
            fd.write('\n')
            fd.flush()

            print(f"{i / 100} p%10 {precision_10[1][0]:.2f} - {precision_10[1][1]:.2f}")


if __name__ == "__main__":
    run_build()
    run_search()
