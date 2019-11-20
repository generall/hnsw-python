import time
import pickle
import numpy as np
import tqdm as tqdm

from cat_hnsw.hnsw import HNSW

dim = 50
num_elements = 10000


def run_build(path_to_save='random.ind'):
    data = np.random.rand(num_elements, dim)

    hnsw = HNSW('cosine', m0=16, ef=128)

    for row in tqdm.tqdm(data):
        hnsw.add(row)

    hnsw.add(np.random.rand(dim))

    # save index
    with open(path_to_save, 'wb') as f:
        pickle.dump(hnsw, f, pickle.HIGHEST_PROTOCOL)


def run_search(path_to_index='random.ind'):
    with open(path_to_index, 'rb') as fr:
        hnsw_n: HNSW = pickle.load(fr)

    add_point_time = time.time()
    idx = hnsw_n.search(np.float32(np.random.random((1, dim))), 10)
    search_time = time.time()

    print("Searchtime: %f" % (search_time - add_point_time))


if __name__ == "__main__":
    run_search()
