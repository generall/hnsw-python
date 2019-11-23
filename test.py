import os

import nmslib
from gensim.models import KeyedVectors

from cat_hnsw.settings import DATA_PATH
from experiments.connectivity_experiment_glove import ConnectivityExperimentGlove


class NMSLIBExperiment(ConnectivityExperimentGlove):

    def search_closest(self, index, target, condition):
        idx, dists = index.knnQuery(target, k=1)

        return [(idx[0], dists[0])]


if __name__ == '__main__':
    experiment = ConnectivityExperimentGlove("test")

    experiment.run_build(param=16)

    index = experiment.load_index()

    print(index._m)

    results = experiment.test_accuracy(
        index.data,
        mask=None,
        index=index,
        attempts=10
    )

    print(results)

    nmslib_exp = NMSLIBExperiment("test2")

    nmslib_index = nmslib.init(method='hnsw', space='cosinesimil')
    nmslib_index.addDataPointBatch(index.data)
    nmslib_index.createIndex({'post': 0, 'M': 16, 'efConstruction': 128}, print_progress=True)

    results = nmslib_exp.test_accuracy(index.data, mask=None, index=nmslib_index)

    print(results)
