import nmslib as nmslib
import numpy

if __name__ == '__main__':

    data = numpy.random.randn(10000, 100).astype(numpy.float32)

    # initialize a new index, using a HNSW index on Cosine Similarity
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.addDataPointBatch(data)
    index.createIndex({'post': 0}, print_progress=True)

    index.saveIndex('my_index.dat')

