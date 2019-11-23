import os

from gensim.models import KeyedVectors

from cat_hnsw.settings import DATA_PATH

if __name__ == '__main__':
    model = KeyedVectors.load_word2vec_format(os.path.join(DATA_PATH, 'glove_50k_50.txt'))

    print(model.vector_size)
