from math import log2
from random import random

from cat_hnsw.hnsw_cat import HNSWCat


class HNSWConsistentBuild(HNSWCat):
    """
    Preserve in-category connectivity.
    """

    def add(self, elem, category=None, ef=None):

        if ef is None:
            ef = self._ef

        distance = self.distance
        data = self.data
        graphs = self._graphs
        category_point = self._category_enter_points.get(category)
        point = self._enter_point

        m = self._m

        # level at which the element will be inserted
        level = int(-log2(random()) * self._level_mult) + 1
        # print("level: %d" % level)

        # elem will be at data[idx]
        idx = len(data)
        data.append(elem)

        self.categories.append(category)

        if point is not None:  # the HNSW is not empty, we have an entry point
            dist = distance(elem, data[point])
            # for all levels in which we dont have to insert elem,
            # we search for the closest neighbor
            for layer in reversed(graphs[level:]):
                point, dist = self._search_graph_ef1(elem, point, dist, layer)
            # at these levels we have to insert elem; ep is a heap of entry points.
            ep = [(-dist, point)]
            layer0 = graphs[0]
            for layer in reversed(graphs[:level]):
                level_m = m if layer is not layer0 else self._m0
                # navigate the graph and update ep with the closest
                # nodes we find
                ep = self._search_graph(elem, ep, layer, ef)
                # insert in g[idx] the best neighbors
                layer[idx] = layer_idx = {}
                self._select(layer_idx, ep, level_m, layer, heap=True)
                # assert len(layer_idx) <= level_m
                # insert backlinks to the new node
                for j, dist in layer_idx.items():
                    self._select(layer[j], (idx, dist), level_m, layer)
                    # assert len(g[j]) <= level_m
                # assert all(e in g for _, e in ep)
        for i in range(len(graphs), level):
            # for all new levels, we create an empty graph
            graphs.append({idx: {}})
            self._enter_point = idx
