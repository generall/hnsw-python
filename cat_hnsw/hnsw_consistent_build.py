from itertools import groupby
from operator import itemgetter
from typing import Dict, Any

import numpy as np
from tqdm import tqdm

from cat_hnsw.hnsw_cat import HNSWCat


class HNSWConsistentBuild(HNSWCat):
    """
    Preserve in-category connectivity.
    """

    def __init__(self, distance_type, m=5, cat_m=5, ef=200, m0=None, heuristic=True, vectorized=False):
        super().__init__(distance_type, m, ef, m0, heuristic, vectorized)
        self._cat_m = cat_m
        self._cat_m0 = cat_m * 2
        self._category_enter_points = {}

    @classmethod
    def _merge_layers(cls, layer_to, layer_from):
        for node, edges in layer_from:
            if node not in layer_to:
                layer_to[node] = edges
            else:
                layer_to[node].update(edges)

    def _merge_graphs(self, graphs: list):
        for layer_idx, layer in enumerate(graphs):
            if layer_idx == len(self._graphs):
                self._graphs.append(layer)
            else:
                self._merge_layers(self._graphs[layer_idx], layer)

    def add_batch(self, data: np.ndarray, categories: Dict[int, Any] = None, ef=None):
        self.data = data
        for i in tqdm(range(self.data.shape[0])):
            self._enter_point = self._add(
                i,
                data=self.data,
                graphs=self._graphs,
                entry_point=self._enter_point,
                m=self._m,
                m0=self._m0,
                ef=ef
            )

        if categories:
            categories = groupby(sorted(categories.items(), key=itemgetter(1)), key=itemgetter(1))
            for category, points in tqdm(categories):

                graphs = []
                entry_point = None
                m = self._cat_m
                m0 = self._cat_m0
                for idx in points:
                    entry_point = self._add(
                        idx,
                        data=self.data,
                        graphs=graphs,
                        entry_point=entry_point,
                        m=m,
                        m0=m0,
                        ef=ef
                    )

                self._category_enter_points[category] = (entry_point, len(graphs) - 1)
                if len(graphs) > len(self._graphs):
                    self._enter_point = entry_point

                self._merge_graphs(graphs)
