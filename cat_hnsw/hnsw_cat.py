# -*- coding: utf-8 -*-
from collections import defaultdict
from heapq import heapify, heappop, heappush, heapreplace, nlargest
from typing import Container, Callable

from cat_hnsw.hnsw import HNSW


def accept_all(item):
    return True


class HNSWCat(HNSW):
    """
    Category-aware HNSW index
    """

    def __init__(self, distance_type, m=5, ef=200, m0=None, heuristic=True, vectorized=False):
        super().__init__(distance_type, m, ef, m0, heuristic, vectorized)

    def init_from_existing(self, other: HNSW):
        self.distance_func = other.distance_func
        self.vectorized_distance = other.vectorized_distance
        self.data = other.data
        self._m = other._m
        self._ef = other._ef
        self._m0 = other._m0
        self._level_mult = other._level_mult
        self._graphs = other._graphs
        self._enter_point = other._enter_point
        self._select = other._select

        return self

    def get_entry_point(self, condition):
        """
        Finds suitable entry point

        :param condition:
        :return: entry point + level
        """
        if condition(self._enter_point):
            return self._enter_point, len(self._graphs)
        else:
            for idx, layer in enumerate(reversed(self._graphs)):
                for point in layer:
                    if condition(point):
                        return point, len(self._graphs) - idx

    def search(self, q, k=None, ef=None, condition=accept_all):

        distance = self.distance
        graphs = self._graphs
        point, start_level = self.get_entry_point(condition)

        if ef is None:
            ef = self._ef

        if point is None:
            raise ValueError("Empty graph")

        dist = distance(q, self.data[point])
        # look for the closest neighbor from the top to the 2nd level
        for layer in reversed(graphs[1:start_level]):
            point, dist = self._search_graph_ef1(q, point, dist, layer, condition=condition)
        # look for ef neighbors in the bottom level
        ep = self._search_graph(q, [(-dist, point)], graphs[0], ef, condition=condition)

        if k is not None:
            ep = nlargest(k, ep)
        else:
            ep.sort(reverse=True)

        return [(idx, -md) for md, idx in ep]

    def _search_graph_ef1(self, q, entry, dist, layer, condition: Callable[[int], bool] = accept_all):

        vectorized_distance = self.vectorized_distance
        data = self.data

        best = entry
        best_dist = dist
        candidates = [(dist, entry)]
        visited = {entry}

        while candidates:
            dist, c = heappop(candidates)
            if dist > best_dist:
                break
            edges = [e for e in layer[c] if (e not in visited) and condition(e)]
            visited.update(edges)
            dists = vectorized_distance(q, [data[e] for e in edges])
            for e, dist in zip(edges, dists):
                if dist < best_dist:
                    best = e
                    best_dist = dist
                    heappush(candidates, (dist, e))
                    # break

        return best, best_dist

    def _search_graph(self, q, ep, layer, ef, condition: Callable[[int], bool] = accept_all):

        vectorized_distance = self.vectorized_distance
        data = self.data

        candidates = [(-mdist, p) for mdist, p in ep]
        heapify(candidates)
        visited = set(p for _, p in ep)

        while candidates:
            dist, c = heappop(candidates)
            mref = ep[0][0]
            if dist > -mref:
                break

            edges = [e for e in layer[c] if (e not in visited) and condition(e)]
            visited.update(edges)
            dists = vectorized_distance(q, [data[e] for e in edges])
            for e, dist in zip(edges, dists):
                mdist = -dist
                if len(ep) < ef:
                    heappush(candidates, (dist, e))
                    heappush(ep, (mdist, e))
                    mref = ep[0][0]
                elif mdist > mref:
                    heappush(candidates, (dist, e))
                    heapreplace(ep, (mdist, e))
                    mref = ep[0][0]

        return ep
