from typing import Dict

from cat_hnsw.queue import FixedLengthPQueue


class GraphLayer:

    def __init__(self, max_edges):
        self.max_edges = max_edges
        self.data: Dict[FixedLengthPQueue] = {}

    def set_max_edges(self, max_edges):
        self.max_edges = max_edges
        for pq in self.data.values():
            pq.length = self.max_edges

    def add_node(self, node):
        self.data[node] = FixedLengthPQueue(self.max_edges)

    def add_min_edge(self, node_from, node_to, distance):
        assert node_from != node_to, 'Layer should not contain self-loops'
        from_adjacency: FixedLengthPQueue = self.data.get(node_from)
        excluded = from_adjacency.push(node_to, distance)

        if excluded is not None:
            from_adjacency = self.data.get(node_to)
            from_adjacency.push(node_from, distance)

    def iter_neighbours(self, node):
        adjacency: FixedLengthPQueue = self.data.get(node)
        if adjacency is None:
            return
        for distance, neighbour_node in adjacency.pq:
            yield neighbour_node, -distance
