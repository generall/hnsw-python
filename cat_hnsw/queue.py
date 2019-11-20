import itertools
import heapq


class FixedLengthPQueue(object):

    def __init__(self, length):
        """
        :param length: max length of queue. Should be greater then 0
        """
        self.pq = []
        self.length = length

    def push(self, elem, priority=0):
        """Add a new element or update the priority of an existing element"""
        entry = (- priority, elem)
        heapq.heappush(self.pq, entry)
        if len(self.pq) > self.length:
            return self.pop()

    def pop(self):
        """Remove and return the lowest priority element. Raise KeyError if empty."""
        while self.pq:
            priority, element = heapq.heappop(self.pq)
            return element
        raise KeyError('pop from an empty priority queue')


if __name__ == '__main__':
    pq = FixedLengthPQueue(3)

    pq.push(1, 1)
    pq.push(2, 0)
    pq.push(3, 1)
    pq.push(4, 3)

    # Should not include 4
    print(pq.pop())
    print(pq.pop())
    print(pq.pop())
