# -*- coding: utf-8 -*-
"""
Formal description of the K Drone Tour state space model.
"""
from random import sample, shuffle
from states import StateSpace
from numpy.linalg import norm


def random_partition(length, parts):
    """
    :param length: the length of the range we are partitioning.
    :type length: int
    :param parts: the number in the partition of the range.
    :type parts: int
    :return: A random partition of the range into the specified sublists.
    :rtype: list[list]

    Partitions the range 0, 1, 2, ..., length-1 into `parts` random sublists.

    I believe this is drawing from a "uniform" distribution over all partitions, but I'm not sure.
    It's safe to assume that the Python functionss `shuffle` and `sample` are implemented to
    conform to uniformity.
    """
    iterable = list(range(length))
    shuffle(iterable)
    indices = [0] + sorted(sample(range(1, length), k=parts-1))
    return [iterable[indices[i]:indices[i+1]] for i in range(parts-1)] + [iterable[indices[-1]:]]


def tour_traversal(tour, points):
    """
    :param tour: The
    :type tour: list[list]
    :param points: A point set in a Euclidean space
    :type points: numpy.ndarray
    :return: The length of traversal over the subset of points specified by the tour.
    :rtype: float

    Computes the total traversal time for a single tour on a sub_et of a selection of points.
    """
    return sum(norm(points[tour[i]] - points[tour[i-1]]) for i, p in enumerate(tour))


def max_tours_traversal(tours, points):
    """
    :param tours: A list of tours over the given points.
    :type tour: list[list]
    :param points: A point set in a Euclidean space
    :type points: numpy.ndarray
    :return: The maximum length of traversal over the subset of points specified by the tours.
    :rtype: float

    Computes the maximum traversal time from a given set of tours over a point set.
    """
    return max(tour_traversal(tour, points) for tour in tours)


class KDroneTour(StateSpace):
    """
    Formal state space description of the k-drone tour problem.
    """


    class State(list):
        """
        Any state is simply a list of lists... right now. Perhaps this could be made more specified
        to the domain.
        """
        pass


    def __init__(self, points, num_drones):
        """
        :param points: A point set in Euclidean spae
        :type points: numpy.ndarray
        :param num_drones: The number of drones in the model.
        :type num_drones: int
        """
        self.points = points
        self.num_drones = num_drones


    def cost(self, state):
        """
        :param state: A current k-tour state configuration
        :type state: KDroneTour.State

        :return: The cost associated with the k-tour
        :rtype: float

        Returns the cost associated with a state
        """
        return max_tours_traversal(state, self.points)


    def random_state(self):
        """
        :return: A random state
        :rtype: KDroneTour.State

        Returns a random state in the state space.
        """
        return random_partition(len(self.points), self.num_drones)


    def neighbor(self, state):
        """
        :param state: The current state.
        :type state: KDroneTour.State
        :return: A neighbor of the given state.
        :rtype: KDroneTour.State

        From a given state, find and return one of its neighbors.

        Currently, this is just by choosing a random state in the state space, but there may be
        better alternatives.
        """
        return self.random_state()
