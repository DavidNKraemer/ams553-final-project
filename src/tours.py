# -*- coding: utf-8 -*-
"""
Formal description of the K Drone Tour state space model.
"""
from random import sample, shuffle
from src.states import StateSpace
import numpy as np


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


def block_site_and_normalize(matrix, site, drone, available_drones):
    """

    :param matrix:
    :param site:
    :param drone:
    :param available_drones:
    :return:
    """
    matrix[:, site] = 0.
    normalization = np.sum(matrix, axis=1)[:, None]
    if normalization[site] == 0.:
        available_drones[drone, 1] = 0.
        available_drones[:, 1] /= np.sum(available_drones[:, 1])

    for row, normalizer in enumerate(normalization):
        if normalizer > 0.:
            matrix /= np.sum(matrix, axis=1)[:, None]


def draw_from(distribution):
    """

    :param distribution:
    :return:
    """

    return np.random.choice(np.arange(len(distribution)), size=1, p=distribution)


def generate_drone_trajectory(matrix, initial_dist, drones):
    """
    :param matrix: A transition probability matrix
    :type matrix: numpy.ndarray
    :param initial_dist:
    :type initial_dist: numpy.ndarray
    :param drones:
    :type int:
    :return: A k-drone tour of list(range(len(initial_dist))) that follows the probabilities in matrix
    :rtype: dict[list]
    """

    available_drones = list(range(drones))
    drone_probabilities = uniform_initial(available_drones)

    # starting sites
    sites = np.random.choice(np.arange(len(initial_dist)), size=drones, p=initial_dist)

    # trajectories data structure
    print(available_drones)
    print(drone_probabilities)

    trajectories = {drone: [sites[drone]] for drone in available_drones}
    for drone in trajectories:
        block_site_and_normalize(matrix, trajectories[drone][-1], drone, drone_probabilities)

    while True:
        drone_id = np.random.choice(available_drones[:, 0], size=1, p=available_drones[:, 1])
        current_site = trajectories[drone_id][-1]
        next_site = draw_from(matrix[current_site, :])
        trajectories[drone_id].append(next_site)
        block_site_and_normalize(matrix, next_site, drone_id, available_drones)

    return trajectories


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
    return sum(np.linalg.norm(points[tour[i]] - points[tour[i-1]]) for i, p in enumerate(tour))


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


def uniform_transition(points):
    """
    :param points: A point set in a Euclidean space
    :type points: numpy.ndarray

    :return: Uniform transition matrix on the points
    :rtype: numpy.ndarray

    Returns a transition matrix corresponding to a uniform distribution on the
    site transitions.
    """
    size = len(points)
    return np.ones((size, size)) / size


def uniform_initial(points):
    """
    :param points: A point set in a Euclidean space
    :type points: numpy.ndarray

    :return: Uniform initial starting distribution on the points
    :rtype: numpy.ndarray

    Returns a distribution vector corresponding to a uniform distribution on the
    sites.
    """
    size = len(points)
    return np.ones(size) / size


class DroneTour(StateSpace):
    """
    Formal state space description of the k-drone tour problem.
    """

    class State(dict):
        """
        Any state is simply a list of lists... right now. Perhaps this could be
        made more specified to the domain.
        """
        pass

    def __init__(self, points, num_drones, transition_matrix=None, initial_distribution=None):
        """
        :param points: A point set in Euclidean spae
        :type points: numpy.ndarray
        :param num_drones: The number of drones in the model.
        :type num_drones: int
        :param transition_matrix: The transition matrix associated with tour generation
        :type transition_matrix: numpy.ndarray
        :param initial_distribution: The distribution of initial sites for the tour generation
        """
        self.points = points
        self.num_drones = num_drones
        self.transition_matrix = transition_matrix or uniform_transition(points)
        self.initial_distribution = initial_distribution or uniform_initial(points)

    def cost(self, state):
        """
        :param state: A current k-tour state configuration
        :type state: KDroneTour.State

        :return: The cost associated with the k-tour
        :rtype: float

        Returns the cost associated with a state
        """
        return max_tours_traversal(state, self.points)

    def generate_state(self):
        """
        :return: A random state
        :rtype: KDroneTour.State

        Returns a random state in the state space according with the distribution parameters.
        """
        return generate_drone_trajectory(self.transition_matrix, self.initial_distribution, self.num_drones)

    def neighbor(self, state):
        """
        :param state: The current state.
        :type state: KDroneTour.State
        :return: A neighbor of the given state.
        :rtype: KDroneTour.State

        From a given state, find and return one of its neighbors.

        Currently, this is just by choosing a random state in the state space,
        but there may be better alternatives.
        """
        return self.generate_state()
