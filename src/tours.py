# -*- coding: utf-8 -*-
"""
Formal description of the K Drone Tour state space model.
"""
from src.states import StateSpace
import numpy as np


def indicator(condition):
    """
    :param condition: A boolean statement on which to evaluate inputs.
    :type condition: bool
    :return: 1 if the condition holds, 0 otherwise.
    :rtype: float

    Implementation of the indicator function 1_{condition}(x)
    """
    return float(condition)


def block_site_and_normalize(matrix, site, drone, drone_distribution):
    """
    :param matrix: n
    :param site:
    :param drone:
    :param drone_distribution:
    :return:
    """
    matrix[:, site] = 0.
    normalization = np.sum(matrix, axis=1)[:, None]
    if normalization[site] == 0.:
        drone_distribution[drone] = 0.
        drone_distribution /= np.sum(drone_distribution)

    for row, normalizer in enumerate(normalization):
        if normalizer > 0.:
            matrix /= np.sum(matrix, axis=1)[:, None]


def generate_drone_trajectory(matrix, initial_dist, num_drones):
    """
    :param matrix: A transition probability matrix
    :type matrix: numpy.ndarray
    :param initial_dist:
    :type initial_dist: numpy.ndarray
    :param num_drones:
    :type num_drones: int
    :return: A k-drone tour of list(range(len(initial_dist))) that follows the probabilities in matrix
    :rtype: dict[list]
    """

    # the set of drones and sites being drawn from
    available_drones = list(range(num_drones))
    available_sites = list(range(len(matrix)))

    drone_probabilities = uniform_initial(available_drones)

    # starting sites
    sites = np.random.choice(np.arange(len(initial_dist)), size=num_drones, p=initial_dist)

    # trajectories data structure
    trajectories = {drone: [sites[drone]] for drone in available_drones}

    # normalize after drawing initial starting sites
    for drone in trajectories:
        block_site_and_normalize(matrix, trajectories[drone][-1], drone, drone_probabilities)

    # loop until the matrix is all zeros
    while np.linalg.norm(matrix, ord=1) != 0.:
        drone_id = np.random.choice(available_drones, p=drone_probabilities)
        current_site = trajectories[drone_id][-1]
        next_site = np.random.choice(available_sites, p=matrix[current_site, :])
        trajectories[drone_id].append(next_site)
        block_site_and_normalize(matrix, next_site, drone_id, drone_probabilities)

    return trajectories


def tour_traversal(tour, points):
    """
    :param tour: The
    :type tour: list[int]
    :param points: A point set in a Euclidean space
    :type points: numpy.ndarray
    :return: The length of traversal over the subset of points specified by the tour.
    :rtype: float

    Computes the total traversal time for a single tour on a sub_et of a selection of points.
    """
    return sum(np.linalg.norm(points[tour[i]] - points[tour[i - 1]]) for i, p in enumerate(tour))


def max_tours_traversal(tours, points):
    """
    :param tours: A list of tours over the given points.
    :type tours: dict[int, list]
    :param points: A point set in a Euclidean space
    :type points: numpy.ndarray
    :return: The maximum length of traversal over the subset of points specified by the tours.
    :rtype: float

    Computes the maximum traversal time from a given set of tours over a point set.
    """
    return max(tour_traversal(tours[drone], points) for drone in tours)


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


def uniform_initial(states):
    """
    :param states: A set of bins on which to define the uniform distribution
    :type states: Iterable

    :return: Uniform initial starting distribution on the points
    :rtype: numpy.ndarray

    Returns a distribution vector corresponding to a uniform distribution on the
    sites.
    """
    size = len(states)
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
        return generate_drone_trajectory(self.transition_matrix.copy(), self.initial_distribution, self.num_drones)

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

    def pmf(self, state):
        """

        :param state:
        :return: The probability mass associated with the given state.
        :rtype: float
        """

        probability = 1.

        for drone in state.keys():
            tour = state[drone]
            probability *= self.initial_distribution[tour[0]]
            for (i, site) in enumerate(tour):
                next_site = tour[(i + 1) % len(tour)]
                probability *= self.transition_matrix[site, next_site]

        return probability

    def estimate_parameters(self, threshold, samples, scores):
        """

        :param scores:
        :param threshold:
        :param samples:
        :return:
        """

        sites = len(self.initial_distribution)

        # compute the new initial distribution parameter
        initial_distribution = np.empty(self.initial_distribution.shape)
        for j in range(sites):
            numerator = sum(
                indicator(scores[ell] >= threshold) *
                indicator(any(j == samples[ell][d][0] for d in samples[ell]))
                for ell in range(len(samples))
            )
            denominator = sum(
                indicator(scores[ell] >= threshold) *
                indicator(any(jp == samples[ell][d][0] for d in samples[ell]))
                for ell in range(len(samples)) for jp in range(sites)
            )
            initial_distribution[j] = numerator / denominator

        # compute the new transition matrix
        transition_matrix = np.empty(self.transition_matrix.shape)
        for i in range(sites):
            for j in range(sites):
                numerator = sum(
                    indicator(scores[ell] >= threshold) *
                    indicator(samples[ell][d][r] == i and samples[ell][d][(r + 1) % len(samples[ell][d])] == j)
                    for ell in range(len(samples))
                    for d in range(self.num_drones)
                    for r in range(len(samples[ell][d]))
                )
                denominator = sum(
                    indicator(scores[ell] >= threshold) *
                    indicator(samples[ell][d][r] == i and samples[ell][d][(r + 1) % len(samples[ell][d])] == jp)
                    for jp in range(sites)
                    for ell in range(len(samples))
                    for d in range(self.num_drones)
                    for r in range(len(samples[ell][d]))
                )

                transition_matrix[i, j] = numerator / denominator

        return {'initial_distribution': initial_distribution, 'transition_matrix': self.transition_matrix}

    def get_parameters(self):
        """

        :return:
        """
        return {'initial_distribution': self.initial_distribution, 'transition_matrix': self.transition_matrix}

    def set_parameters(self, parameters):
        """

        :param parameters:
        :return:
        """
        self.initial_distribution = parameters['initial_distribution']
        self.transition_matrix = parameters['transition_matrix']
