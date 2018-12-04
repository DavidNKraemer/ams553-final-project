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
    :param matrix: a (possibly sub-)stochastic matrix
    :type matrix: numpy.ndarray
    :param site: the current site to update in the matrix
    :type site: int
    :param drone: the current drone to possibly update
    :type drone: int
    :param drone_distribution: the distribution of drone draws, which may need to be updated
    :type drone_distribution: numpy.ndarray
    :return:

    Given a stochastic matrix , the current site, a drone, and the distribution of drones, update the matrix so that the
    current site can no longer be reached. If a resulting row has no more positive entries, it is no longer reachable,
    and therefore isn't normalized. In this case, the drone can no longer proceed, and so the drone distribution is
    updated to make future draws ignore the given drone.
    """

    # zero out the column associated with the current site
    matrix[:, site] = 0.
    normalization = np.sum(matrix, axis=1)[:, None]  # pre-compute all normalization values
    if normalization[site] == 0.:  # in this case, the drone can no longer proceed to new sites, so the drone
                                    # distribution is updated
        drone_distribution[drone] = 0.
        drone_distribution /= np.sum(drone_distribution)

    for row, normalizer in enumerate(normalization):
        if normalizer > 0.:  # only normalize reachable (i.e., nonzero) rows.
            matrix[row, :] /= normalizer


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

    Generates a drone trajectory according to a (possibly sub-)stochastic matrix and initial site distribution, along
    with a specified number of drones to draw.

    See Rubinstein and Kroese (2004) for a description on how to generate such trajectories for the TSP (i.e., 1-drone
    case).
    """

    # the set of drones and sites being drawn from
    available_drones = list(range(num_drones))
    available_sites = list(range(len(matrix)))

    drone_probabilities = uniform_initial(available_drones)

    # starting sites
    sites = np.random.choice(available_sites, size=num_drones, p=initial_dist)

    # trajectories data structure
    trajectories = {drone: [sites[drone]] for drone in available_drones}

    # normalize after drawing initial starting sites
    for drone in trajectories:
        block_site_and_normalize(matrix, trajectories[drone][-1], drone, drone_probabilities)

    # loop until the matrix is all zeros
    while np.linalg.norm(matrix, ord=1) != 0.:
        drone_id = np.random.choice(available_drones, p=drone_probabilities)  # choose a drone
        current_site = trajectories[drone_id][-1]
        next_site = np.random.choice(available_sites, p=matrix[current_site, :])  # choose the next site
        trajectories[drone_id].append(next_site)
        block_site_and_normalize(matrix, next_site, drone_id, drone_probabilities)  # normalize and proceed

    return trajectories


def tour_traversal(tour, sites):
    """
    :param tour: The
    :type tour: list[int]
    :param sites: A point set in a Euclidean space
    :type sites: numpy.ndarray
    :return: The length of traversal over the subset of sites specified by the tour.
    :rtype: float

    Computes the total traversal time for a single tour on a sub_et of a selection of sites.
    """
    return sum(np.linalg.norm(sites[tour[i]] - sites[tour[i - 1]]) for i, p in enumerate(tour))


def max_tours_traversal(tours, sites):
    """
    :param tours: A list of tours over the given sites.
    :type tours: dict[int, list]
    :param sites: A point set in a Euclidean space
    :type sites: numpy.ndarray
    :return: The maximum length of traversal over the subset of sites specified by the tours.
    :rtype: float

    Computes the maximum traversal time from a given set of tours over a point set.
    """
    return max(tour_traversal(tours[drone], sites) for drone in tours)


def uniform_transition(sites):
    """
    :param sites: A point set in a Euclidean space
    :type sites: numpy.ndarray

    :return: Uniform transition matrix on the sites
    :rtype: numpy.ndarray

    Returns a transition matrix corresponding to a uniform distribution on the
    site transitions.
    """
    size = len(sites)
    return np.ones((size, size)) / size


def uniform_initial(sites):
    """
    :param sites: A set of bins on which to define the uniform distribution
    :type sites: Iterable

    :return: Uniform initial starting distribution on the sites
    :rtype: numpy.ndarray

    Returns a distribution vector corresponding to a uniform distribution on the
    sites.
    """
    size = len(sites)
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

    def __init__(self, sites, num_drones, transition_matrix=None, initial_distribution=None):
        """
        :param sites: A point set in Euclidean spae
        :type sites: numpy.ndarray
        :param num_drones: The number of drones in the model.
        :type num_drones: int
        :param transition_matrix: The transition matrix associated with tour generation
        :type transition_matrix: numpy.ndarray
        :param initial_distribution: The distribution of initial sites for the tour generation
        """
        self.sites = sites
        self.num_drones = num_drones
        self.transition_matrix = transition_matrix or uniform_transition(sites)
        self.initial_distribution = initial_distribution or uniform_initial(sites)

    def cost(self, state):
        """
        :param state: A current k-tour state configuration
        :type state: DroneTour.State

        :return: The cost associated with the k-tour
        :rtype: float

        Returns the cost associated with a state
        """
        return max_tours_traversal(state, self.sites)

    def generate_state(self):
        """
        :return: A random state
        :rtype: DroneTour.State

        Returns a random state in the state space according with the distribution parameters.
        """
        return generate_drone_trajectory(self.transition_matrix.copy(), self.initial_distribution, self.num_drones)

    def neighbor(self, state):
        """
        :param state: The current state.
        :type state: DroneTour.State
        :return: A neighbor of the given state.
        :rtype: DroneTour.State

        From a given state, find and return one of its neighbors.

        Currently, this is just by choosing a random state in the state space,
        but there may be better alternatives.
        """
        return self.generate_state()

    def pmf(self, state):
        """
        :param state: A specific k-drone tour
        :type state: dict[list[int]]
        :return: The probability mass associated with the given state.
        :rtype: float

        Computes the pmf of a given k-drone tour according the the transition probability matrix and initial
        distribution of sites.

        The pmf is the product of the initial probabilities of starting states with the product of the transition
        probabilities of the tours.
        """

        probability = 1.

        for drone in state:
            tour = state[drone]
            probability *= self.initial_distribution[tour[0]]
            for (i, site) in enumerate(tour):
                next_site = tour[(i + 1) % len(tour)]
                probability *= self.transition_matrix[site, next_site]

        return probability

    def estimate_parameters(self, threshold, samples, scores):
        """
        :param threshold: a score threshold for comparing against the sample scores
        :type threshold: float
        :param samples: a set of samples generated from cross entropy
        :type samples: list[DroneTour.State]
        :param scores: a set of scores associated with each sample in samples
        :type scores: list[float]

        :return: a dictionary containing the newly computed initial_distribution and transition probabilities
        :rtype: dict

        Estimates and returns the cross entropy-minimizing parameters from the available sample data. See the references
        to cross_entropy.CrossEntropy.

        The keys in the dictionary are
        - 'initial_distribution': the initial distribution of sites
        - 'transition_matrix': the transition probabilities on the sites
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
        :return: The initial distribution of sites and the transition matrix packaged into a dictionary
        :rtype: dict

        Retrieves the current probability distribution parameters of the StateSpace

        The keys in the dictionary are
        - 'initial_distribution': a numpy.ndarray; the initial distribution of sites
        - 'transition_matrix': a numpy.ndarray; the transition probabilities on the sites
        """
        return {'initial_distribution': self.initial_distribution, 'transition_matrix': self.transition_matrix}

    def set_parameters(self, parameters):
        """
        :param parameters: A dictionary of the DroneTour parameters
        :type parameters: dict
        :return: N/A [called for side effects]

        Sets the current StateSpace probability distribution parameters to the provided parameters.

        The dictionary parameters has the following keys:
        - 'initial_distribution': a numpy.ndarray; the initial distribution of sites
        - 'transition_matrix': a numpy.ndarray; the transition probabilities on the sites
        """
        self.initial_distribution = parameters['initial_distribution']
        self.transition_matrix = parameters['transition_matrix']
