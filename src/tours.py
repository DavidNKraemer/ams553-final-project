# -*- coding: utf-8 -*-
"""
Formal description of the K Drone Tour state space model.
"""
from src.states import StateSpace
from src.tour_generation import TourGenerator
import numpy as np


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
    if sum(len(tours[d]) for d in tours) >= len(sites):
        return max(tour_traversal(tours[drone], sites) for drone in tours)
    else:
        return np.inf


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

    def __init__(self, sites, num_drones, tour_generator: TourGenerator):
        """
        :param sites: A point set in Euclidean spae
        :type sites: numpy.ndarray
        :param num_drones: The number of drones in the model.
        :type num_drones: int
        :param tour_generator: A TourGenerator which determines the parameterization of the space of all tours.
        :type tour_generator: TourGenerator
        """
        self.sites = sites
        self.num_drones = num_drones
        self.tour_generator = tour_generator

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
        return self.tour_generator.generate_tour()

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

        Computes the pmf of a given k-drone tour according the the transition probability transition_matrix and initial
        distribution of sites.

        The pmf is the product of the initial probabilities of starting states with the product of the transition
        probabilities of the tours.
        """

        return self.tour_generator.pmf(state)

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
        """

        return self.tour_generator.estimate_parameters(threshold, samples, scores)

    def get_parameters(self):
        """
        :return: The initial distribution of sites and the transition transition_matrix packaged into a dictionary
        :rtype: dict

        Retrieves the current probability distribution parameters of the StateSpace
        """
        return self.tour_generator.get_parameters()

    def set_parameters(self, parameters):
        """
        :param parameters: A dictionary of the DroneTour parameters
        :type parameters: dict
        :return: N/A [called for side effects]

        Sets the current StateSpace probability distribution parameters to the provided parameters.
        """
        return self.tour_generator.set_parameters(parameters)
