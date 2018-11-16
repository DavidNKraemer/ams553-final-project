# -*- coding: utf-8 -*-
"""
Data structures for computing simulated annealing.
"""
from states import StateSpace
from numpy import exp
from numpy.random import rand


def _metropolis_hastings_probability(cost1, cost2, temperature):
    r"""
    :param cost1:
    :type cost1: float
    :param cost2:
    :type cost2: float
    :param temperature:
    :type temperature: float

    :return: The Metropolis-Hastings probability.
    :rtype float:

    Computes the value

    .. math:: \exp(-\frac{c_2 - c_1}{T})

    This is guaranteed to be between 0 and 1, which makes it suitable as a kind of probability.
    """
    return exp(-(cost2 - cost1) / temperature)


class Annealer:
    """
    A general class for performing simulated annealing.

    .. [AnnealingWiki] https://en.wikipedia.org/wiki/Simulated_annealing
    .. [AnnealingMathWorld] http://mathworld.wolfram.com/SimulatedAnnealing.html
    """


    def __init__(self, space: StateSpace, inf_temp=1e-5, decay=9e-1, internal_iter=100):
        """
        :param space: The state space on which the annealing searches.
        :type space: StateSpae
        :param inf_temp: The minimum "temperature" at which simulated annealing exits.
        :type inf_temp: float
        :param decay: The "decay rate" of the temperature.
        :type decay: float
        :param internal_iter: The number of iterations each step of annealing performs.
        :type internal_iter: int
        """
        self.space = space
        self.inf_temperature = inf_temp
        self.decay = decay
        self.internal_iterations = internal_iter


    @staticmethod
    def acceptance_probability(cost1, cost2, temperature) -> float:
        r"""
        :param cost1: The first cost value. Not really much to be said here.
        :type cost1: float
        :param cost2: The second cost value. Not really much to be said here.
        :type cost2: float
        :param temperature: The current temperature.

        :return: The acceptance probability.
        :rtype: float

        The acceptance probability function specified in the simulated annealing algorithm. As the
        temperature decreases, the probabilities approach zero.

        This defaults to the Metropolis-Hastings acceptance probability formula. At some point we
        could try different options.
        """
        return _metropolis_hastings_probability(cost1, cost2, temperature)


    def anneal(self, starting_state: StateSpace.State) -> (StateSpace.State, float):
        """
        :param starting_state: The place at which the annealing begins.
        :type starting_state: StateSpace.State

        :return: The terminal state space, along with its associated cost.
        :rtype: (StateSpace.State, float)

        Performs the simulated annealing method.
        """
        state = starting_state
        old_cost = self.space.cost(state)
        temperature = 1e0
        while temperature > self.inf_temperature:
            # until the temperature has cooled completely, search for new neighbors.
            for _ in range(self.internal_iterations):
                # draw a neighbor from the current state, and compute its associated cost.
                new_state = self.space.neighbor(state)
                new_cost = self.space.cost(new_state)
                if rand() < self.acceptance_probability(old_cost, new_cost, temperature):
                    # flip a coin. if heads, accept the new state. otherwise, proceed as before.
                    state, old_cost = new_state, new_cost
            # lower the temperature
            temperature *= self.decay
        return state, old_cost
