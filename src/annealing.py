# -*- coding: utf-8 -*-
"""
Data structures for computing simulated annealing.
"""
from src.states import StateSpace
import numpy as np


def _metropolis_hastings_probability(old_cost, new_cost, temperature):
    r"""
    :param old_cost:
    :type old_cost: float
    :param new_cost:
    :type new_cost: float
    :param temperature:
    :type temperature: float

    :return: The Metropolis-Hastings probability.
    :rtype float:

    Computes the value

    .. math:: \exp(-\frac{c_2 - c_1}{T})

    This is guaranteed to be between 0 and 1, which makes it suitable as a kind of probability.
    """
    return np.exp(-(new_cost - old_cost) / temperature)


class Annealer:
    """
    A general class for performing simulated annealing.

    References
    ----------

    - https://en.wikipedia.org/wiki/Simulated_annealing
    - http://mathworld.wolfram.com/SimulatedAnnealing.html
    """

    def __init__(self, space: StateSpace, inf_temp=1e-5, decay=9e-1, internal_iter=100):
        """
        :param space: The state space on which the annealing searches.
        :type space: StateSpace
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
    def acceptance_probability(old_cost, new_cost, temperature) -> float:
        r"""
        :param old_cost: The first cost value. Not really much to be said here.
        :type old_cost: float
        :param new_cost: The second cost value. Not really much to be said here.
        :type new_cost: float
        :param temperature: The current temperature.

        :return: The acceptance probability.
        :rtype: float

        The acceptance probability function specified in the simulated annealing algorithm. As the
        temperature decreases, the probabilities approach zero.

        This defaults to the Metropolis-Hastings acceptance probability formula. At some point we
        could try different options.
        """
        return _metropolis_hastings_probability(old_cost, new_cost, temperature)

    def anneal(self, callback=None) -> (StateSpace.State, float):
        """
        :param callback:
        The keyword arguments passed to callback are
            - iteration, the current iteration
            - current state, the state associated with the current annealing iteration
            - current_cost, the cost associated with the current state
            - new_state, a new state drawn in the current iteration
            - new_cost, the cost associated with the new state
            - acceptance_probability, the probability that the annealer will accept the new state as the next state in
                                      the current iteration
            - coin_flip, the value of the coin flip associated with the current iteration
            - current_temperature, the temperature associated with the current iteration

        :return: The terminal state space, along with its associated cost.
        :rtype: (StateSpace.State, float)

        Performs the simulated annealing method.
        """
        state = self.space.generate_state()
        cost = self.space.cost(state)
        temperature = 1e0
        while temperature > self.inf_temperature:
            # until the temperature has cooled completely, search for new neighbors.
            for iteration in range(self.internal_iterations):

                # draw a neighbor from the current state, and compute its associated cost.
                new_state = self.space.neighbor(state)
                new_cost = self.space.cost(new_state)
                acceptance = self.acceptance_probability(cost, new_cost, temperature)
                coin = np.random.rand()

                if callback:  # this is for additional data gathering and analysis by the user
                    callback(
                        iteration=iteration,
                        current_state=state,
                        current_cost=cost,
                        new_state=new_state,
                        new_cost=new_cost,
                        acceptance_probability=acceptance,
                        coin_flip=coin,
                        current_temperature=temperature
                    )

                if coin < acceptance:
                    # flip a coin. if heads, accept the new state. otherwise, proceed as before.
                    state, cost = new_state, new_cost
            # lower the temperature
            temperature *= self.decay
        return state, cost
