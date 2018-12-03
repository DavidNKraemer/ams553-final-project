# -*- coding: utf-8 -*-
"""
Abstract definition of the state space for simulated annealing and cross entropy minimization.
"""
import abc


class StateSpace(abc.ABC):
    """
    Class definition for a state space defined in abstract terms.

    A StateSpace is specified by a description of each internal state, a method for computing the
    cost associated with each state, a method for computing neighbors of a given state, and a method
    for generating a random state.
    """

    class State(abc.ABC):
        """
        The specification of the embedded state in the container state space.
        """
        pass

    @abc.abstractmethod
    def cost(self, state: "State") -> float:
        """
        :param state: The state at which the cost will be computed.
        :type state: StateSpace.State

        :return: The associated cost for the given state
        :rtype: float
        """
        raise NotImplementedError

    @abc.abstractmethod
    def neighbor(self, state: "State") -> "State":
        """
        :param state: The state at which a neighbor is generated.
        :type state: StateSpace.State

        :return: A neighbor of the given state
        :rtype: StateSpace.State
        """
        raise NotImplementedError

    @abc.abstractmethod
    def generate_state(self) -> "State":
        """
        :return: A random state in the state space
        :rtype: StateSpace.Space

        Returns a state from the StateSpace
        """
        raise NotImplementedError

    @abc.abstractmethod
    def pmf(self, state: "State") -> float:
        """
        :param state: A state in the state space
        :type state: StateSpace.State
        :return: The probability mass function associated with the state
        :rtype: float

        Computes the pmf associated with a given state in the StateSpace
        """
        raise NotImplementedError

    @abc.abstractmethod
    def estimate_parameters(self, threshold, samples, scores):
        """
        :param threshold: a score threshold for comparing against the sample scores
        :type threshold: float
        :param samples: a set of samples generated from cross entropy
        :type samples: list[StateSpace.State]
        :param scores: a set of scores associated with each sample in samples
        :type scores: list[float]

        :return: cross entropy-minimizing parameters given the available data
        :rtype: dict

        Estimates and returns the cross entropy-minimizing parameters from the available sample data. See the references
        to CrossEntropy.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_parameters(self):
        """
        :return: the probability distribution parameters
        :rtype: dict

        Retrieves the current probability distribution parameters of the StateSpace
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_parameters(self, parameters):
        """
        :param parameters:
        :return: N/A [called for side effects]

        Sets the current StateSpace probability distribution parameters to the provided parameters.
        """
        raise NotImplementedError
