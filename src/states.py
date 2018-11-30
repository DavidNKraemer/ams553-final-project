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
        """
        raise NotImplementedError
