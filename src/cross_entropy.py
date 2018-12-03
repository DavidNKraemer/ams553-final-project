from math import ceil


class CrossEntropy:
    """
    A general class definition of the cross entropy method.

    References
    ----------

    - https://en.wikipedia.org/wiki/Cross-entropy_method
    - Rubinstein, Reuven Y. and Kroese, Dirk P. *The cross-entropy method: a unified approach to combinatorial
      optimization, Monte-Carlo simulation and machine learning* (2004). Springer.
    """

    def __init__(self, state_space, sample_size, quantile, iterations=100):
        """
        :param state_space: an underlying state space with a parameterized probability distribution over its states.
        :type state_space: src.space.StateSpace
        :param sample_size: the number of sample states which are drawn and analyzed in each iteration of the method.
        :type sample_size: int
        :param quantile: the upper-cutoff quantile for examining high performers in each iteration of the method.
        :type quantile: float
        :param iterations: [default=100] the number of iterations employed by the method.
        :type iterations: int

        Initializer method for the CrossEntropy object
        """
        self.state_space = state_space
        self.sample_size = sample_size
        self.quantile = quantile  # quantile must be nonzero
        self.iterations = iterations

    def minimize(self, callback=None):
        """
        :param callback: Function
        :type callback: Function(**kwargs)
        :return: N/A [called for side effects]

        Performs the cross entropy method for minimizing an objective function over a parameterized state space.
        The keyword arguments passed to callback are
            - sample_states, the current sample of states in the given round of annealing
            - sample_scores, the scores associated with the sampled states in the given round of annealing
            - threshold, the scores which form the (1-quantile) quantile of the sample scores
            - distribution_parameters, the current parameters of the distribution of states in the state space
        Cross entropy modifies the parameters of the underlying state space, but does not return anything in particular.
        """
        for _ in range(self.iterations):
            samples = [self.state_space.generate_state() for _ in range(self.sample_size)]

            # compute the scores associated with the samples and put them in descending (low = good) order
            scores = [self.state_space.cost(state) for state in samples]
            scores.sort(reverse=True)

            # find the upper (1-self.quantile)% threshold of the scores
            threshold = scores[self.sample_size - int(ceil(self.quantile * self.sample_size))]

            # update the parameters of the state space for the next round
            parameters = self.state_space.estimate_parameters(threshold, samples, scores)

            if callback:  # this is for additional data gathering and analysis by the user
                callback(
                    sample_states=samples,
                    sample_scores=scores,
                    threshold=threshold,
                    distribution_parameters=parameters,
                )

            self.state_space.set_parameters(parameters)
