from math import ceil


class CrossEntropy:
    """

    """

    def __init__(self, state_space, sample_size, quantile, iterations=100):
        self.state_space = state_space
        self.sample_size = sample_size
        self.quantile = quantile  # quantile must be nonzero
        self.iterations = iterations

    def minimize(self, callback=None):
        """

        :param callback: Function
        The keyword arguments passed to callback are
            - sample_states, the current sample of states in the given round of annealing
            - sample_scores, the scores associated with the sampled states in the given round of annealing
            - threshold, the scores which form the (1-quantile) quantile of the sample scores
            - distribution_parameters, the current parameters of the distribution of states in the state space
        :return:
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

            if callback:
                callback(
                    sample_states=samples,
                    sample_scores=scores,
                    threshold=threshold,
                    distribution_parameters=parameters,
                )

            self.state_space.set_parameters(parameters)
