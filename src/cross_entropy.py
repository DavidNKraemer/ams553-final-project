from math import ceil


class CrossEntropy:
    """

    """

    def __init__(self, state_space, sample_size, quantile, iterations=100):
        self.state_space = state_space
        self.sample_size = sample_size
        self.quantile = quantile  # quantile must be nonzero
        self.iterations = iterations

    def minimize(self):
        parameters = self.state_space.get_parameters()
        for _ in range(self.iterations):
            samples = [self.state_space.generate_state() for _ in range(self.sample_size)]

            # compute the scores associated with the samples and put them in descending (low = good) order
            scores = [self.state_space.cost(state) for state in samples]
            scores.sort(reverse=True)

            # find the upper (1-self.quantile)% threshold of the scores
            threshold = scores[self.sample_size - int(ceil(self.quantile * self.sample_size))]

            # update the parameters of the state space for the next round
            parameters = self.state_space.estimate_parameters(threshold, samples, scores)
            self.state_space.set_parameters(parameters)
