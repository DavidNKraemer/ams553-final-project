from math import ceil

class Distribution:
    def __init__(self, density, state_space):
        self.density = density
        self.state_space = state_space

    def sample(self, v):
        pass

    def pdf(self, v, state):
        return self.density(v, state)



class CrossEntropy:
    def __init__(self, f, n, reward, rho):
        self.f = f
        self.n = n
        self.reward = reward
        self.rho = rho # rho must be nonzero

    def maximize(self, v0):
        v = v0
        for i in range(100):
            samples = [self.f.sample(v) for _ in range(self.n)]
            rewards = [self.reward(x) for x in samples]
            rewards.sort()
            gamma = rewards[self.n - int(ceil(self.rho * self.n))]
            v = self.compute_argmax(gamma, samples)

        return v