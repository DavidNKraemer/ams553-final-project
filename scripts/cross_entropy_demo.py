import numpy as np

import src.tours as tours
from src.cross_entropy import CrossEntropy

# drone tour parameters
num_sites, num_drones = 10, 4
sites = np.random.rand(num_sites, 2)  # the 2 indicates we are sampling sites from the plane

drone_tour = tours.DroneTour(sites, num_drones)

# cross entropy parameters
x_entropy_sample_size = 1
x_entropy_quantile = 0.1  # specifies the 90% percentile

x_entropy_estimator = CrossEntropy(drone_tour, x_entropy_sample_size, x_entropy_quantile)

# an example way to use the callback -- see below for details
samples_drawn = []


def callback(**kwargs):
    """

    :param kwargs:
    The keyword arguments passed to callback are
            - iteration, the current iteration
            - sample_states, the current sample of states in the given round of annealing
            - sample_scores, the scores associated with the sampled states in the given round of annealing
            - threshold, the scores which form the (1-quantile) quantile of the sample scores
            - distribution_parameters, the current parameters of the distribution of states in the state space
    :return:
    """
    samples_drawn.append(kwargs['sample_states'])  # for example, maybe we want a list of all samples drawn during
    # the run of cross entropy


x_entropy_estimator.minimize(callback=callback)

for sample in samples_drawn:
    for tour in sample:
        print(tour)
