import numpy as np

import src.tours as tours
from src.cross_entropy import CrossEntropy
from src.tour_generation import ModifiedKroeseTourGenerator, MultiTSPTourGenerator, MultiDronePathGenerator

# drone tour parameters
num_sites, num_drones = 5, 2
sites = np.random.rand(num_sites, 2)  # the 2 indicates we are sampling sites from the plane

tour_generator = ModifiedKroeseTourGenerator(num_sites, num_drones, replace=False)
alt_tour_generator = MultiTSPTourGenerator(num_sites, num_drones)
path_generator = MultiDronePathGenerator(num_sites, num_drones)

drone_tour = tours.DroneTour(sites, num_drones, tour_generator)
alt_drone_tour = tours.DroneTour(sites, num_drones, alt_tour_generator)
drone_path = tours.DroneTour(sites, num_drones, path_generator)

# cross entropy parameters
x_entropy_sample_size = 1
x_entropy_quantile = 0.1  # specifies the 90% percentile

x_entropy_estimator = CrossEntropy(drone_tour, x_entropy_sample_size, x_entropy_quantile)
alt_x_entropy_estimator = CrossEntropy(alt_drone_tour, x_entropy_sample_size, x_entropy_quantile)

# an example way to use the callback -- see below for details
samples_drawn = []
alt_samples_drawn = []


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
alt_x_entropy_estimator.minimize(callback=callback)

for sample, alt_sample in zip(samples_drawn, alt_samples_drawn):
    for tour, alt_tour in zip(sample, alt_sample):
        print("v1: ", tour)
        print("v2: ", alt_tour)
