import numpy as np

import src.tours as tours
from src.annealing import Annealer


# drone tour parameters
num_sites, num_drones = 10, 4
sites = np.random.rand(num_sites, 2)  # the 2 indicates we are sampling sites from the plane

drone_tour = tours.DroneTour(sites, num_drones)

# cross entropy parameters
annealer = Annealer(drone_tour)


# an example way to use the callback -- see below for details
tours_drawn = []


def callback(**kwargs):
    """

    :param kwargs:
    The keyword arguments passed to callback are
            - current state, the state associated with the current annealing iteration
            - current_cost, the cost associated with the current state
            - new_state, a new state drawn in the current iteration
            - new_cost, the cost associated with the new state
            - acceptance_probability, the probability that the annealer will accept the new state as the next state in
                                      the current iteration
            - coin_flip, the value of the coin flip associated with the current iteration
            - current_temperature, the temperature associated with the current iteration
    :return:
    """
    tours_drawn.append(kwargs['current_state'])  # for example, maybe we want a list of all samples drawn during
                                                   # the run of cross entropy


annealer.anneal(callback=callback)

for tour in tours_drawn:
    print(tour)
