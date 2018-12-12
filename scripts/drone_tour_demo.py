import numpy as np

import src.tours as tours


def stochastic_matrix(matrix):
    """
    :param matrix: The nonnegative transition_matrix to be normalized.
    :return: a normalized version of transition_matrix

    Given a nonnegative transition_matrix with positive row sums, return a normalized version so that the rows sum to 1.
    """
    return matrix / np.sum(matrix, axis=1)[:, None]


# drone tour parameters
num_sites, num_drones = 10, 4
sites = np.random.rand(num_sites, 2)  # 2 indicates that the sites are drawn from the plane

drone_tour = tours.DroneTour(sites, num_drones)

# here's how to generate an arbitrary tour
tour = drone_tour.generate_state()

# here's the cost associated with the tour
cost = drone_tour.cost(tour)

# here's the probability mass function of the tour
prob = drone_tour.pmf(tour)

# here's how to access the transition transition_matrix
transition = drone_tour.transition_matrix

# here's how to change the transition transition_matrix
new_transition_matrix = stochastic_matrix(np.random.rand(num_sites, num_sites))
drone_tour.transition_matrix = new_transition_matrix

# here's how to access the initial distribution of sites
initial_distribution = drone_tour.initial_distribution

# here's how to change the initial distribution of sites
new_initial_distribution = np.random.rand(num_sites)
new_initial_distribution /= np.sum(new_initial_distribution)
drone_tour.initial_distribution = new_initial_distribution
