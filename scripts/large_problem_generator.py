import numpy as np
import scipy.stats as stats
from collections import defaultdict
import pandas as pd
from src.annealing import Annealer
from src.cross_entropy import CrossEntropy
from src.tours import DroneTour
from src.tour_generation import ModifiedKroeseTourGenerator
from src.brute_force import find_optimal_team_tour

# experiment parameters
num_prob_instances = 25
num_runs = 25
num_sites = 30
binom_prob = 0.5

x_entropy_sample_size = 5
x_entropy_upper_quantile = 0.2

# site distributions
uniform_site_dist = lambda n: np.random.rand(n, 2)
exp_site_dist = lambda n: -np.log(np.random.rand(n,2) * (1 - np.power(np.e, -1.))) / 10.  # TODO: fix this garbage
gaussian_site_dist = lambda n: np.array([0.5, 0.5]) + 0.1 * stats.truncnorm.rvs(-5, 5, size=(n,2))
site_distributions = [uniform_site_dist, exp_site_dist, gaussian_site_dist]

# drone distributions
uniform_drone_dist = lambda: np.random.randint(1, num_sites)
binom_drone_dist = lambda: np.random.binomial(num_sites-2, binom_prob) + 1
drone_distributions = [uniform_drone_dist, binom_drone_dist]

data_dict = defaultdict(list)
id = 0

# main loop
for site_dist_id, site_dist in enumerate(site_distributions):
    for drone_dist_id, drone_dist in enumerate(drone_distributions):

        sites = site_dist(num_sites)
        num_drones = drone_dist()

        for run in range(num_runs):
            data_dict[id] += [site_dist_id, drone_dist_id, run]

            tour_generator = ModifiedKroeseTourGenerator(num_sites, num_drones)
            drone_tour = DroneTour(sites, num_drones, tour_generator)

            annealer = Annealer(drone_tour)
            annealed_tour, annealed_cost = annealer.anneal()

            data_dict[id].append(annealed_tour)
            data_dict[id].append(annealed_cost)

            cross_entropy = CrossEntropy(drone_tour, sample_size=x_entropy_sample_size, quantile=x_entropy_upper_quantile)
            cross_entropy.minimize()

            data_dict[id].append(drone_tour.generate_state())
            data_dict[id].append(drone_tour.cost(data_dict[id][-1]))

            id += 1


data_df = pd.DataFrame.from_dict(
    data_dict,
    orient='index',
    columns=[
        "site_dist_id", "drone_dist_id", "run",
        "anneal_tour", "anneal_cost", "xentropy_tour", "xentropy_cost"
    ]
)

data_df.to_csv(f"../data/large_problem_data_{num_prob_instances}_{num_runs}_{num_sites}.csv", index=False)