import sys
import numpy as np

import src.tours as tours
from src.cross_entropy import CrossEntropy

sys.path.insert(0, '../')

n = 10
sites = np.random.rand(n, 2)

drone_tour = tours.DroneTour(sites, 4)

ce_sample_size = 10
ce_quantile = 0.1

ce_estimator = CrossEntropy(drone_tour, ce_sample_size, ce_quantile)

ce_estimator.minimize()

print(drone_tour.initial_distribution)
print(drone_tour.transition_matrix)
