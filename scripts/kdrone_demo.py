import sys
import numpy as np

import src.tours as tours

sys.path.insert(0, '../')

n = 10
sites = np.random.rand(n, 2)

drone_tour = tours.DroneTour(sites, 4)

trials = 100
for _ in range(trials):
    tour = drone_tour.generate_state()
    print(tour, drone_tour.pmf(tour))
