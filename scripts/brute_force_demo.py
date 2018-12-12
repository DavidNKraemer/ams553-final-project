import numpy as np
import src.tours as tours
import src.brute_force as bf
import matplotlib.pyplot as plt

from src.utils import plot_policies

# Drone tour parameters
num_sites, num_drones = 11, 4
sites = np.random.rand(num_sites, 2)  # the 2 indicates we are sampling sites from the plane




