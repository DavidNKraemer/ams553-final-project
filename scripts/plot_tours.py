import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.tours import DroneTour
from src.tour_generation import ModifiedKroeseTourGenerator, MultiTSPTourGenerator
from src.annealing import Annealer
from src.cross_entropy import CrossEntropy
from src.brute_force import find_optimal_team_tour
from src.utils import plot_policies


# Parameters
sns.set_context("talk")

num_sites, num_drones = 10, 4
sites = np.random.rand(num_sites, 2)

# tour_generator = ModifiedKroeseTourGenerator(num_sites, num_drones)
tour_generator = MultiTSPTourGenerator(num_sites, num_drones)

drone_tour = DroneTour(sites, num_drones, tour_generator)

random_tour = drone_tour.generate_state()


# Simulated Annealing
annealer = Annealer(drone_tour)

annealed_tour, _ = annealer.anneal()

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
plot_policies(random_tour, sites, axis=axes[0])
plot_policies(annealed_tour, sites, axis=axes[1])

fig.suptitle("Uniformly distributed sites on $[0,1]^2$")
axes[0].set_title(f"Random {num_drones}-tour")
axes[1].set_title(f"Annealed {num_drones}-tour")

for ax in axes:
   sns.despine(fig=fig, ax=ax, top=True, right=True, left=True, bottom=True, trim=True)
   ax.set_axis_off()

fig.savefig('../plots/annealed_tours.png', bbox_inches='tight', transparent=True)


# Cross Entropy
x_entropy_sample_size = 50
x_entropy_quantile = 0.1
x_entropy = CrossEntropy(drone_tour, x_entropy_sample_size, x_entropy_quantile, iterations=100)
x_entropy.minimize()
x_entropy_tour = drone_tour.generate_state()

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
plot_policies(random_tour, sites, axis=axes[0])
plot_policies(x_entropy_tour, sites, axis=axes[1])

fig.suptitle(f"{num_sites} Uniformly distributed sites on $[0,1]^2$")
axes[0].set_title(f"Random {num_drones}-tour")
axes[1].set_title(f"X-Entropy {num_drones}-tour (MultiTSP)")

for ax in axes:
   sns.despine(fig=fig, ax=ax, top=True, right=True, left=True, bottom=True, trim=True)
   ax.set_axis_off()


fig.savefig('../plots/cross_entropy_tours.png', bbox_inches='tight', transparent=True)

# Brute Force Method
optimal_tour, _ = find_optimal_team_tour(sites, num_drones)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
plot_policies(random_tour, sites, axis=axes[0])
plot_policies(optimal_tour, sites, axis=axes[1])

fig.suptitle("Uniformly distributed sites on $[0,1]^2$")
axes[0].set_title(f"Random {num_drones}-tour")
axes[1].set_title(f"Optimal {num_drones}-tour")

for ax in axes:
   sns.despine(fig=fig, ax=ax, top=True, right=True, left=True, bottom=True, trim=True)
   ax.set_axis_off()

fig.savefig('../plots/optimal_tours.png', bbox_inches='tight', transparent=True)