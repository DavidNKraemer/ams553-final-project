import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.tours import DroneTour
from src.annealing import Annealer
from src.cross_entropy import CrossEntropy
from src.utils import plot_policies

sns.set_context("talk")

num_sites, num_drones = 10, 4
sites = np.random.rand(num_sites, 2)

drone_tour = DroneTour(sites, num_drones)

random_tour = drone_tour.generate_state()

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
axes[1].set_title(f"X-Entropy {num_drones}-tour")

for ax in axes:
   sns.despine(fig=fig, ax=ax, top=True, right=True, left=True, bottom=True, trim=True)
   ax.set_axis_off()

fig.savefig('../plots/cross_entropy_tours.png', bbox_inches='tight', transparent=True)

