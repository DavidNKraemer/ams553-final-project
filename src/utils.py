import numpy as np
import matplotlib.pyplot as plt


def plot_policies(tours, sites, axis=plt.gca()):
    """
    :param tours: a given k-tour of the point set
    :type tours: list[list]
    :param sites: a two dimensional point set
    :type sites: numpy.ndarray
    :param axis: the object that gets the plot drawn on it.
    :type axis: matplotlib.axes.Axes

    :return: n/a (called for side effects)
    :rtype: None

    Given a set of tours and a two dimensional point set, plot the sites, and draw the tours.
    The axis that is specified is where the plot is actually drawn on.
    """
    # plot the point set
    axis.plot(*sites.T, 'o')

    # label each point 0, ..., sites.shape[0] - 1
    for i in range(sites.shape[0]):
        axis.annotate(str(i), sites[i])

    for drone, tour in tours.items():
        # for each tour, draw a polygon on the relevant sites
        axis.fill(*np.array([sites[x] for x in tour]).T, fill=False)

    axis.grid(False)