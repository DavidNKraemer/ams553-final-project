import numpy as np
from itertools import product
from src.utils import uniform_transition, uniform_initial, indicator
import abc


def block_site_and_normalize(matrix, site):
    """
    :param matrix: a (possibly sub-)stochastic transition_matrix
    :type matrix: numpy.ndarray
    :param site: the current site to update in the transition_matrix
    :type site: int
    :param drone: the current drone to possibly update
    :type drone: int
    :param drone_distribution: the distribution of drone draws, which may need to be updated
    :type drone_distribution: numpy.ndarray
    :return:

    Given a stochastic transition_matrix , the current site, a drone, and the distribution of drones, update the
    transition_matrix so that the current site can no longer be reached. If a resulting row has no more positive
    entries, it is no longer reachable, and therefore isn't normalized. In this case, the drone can no longer
    proceed, and so the drone distribution is updated to make future draws ignore the given drone.
    """

    # zero out the column associated with the current site
    matrix[:, site] = 0.
    normalization = np.sum(matrix, axis=1)[:, None]  # pre-compute all normalization values

    for row, normalizer in enumerate(normalization):
        if normalizer > 0.:  # only normalize reachable (i.e., nonzero) rows.
            matrix[row, :] /= normalizer


class TourGenerator(abc.ABC):
    """
    An abstract class to house drone tour generation methods.

    We want to compare cross entropy across different parameterizations of the space of all drone tours, and so the
    easiest way to accomplish this is by extracting the logic for generating tours into their own data structure, and
    pass it into a new drone tour object.

    The TourGenerator class implements no methods --- they must all be supplied by the subclasses.
    """

    @abc.abstractmethod
    def generate_tour(self):
        """
        Generate a drone tour according to the current parameterization of the space of all tours.

        :return:
        """
        pass

    @abc.abstractmethod
    def pmf(self, state):
        """
        Compute the probability mass function associated with a particular state in view of the current parameterization
        of the distribution on the space of all tours.

        :param state: a drone tour trajectory
        :type state: dict[int, list]
        :return:
        """
        pass

    @abc.abstractmethod
    def estimate_parameters(self, threshold, samples, scores):
        """
        :param threshold: a specified score that the samples must beat to be considered
        :type threshold: float
        :param samples: a list of drone tour trajectories
        :type samples: list[dict[int,list]]
        :param scores: a list of associated scores for the drone tour trajectories
        :type score: list[float]
        :return:

        Estimates parameters that maximize the odds of beating the threshold given the current crop of samples and their
        associated scores
        """
        pass

    @abc.abstractmethod
    def get_parameters(self):
        """
        Package the parameters as a dictionary and pass them as return values.
        :return:
        """
        pass

    @abc.abstractmethod
    def set_parameters(self, parameters):
        """
        Update the parameters from a dictionary of inputs.
        :param parameters: A dictionary of all relevant parameters
        :type parameters: dict
        :return:
        """
        pass


class ModifiedKroeseTourGenerator(TourGenerator):
    """
    A tour generation and parameterization technique which is based on the cross entropy implementation in
    Rubinstein and Kroese (2004) for the traveling salesman problem.
    """

    def __init__(self, num_sites, num_drones, transition_matrix=None, initial_distribution=None, theta=0.5, replace=True):
        """
        Initializer method.

        :param num_sites: the number of sites which are to be visited by the drone tours
        :type num_sites: int
        :param num_drones:  the number of available drones
        :type num_drones: int
        :param transition_matrix: the stochastic matrix which dictates the transition betweeen sites for the drones
        :type transition_matrix: np.ndarray
        :param initial_distribution: the distribution of sites for the starting positions of the drones
        :type initial_distribution: np.ndarray
        :param theta: update smoothing parameter
        :type theta: float
        :param replace: whether initial sites are selected independently (with replacement) or in sequence (without replacement)
        :type replace: bool
        """
        self.num_drones = num_drones
        self.transition_matrix = transition_matrix or uniform_transition([i for i in range(num_sites)])
        self.initial_distribution = initial_distribution or uniform_initial([i for i in range(num_sites)])
        self.theta = theta
        self.replace = replace

    def generate_tour(self):
        """
        :return: A k-drone tour of list(range(len(initial_dist))) that follows the probabilities in transition_matrix
        :rtype: dict[list]

        Generates a drone trajectory according to a (possibly sub-)stochastic transition_matrix and initial site distribution, along
        with a specified number of drones to draw.

        See Rubinstein and Kroese (2004) for a description on how to generate such trajectories for the TSP (i.e., 1-drone
        case).
        """

        # the set of drones and sites being drawn from
        transitions = self.transition_matrix.copy()
        init_dist = self.initial_distribution.copy()

        available_drones = list(range(self.num_drones))
        available_sites = list(range(len(transitions)))

        drone_probabilities = uniform_initial(available_drones)

        # starting sites
        sites = np.random.choice(available_sites, size=self.num_drones, p=init_dist, replace=self.replace)

        # trajectories data structure
        trajectories = {drone: [sites[drone]] for drone in available_drones}

        # normalize after drawing initial starting sites
        for drone in trajectories:
            block_site_and_normalize(transitions, trajectories[drone][-1])

        # loop until the transition_matrix is all zeros
        while np.any(drone_probabilities > 0.):
            drone_id = np.random.choice(available_drones, p=drone_probabilities)  # choose a drone
            current_site = trajectories[drone_id][-1]
            if np.sum(transitions[current_site, :]) == 0.:
                drone_probabilities[drone_id] = 0.
                if np.any(drone_probabilities > 0.):
                    drone_probabilities /= np.sum(drone_probabilities)
            else:
                next_site = np.random.choice(available_sites, p=transitions[current_site, :])  # choose the next site
                trajectories[drone_id].append(next_site)
                block_site_and_normalize(transitions, next_site)

        return trajectories

    def estimate_parameters(self, threshold, samples, scores):
        """
        :param threshold: a specified score that the samples must beat to be considered
        :type threshold: float
        :param samples: a list of drone tour trajectories
        :type samples: list[dict[int,list]]
        :param scores: a list of associated scores for the drone tour trajectories
        :type score: list[float]
        :return: a dictionary of the estimated parameters
        :rtype: dict, containing the keys
            - 'initial_distribution' (np.ndarray)
            - 'transition_matrix' (np.ndarray)

        Estimates parameters that maximize the odds of beating the threshold given the current crop of samples and their
        associated scores. The estimation is specifically determined by solving a Lagrange multiplier problem on the
        cross entropy function.
        """
        sites = len(self.initial_distribution)

        # compute the new initial distribution parameter
        initial_distribution = np.empty(self.initial_distribution.shape)
        for site in range(sites):
            numerator = sum(
                indicator(score <= threshold) *
                indicator(any(site == sample[drone][0] for drone in sample))
                for score, sample in zip(scores, samples)
            )
            denominator = sum(
                indicator(score <= threshold) *
                indicator(any(alt_site == sample[drone][0] for drone in sample))
                for score, sample in zip(scores, samples)
                for alt_site in range(sites)
            )
            initial_distribution[site] = (numerator / denominator) if denominator != 0. else 0.

        # compute the new transition transition_matrix
        transition_matrix = np.empty(self.transition_matrix.shape)
        for current_site, next_site in product(range(sites), repeat=2):
            numerator = sum(
                indicator(
                    (score <= threshold) and
                    (sample[drone][r] == current_site and sample[drone][(r + 1) % len(sample[drone])] == next_site)
                )
                for score, sample in zip(scores, samples)
                for drone in range(self.num_drones)
                for r in range(len(sample[drone]))
            )
            denominator = sum(
                indicator(
                    score <= threshold
                ) * indicator(
                    sample[drone][r] == current_site and sample[drone][(r + 1) % len(sample[drone])] == site
                )
                for site in range(sites)
                for score, sample in zip(scores, samples)
                for drone in range(self.num_drones)
                for r in range(len(sample[drone]))
            )
            transition_matrix[current_site, next_site] = (numerator / denominator) if denominator != 0. else 0.

        return {'initial_distribution': initial_distribution, 'transition_matrix': transition_matrix}

    def pmf(self, state):
        """
        :param state: A specific k-drone tour
        :type state: dict[list[int]]
        :return: The probability mass associated with the given state.
        :rtype: float

        Computes the pmf of a given k-drone tour according the the transition probability transition_matrix and initial
        distribution of sites.

        The pmf is the product of the initial probabilities of starting states with the product of the transition
        probabilities of the tours.
        """
        probability = 1.

        for drone in state:
            tour = state[drone]
            probability *= self.initial_distribution[tour[0]]
            for (i, site) in enumerate(tour):
                next_site = tour[(i + 1) % len(tour)]
                probability *= self.transition_matrix[site, next_site]

        return probability

    def set_parameters(self, parameters):
        """
        :param parameters: A dictionary of the DroneTour parameters
        :type parameters: dict
        :return: N/A [called for side effects]

        Sets the current StateSpace probability distribution parameters to the provided parameters.

        The dictionary parameters has the following keys:
        - 'initial_distribution': a numpy.ndarray; the initial distribution of sites
        - 'transition_matrix': a numpy.ndarray; the transition probabilities on the sites
        """
        self.initial_distribution = self.theta * self.initial_distribution + (1-self.theta) * parameters['initial_distribution']
        self.transition_matrix = self.theta * self.transition_matrix + (1-self.theta) * parameters['transition_matrix']

    def get_parameters(self):
        """
        :return: The initial distribution of sites and the transition transition_matrix packaged into a dictionary
        :rtype: dict

        Retrieves the current probability distribution parameters of the StateSpace

        The keys in the dictionary are
        - 'initial_distribution': a numpy.ndarray; the initial distribution of sites
        - 'transition_matrix': a numpy.ndarray; the transition probabilities on the sites
        """
        return {'initial_distribution': self.initial_distribution, 'transition_matrix': self.transition_matrix}


class MultiTSPTourGenerator(TourGenerator):
    """
    A TourGenerator in which each drone is given its own transition matrix to update. These updates occur independently
    from the other drones, which is the main difference between MultiTSPTourGenerator and ModifiedKroeseTourGenerator
    """

    def __init__(self, num_sites, num_drones, transition_matrices={}, initial_distributions={}, theta=0.5):
        """
        Initializer method.

        :param num_sites: the number of sites which are to be visited by the drone tours
        :type num_sites: int
        :param num_drones:  the number of available drones
        :type num_drones: int
        :param transition_matrices: the set of stochastic matrices which dictate the transition between sites for each drone
        :type transition_matrices: dict[int, np.ndarray]
        :param initial_distributions: the set of distributions of sites for the starting positions of each drone
        :type initial_distribution: dict[int, np.ndarray]
        :param theta: update smoothing parameter
        :type theta: float
        :param replace: whether initial sites are selected independently (with replacement) or in sequence (without replacement)
        :type replace: bool
        """
        self.num_sites = num_sites
        self.num_drones = num_drones
        self.site_indices = [i for i in range(self.num_sites)]
        self.transition_matrices = {i: transition_matrices.get(i) or uniform_transition(self.site_indices) for i in range(self.num_drones)}
        self.initial_distributions = {i: initial_distributions.get(i) or uniform_initial(self.site_indices) for i in range(self.num_drones)}
        self.theta = theta

    def pmf(self, state):
        """
        :param state: A specific k-drone tour
        :type state: dict[list[int]]
        :return: The probability mass associated with the given state.
        :rtype: float

        Computes the pmf of a given k-drone tour according the the transition probability transition_matrix and initial
        distribution of sites.

        The pmf is the product of the initial probabilities of starting states with the product of the transition
        probabilities of the tours.
        """
        pass

    def estimate_parameters(self, threshold, samples, scores):
        """
        :param threshold: a specified score that the samples must beat to be considered
        :type threshold: float
        :param samples: a list of drone tour trajectories
        :type samples: list[dict[int,list]]
        :param scores: a list of associated scores for the drone tour trajectories
        :type score: list[float]
        :return: a dictionary of the estimated parameters
        :rtype: dict, containing the keys
            - 'initial_distribution' (np.ndarray)
            - 'transition_matrix' (np.ndarray)

        Estimates parameters that maximize the odds of beating the threshold given the current crop of samples and their
        associated scores. The estimation is specifically determined by solving a Lagrange multiplier problem on the
        cross entropy function.
        """
        init_dists = {i: np.empty(self.initial_distributions[i].shape) for i in range(self.num_drones)}
        transition_mats = {i: np.empty(self.transition_matrices[i].shape) for i in range(self.num_drones)}

        for drone in init_dists:


            for site in range(self.num_sites):
                numerator = sum(
                    indicator(score <= threshold) *
                    indicator(sample[drone][0] == site)
                    for score, sample in zip(scores, samples)
                )
                denominator = sum(
                    indicator(score <= threshold) *
                    sum(
                        indicator(sample[drone][0] == alt_site)
                        for alt_site in range(self.num_sites)
                    )
                    for score, sample in zip(scores, samples)
                )
                init_dists[drone][site] = (numerator / denominator) if denominator != 0. else 0.

        for drone in transition_mats:

            for current_site, next_site in product(range(self.num_sites), repeat=2):
                numerator = sum(
                    indicator(score <= threshold) *
                    sum(
                        indicator(
                            sample[drone][r] == current_site and sample[drone][(r+1) % len(sample[drone])] == next_site
                        )
                        for r in range(len(sample[drone]))
                    )
                    for score, sample in zip(scores, samples)
                )
                denominator = sum(
                    indicator(score <= threshold) *
                    sum(
                        indicator(
                            sample[drone][r] == current_site and sample[drone][(r+1) % len(sample[drone])] == alt_site
                        )
                        for alt_site in range(self.num_sites)
                        for r in range(len(sample[drone]))
                    )
                    for score, sample in zip(scores, samples)
                )

                transition_mats[drone][current_site, next_site] = (numerator / denominator) if denominator != 0. else 0.

        return {'initial_distributions': init_dists, 'transition_matrices': transition_mats}


    def get_parameters(self):
        """
        :return: The initial distribution of sites and the transition transition_matrix packaged into a dictionary
        :rtype: dict

        Retrieves the current probability distribution parameters of the StateSpace

        The keys in the dictionary are
        - 'initial_distribution': a numpy.ndarray; the initial distribution of sites
        - 'transition_matrix': a numpy.ndarray; the transition probabilities on the sites
        """
        return {'initial_distributions': self.initial_distributions, 'transition_matrices': self.transition_matrices}

    def set_parameters(self, parameters):
        """
        :param parameters: A dictionary of the DroneTour parameters
        :type parameters: dict
        :return: N/A [called for side effects]

        Sets the current StateSpace probability distribution parameters to the provided parameters.

        The dictionary parameters has the following keys:
        - 'initial_distributions': a dict[int, numpy.ndarray]; the initial distribution of sites
        - 'transition_matrices': a dict[int, numpy.ndarray]; the transition probabilities on the sites
        """
        self.transition_matrices = parameters['transition_matrices']
        self.initial_distributions = parameters['initial_distributions']

    def generate_tour(self):
        """
        :return: A k-drone tour of list(range(len(initial_dist))) that follows the probabilities in the transition
                 matrices associated with the current state parameters
        :rtype: dict[list]

        Generates a drone trajectory according to a set of (possibly sub-)stochastic transition matrices and initial
        site distributions.

        See Rubinstein and Kroese (2004) for a description on how to generate such trajectories for the TSP
        (i.e., 1-drone case).
        """
        transitions = {drone: self.transition_matrices[drone].copy() for drone in self.transition_matrices}
        init_dists = {drone: self.initial_distributions[drone].copy() for drone in self.initial_distributions}

        site_labels = [site for site in range(self.num_sites)]
        drones = [drone for drone in range(self.num_drones)]

        remaining_sites = set(site_labels)

        trajectories = {drone: [] for drone in range(self.num_drones)}

        # choose initial starting sites
        for drone, trajectory in trajectories.items():
            trajectory.append(np.random.choice(site_labels, p=init_dists[drone]))
            remaining_sites.discard(trajectory[-1])

        while len(remaining_sites) > 0 and len(drones) > 0:
            drone_id = np.random.choice(drones)
            current_site = trajectories[drone_id][-1]
            if np.sum(transitions[drone_id][current_site, :]) == 0.:
                drones.remove(drone_id)
            else:
                next_site = np.random.choice(site_labels, p=transitions[drone_id][current_site, :])
                remaining_sites.discard(next_site)
                trajectories[drone_id].append(next_site)
                block_site_and_normalize(transitions[drone_id], next_site)

        return trajectories
