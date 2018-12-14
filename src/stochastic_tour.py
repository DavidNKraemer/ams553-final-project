
import numpy as np
import random
from collections import namedtuple


def generate_prob_matrix(n):
    matrix = np.random.rand(n, n)

    for i in range(n):
        matrix[i][i] = 0

    for i in range(n):
        matrix[i] = (1/np.sum(matrix[i]))*matrix[i]

    return matrix


def categorical(p):
    return np.random.choice(len(p), 1, p=p)[0]


Drone = namedtuple('Drone', 'speed probability')
Site = namedtuple('Site', 'location')


class System:

    def __init__(self, sites, drones):
        self.sites = {}
        self.drones = {}
        n = len(sites)

        for i, drone in enumerate(drones):
            self.drones[i] = drone

        for i, site in enumerate(sites):
            self.sites[i] = site

        distance = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                if i < j:
                    x = np.subtract(sites[i], sites[j])
                    d = np.linalg.norm(x)
                    distance[i][j] = d
                    distance[j][i] = d
        self.distance = distance

    def get_site(self, site_id):
        return self.sites[site_id]

    def get_drone(self, drone_id):
        return self.drones[drone_id]

    def compute_path_distance(self, path):
        n = len(path)
        d = 0
        for i in range(n - 1):
            d += self.distance[path[i]][path[i + 1]]
        return d

    def compute_path_time(self, path, drone_id):
        d = self.compute_path_distance(path)
        return d/self.get_drone(drone_id).speed

    def generate_path_of_length(self, length, drone_id):
        path = []
        P = self.get_drone(drone_id).probability
        num_sites = len(self.sites)
        s = categorical([1/num_sites]*num_sites)
        path.append(s)
        site = s
        for i in range(length):
            site = categorical(P[site])
            path.append(site)
        return path

    def generate_path(self, s, t, drone_id):
        path = [s]
        P = self.get_drone(drone_id).probability
        site = categorical(P[s])
        path.append(site)
        while site != t:
            site = categorical(P[site])
            path.append(site)
        return path

    @staticmethod
    def generate_random_system(n, k):

        locations = np.random.rand(n, 2)
        sites = []
        for i in locations:
            sites.append(Site(i))

        drones = []
        for i in range(k):
            speed = abs(random.random())
            probability = generate_prob_matrix(n)
            drones.append(Drone(speed, probability))

        return System(sites, drones)



def compute_arrival_times(path, drone_id):
    arrival_times = []
    t = 0
    for i in range(len(path) - 1):
        t += system.compute_path_time(path[i:i+2], drone_id=drone_id)
        arrival_times.append((drone_id, path[i], path[i+1], t))
    return arrival_times


def generate_arrival_times(system, length):
    arrival_times = [[] for _ in range(len(system.sites))]

    events = []
    for i in range(len(system.drones)):
        path = system.generate_path_of_length(length, i)
        events.extend(compute_arrival_times(path, i))

    def get_key(item):
        return item[3]

    events = sorted(events, key=get_key)

    for event in events:
        drone_id = event[0]
        site_id = event[2]
        time = event[3]
        arrival_times[site_id].append((drone_id, time))

    return arrival_times


def compute_cost(system, n):
    arrival_times = generate_arrival_times(system, n)
    interarrival_times = [[] for _ in range(len(system.sites))]
    for i in range(len(arrival_times)):
        arrivals = arrival_times[i]
        for j in range(len(arrivals) - 1):
            interarrival_times[i].append(arrivals[j+1][1] - arrivals[j][1])

    interarrival_avgs = [compute_average(i) for i in interarrival_times]
    return max(interarrival_avgs)


def compute_average(data):
    return (1/len(data))*sum(data)
