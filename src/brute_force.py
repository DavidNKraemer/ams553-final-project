from src.tours import max_tours_traversal
import itertools

def partition(ns, m):
    """
    This algorithm will generate the list of all partitions of a given set of
    indices. This algorithm was created by Donald Knuth and this implementation
    was taken from StackExchange.

    References:
        - https://codereview.stackexchange.com/questions/1526/finding-all-k-subset-partitions

    :param ns: An array of integers to be partitioned
    :param m: the number of partitions that should be created.
    :return:
    The list of all partitions of the given list of integers.
    """

    def visit(n, a):
        ps = [[] for i in range(m)]
        for j in range(n):
            ps[a[j + 1]].append(ns[j])
        return ps

    def f(mu, nu, sigma, n, a):
        if mu == 2:
            yield visit(n, a)
        else:
            for v in f(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v
        if nu == mu + 1:
            a[mu] = mu - 1
            yield visit(n, a)
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                yield visit(n, a)
        elif nu > mu + 1:
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = mu - 1
            else:
                a[mu] = mu - 1
            if (a[nu] + sigma) % 2 == 1:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v

    def b(mu, nu, sigma, n, a):
        if nu == mu + 1:
            while a[nu] < mu - 1:
                yield visit(n, a)
                a[nu] = a[nu] + 1
            yield visit(n, a)
            a[mu] = 0
        elif nu > mu + 1:
            if (a[nu] + sigma) % 2 == 1:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] < mu - 1:
                a[nu] = a[nu] + 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = 0
            else:
                a[mu] = 0
        if mu == 2:
            yield visit(n, a)
        else:
            for v in b(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v

    n = len(ns)
    if m == 1:
        return itertools.permutations(range(n))
    a = [0] * (n + 1)
    for j in range(1, m + 1):
        a[n - m + j] = j - 1
    return f(m, n, 0, n, a)


def find_optimal_partition(n, k, cost_fn=None) -> (dict, float):
    """
    Finds the optimal partition and corresponding cost, with respect to
    minimizing the provided cost function.

    :param n: the number of elements in the set
    :param k: the number of partitions to be made
    :param cost_fn: a cost function that receives a partition and returns
        a cost value
    :return: a pair (partition, value) where partition is the optimal partition
        given the cost function provided and value is the corresponding optimal value.
    """
    k_partitions = partition(list(range(n)), k)
    partition_2_value = [(list(p), cost_fn(list(p))) for p in k_partitions]
    team_tour, value = min(partition_2_value, key=lambda x: x[1])

    team_tour = [team_tour] if k == 1 else team_tour
    team_tour_dict = {d: tour for d, tour in enumerate(team_tour)}

    return team_tour_dict, value



def find_optimal_team_tour(sites, k):

    def cost(team_tour):
        team_tour = [team_tour] if k == 1 else team_tour
        team_tour_dict = {d: tour for d, tour in enumerate(team_tour)}
        return max_tours_traversal(team_tour_dict, sites)
    n = len(sites)
    return find_optimal_partition(n, k, cost_fn=cost)
