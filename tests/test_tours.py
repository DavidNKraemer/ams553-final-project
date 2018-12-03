from src import tours
import numpy as np

tolerance = 1e-4


def test_tour_traversal():
    square = np.array([[0, 0], [0, 1], [1, 1], [1, 0.]])
    tour = [0, 1, 2, 3]
    assert abs(tours.tour_traversal(tour, square) - 4.) < tolerance
