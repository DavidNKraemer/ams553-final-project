
import collections

Arrival = collections.namedtuple('Arrival', 'drone_id site_id time')
Departure = collections.namedtuple('Departure', 'drone_id site_id time')


class Drone:

    def __init__(self, max_speed, cycle):
        self.max_speed = max_speed
        self.cycle = cycle
        self.location = [0,0]
        self.state(0)

    @property
    def state(self):
        return self.state


class Site:

    def __init__(self, location):
        self.location = location


class System:

    SPEED_DIST = None
    CYCLE_DIST = None
    LOCATION_DIST = None


    def __init__(self, drones, sites):
        self.drones = drones
        self.sites = sites

        self.state = [drone.state for drone in drones]