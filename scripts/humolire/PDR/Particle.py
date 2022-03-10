from argparse import Namespace

"""
This class is a dependency for 2 other classes. Thus, it has its own file. It is also made to make modifying a 
Particle's state variable easier
"""


class Particle:
    position = None
    heading = None
    weight = None

    def __init__(self, x: float, y: float, heading: float, weight: float):
        """
        a particle that is represented by it is position : position.x and position.y, and heading.
        Parameters
        ----------
        x: float, in meters
        y: float, in meters
        heading: float, in radians. Yaw angle where the particle is heading
        weight: float, no units. The weight associated to the particle
        """
        self.position = Namespace(**dict(x=x, y=y))
        self.heading = heading
        self.weight = weight

    def __str__(self):
        return f" x = {self.position.x:10.4f}," \
               f" y = {self.position.y:10.4f}," \
               f" heading = {self.heading:10.4f} " \
               f" weight = {self.weight:10.4f}\n "
