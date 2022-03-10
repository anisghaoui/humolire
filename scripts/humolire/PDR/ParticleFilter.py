"I commented some part to make it easier to retest, for example resampling"
import math
import random
from argparse import Namespace
from copy import deepcopy
from functools import reduce
from typing import Tuple, List

import numpy as np

from .MapHandler import MapHandler
from .Particle import Particle
from .Resampling import multinomial, pham_criterion


class ParticleFilter:
    """
    A particle filter instance that allows to change the states variable by update the particle via observations.
    Accepts a MapHandler object as a map-aiding object to perform correction

    Parameters
    ----------
    particles_count : int
        must be strictly positive
    map_handler : MapHandler instance
        will influence the correction of the filter by leverage map data
    initial_position : dict, default : None
         {x=float,y= float}. x and y in meters.
    initial_heading : float, in radians. default : 0
        0 is east.
    step_range : Tuple[float,float], default : (-0.1, 0.1),
        Interval of the possible radius in meters that can be drawn from the distribution.
    heading_range : Tuple[float,float], default : (-math.pi / 20, +math.pi / 20)
        Interval of the possible angle in radians that can be drawn from the distribution.
    max_particles_count : int
        If not specified default to particles_count. Maximum number of particles the PF can hold.
         can be different from particles_count
    distribution_law_fn : Callable, default :func: random.uniform
        Any function/callable object that takes at least 2 parameters and returns a float value
    regen_method : str, default "subset"
        particle regeneration method. Currently, either "subset" calls subset_regen or or "exhaustive"
        calls exhaustive_regen. Else, raise a ValueError
    subset_size : int, default 10
        Size of subset to regenerate the particles from
    """

    def __init__(self, particles_count: int,
                 step_range: Tuple[float, float],
                 heading_range: Tuple[float, float],
                 map_handler: MapHandler = None, initial_position: {float, float} = None,
                 initial_heading: float = 0, **kwargs):

        # constants
        assert particles_count > 0
        self.particles_count = particles_count
        self.max_particles_count = kwargs.get("max_particles_count", particles_count)

        if initial_position is not None:
            self.initial_position = Namespace(**initial_position)
            self.current_position = self.initial_position

        else:
            self.initial_position = None
            self.current_position = Particle(0, 0, 0, 1).position

        self.initial_heading = initial_heading
        # Kwargs:
        self.subset_size = kwargs.get("subset_size", 10)
        self.regen_method = kwargs.get("regen_method", "subset")
        self.dist_law_fn = random.gauss

        # variables
        self.current_heading = initial_heading

        # map_handler
        self.map_handler = map_handler
        self.build_grid = kwargs.get('build_grid', False)

        # random generator
        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.choice.html
        rand_state = np.random.get_state()  # The new numpy API cleanup draw the seed from the OS not numpy seed
        self.rng = np.random.default_rng(seed=rand_state[1])  # so save it then reuse it

        # statical properties
        self.step_range = step_range
        self.heading_range = heading_range
        self.weight_threshold = kwargs.get('weight_threshold', 0.0)

        self.resampling = False

        # initialise
        if map_handler is not None and initial_position is None:
            self.particles = self.generate_global()
        else:
            self.particles = self.initialize(**kwargs)

    def update(self, step_length: float, step_heading: float):
        """
        Updates particle filter by calling self.update_particles(). Then, if a MapHandler is available, will perform
        particle filtering until obtaining the correct particle count. finally, compute the mean position and heading
        of the particles as the new pf position and heading.

        Parameters
        ----------
        step_length : float
            Length of a walked step in meters. for a human, can hardly be more than 1.5 m
        step_heading : float
         heading of the last step in radians.

        Returns
        -------
         self: ParticleFilter
            The current instance of the particle filter.
        """
        # save the current state a deepcopy is required in this case because
        # the references to the particles are kept even if the lists have been copied
        if self.map_handler is not None:
            previous_particles = deepcopy(self.particles)

        # estimation using the predictions: move the particles to their new supposed location
        self.particles = self.update_particles(step_length, step_heading)
        self.particles = self.filter_light_particles(self.particles)
        self.particles = self.normalize_weights(self.particles)
        # filter those hitting the wall
        if self.map_handler is not None:
            # noinspection PyUnboundLocalVariable
            self.particles, recandidates = self.map_handler.filter(previous_particles, self.particles)
            self.particles_count = len(self.particles)
            if self.particles_count == 0:
                print("Warning: all particles are dead")
                return

            # there are less particles than the maximum,
            # we need to regenerate some until we reach the particles' max count
            if self.regen_method == "subset":
                regenerated_particles = []
                while self.particles_count < self.max_particles_count:
                    regenerated, _ = self.subset_regen(self.max_particles_count - self.particles_count)
                    regenerated_particles.extend(regenerated)
                    self.particles_count += len(regenerated)
                self.particles.extend(regenerated_particles)
                self.particles_count = len(self.particles)

            elif self.regen_method == "exhaustive":
                self.exhaustive_regen(recandidates,
                                      step_length=step_length,
                                      step_heading=step_heading)
            else:
                raise ValueError(f'Wrong {self.regen_method = }. It is either "exhaustive" or "subset"')
            # we should now have the maximum number of particles in the filter.
            assert self.max_particles_count == self.particles_count

        self.particles = self.normalize_weights(self.particles)
        # if self.resampling:
        #     if pham_criterion([p.weight for p in self.particles]) < 0.125:  # quite arbitrary value
        #         self.resample(self.particles) # we ain't using resampling

        # choose what are the new position and heading
        elected_particle = self.elect_particle(self.particles)
        self.current_position = elected_particle.position
        self.current_heading = elected_particle.heading

    def update_particles(self, step_length: float, step_heading: float, particles_subset: list = None) -> List[
        Particle]:
        """
        Updates the particles by translating them by step_length meters in the step heading radians Yaw direction.
        Adds a random radius and angle drawn from the self.dist_law_fn to each of step_length and step_heading.

        TODO : add equations

        Parameters
        ----------
        step_length : float in meters
        step_heading  : float in radians
        particles_subset : list. default to self.particles
            if given, will be updated.

        Returns
        -------
        particles : List[Particle]
            The list of updated particles
        """
        particles = self.particles if particles_subset is None else particles_subset
        for particle in particles:
            # phi_t =  phi_t-1 + d phi_t + noise
            particle.heading += step_heading + self.dist_law_fn(*self.heading_range)
            # cos/sin(phi_t) * (step_length + noise)
            radius = self.dist_law_fn(*self.step_range)
            dx = np.cos(particle.heading) * (step_length + radius)
            dy = np.sin(particle.heading) * (step_length + radius)
            particle.position.x += dx
            particle.position.y += dy

            if self.map_handler is not None and self.map_handler.grid is not None:
                particle.weight *= self.map_handler.get_likelihood_from_grid(particle.position.y, particle.position.x)
                # if grid is not used, the weight returned is 1
            else:
                particle.weight = 1 / self.particles_count
            assert 0 <= particle.weight <= 1

        return particles

    def subset_regen(self, missing_particles_count: int) -> Tuple[List[Particle], List[Particle]]:
        candidate_particles = []
        mean_particles = []
        for _ in range(missing_particles_count):
            # when there aren't enough particle to for a subset_size, pick whatever is left from self.particles
            subset_size = min(len(self.particles), self.subset_size)  # replace=False => unique particles selected
            particles_subset = self.rng.choice(self.particles, size=subset_size, replace=False)

            # compute the mean position and heading of the particles' subset
            # this also sets the average particle to be the closest one because it calls closest particle
            mean_particle = self.elect_particle(self.normalize_weights(deepcopy(particles_subset)))
            mean_particles.append(mean_particle)

            x_range, y_range, heading_range, weight_range = self.compute_statistics(particles_subset, random.uniform)
            heading = random.uniform(*heading_range)
            x = random.uniform(*x_range)
            y = random.uniform(*y_range)
            weight = max(min(random.uniform(*weight_range), 1), 0)  # ensure 0 <= weight <= 1

            candidate_particle = Particle(x=x, y=y, heading=heading, weight=weight)
            candidate_particles.append(candidate_particle)

        # test if the candidates go through the walls
        valid_particles, rejected_particles = self.map_handler.filter(mean_particles, candidate_particles)

        return valid_particles, rejected_particles

    def exhaustive_regen(self, previous_particles: List[Particle], step_length: float, step_heading: float,
                         max_generation_try: int = 8):  # not so good keeping it as a reminder
        """
        # TODO: document

        """
        recandidates = previous_particles  # these will recandidate
        generation_try = 0
        while self.particles_count < self.max_particles_count:
            # try to regenerate new particle movement from the recandidates
            previous_recandidate = deepcopy(recandidates)
            recandidates = self.update_particles(step_length, step_heading, recandidates)
            valid_candidates, recandidates = self.map_handler.filter(previous_recandidate, recandidates)
            self.particles.extend(valid_candidates)
            self.particles_count = len(self.particles)

            # now some recandidates are simply in a dead posture. if it is the case, break the loop
            has_changed = False if len(recandidates) - len(previous_recandidate) == 0 else True
            if not has_changed:
                generation_try += 1
            if generation_try == max_generation_try:
                break
        # we will generate the missing particles from the current position
        current_particle = Particle(self.current_position.x,
                                    self.current_position.y,
                                    self.current_heading,
                                    1)

        while self.max_particles_count - self.particles_count != 0:
            missing_particles_count = self.max_particles_count - self.particles_count
            candidates = self.generate(missing_particles_count)

            valid_candidates, _ = self.map_handler.filter(
                [current_particle] * missing_particles_count,
                candidates
            )
            self.particles.extend(valid_candidates)
            self.particles_count = len(self.particles)
        return self.particles

    @staticmethod
    def elect_particle(particles: List[Particle]) -> Particle:  # weighed average
        # The new position is the average of all particles
        # Each particle needs to have its heading updated
        x_mean = reduce(lambda accu, p: p.weight * p.position.x + accu, particles, 0.0)
        y_mean = reduce(lambda accu, p: p.weight * p.position.y + accu, particles, 0.0)
        heading_mean = reduce(lambda accu, p: p.weight * p.heading + accu, particles, 0.0)
        weight_mean = reduce(lambda accu, p: p.weight + accu, particles, 0.0) / len(particles)
        assert weight_mean <= 1 and weight_mean >= 0

        average_particle = Particle(x=x_mean, y=y_mean, heading=heading_mean, weight=weight_mean)
        # match the average particle to the closest one, ensuring the selected particle is valid and not inside a wall
        average_particle = ParticleFilter.closest_particle(average_particle, particles)
        # a new object is to built as return value
        return Particle(x=average_particle.position.x,
                        y=average_particle.position.y,
                        heading=average_particle.heading,
                        weight=average_particle.weight)

    @staticmethod
    def closest_particle(average_particle: Particle, particles: List[Particle]) -> Particle:
        closest = None
        min_distance = np.inf
        for particle in particles:
            distance = np.sqrt((average_particle.position.x - particle.position.x) ** 2 +
                               (average_particle.position.y - particle.position.y) ** 2)
            closest = closest if min_distance < distance else particle  # this particle is closer to the average
            min_distance = min(distance, min_distance)
        # set the average particle to the closest one
        average_particle = deepcopy(closest)
        return average_particle

    @staticmethod
    def compute_statistics(particles: List[Particle], dist_law_fn=random.gauss):

        if dist_law_fn == random.uniform:  # never using unform again; keeping it as a reminder
            heading_min, heading_max = +np.inf, -np.inf
            x_min, x_max = +np.inf, -np.inf
            y_min, y_max = +np.inf, -np.inf
            weight_min, weight_max = +np.inf, -np.inf
            for particle in particles:
                heading_min = min(heading_min, particle.heading)
                heading_max = max(heading_max, particle.heading)

                x_min = min(x_min, particle.position.x)
                x_max = max(x_max, particle.position.x)

                y_min = min(y_min, particle.position.y)
                y_max = max(y_max, particle.position.y)

                weight_min = min(weight_min, particle.weight)
                weight_max = max(weight_max, particle.weight)

            heading_range = (heading_min, heading_max)
            x_range = (x_min, x_max)
            y_range = (y_min, y_max)
            weight_range = (weight_min, weight_max)
            return x_range, y_range, heading_range, weight_range

        elif dist_law_fn == random.gauss:
            x = [p.position.x for p in particles]
            x_params = [np.mean(x), np.std(x)]
            y = [p.position.y for p in particles]
            y_params = [np.mean(y), np.std(y)]
            h = [p.heading for p in particles]
            h_params = [np.mean(h), np.std(h)]
            w = [p.weight for p in particles]
            w_params = [np.mean(w), np.std(w)]
            return x_params, y_params, h_params, w_params

        else:
            raise ValueError("self.dist_law_fn is neither random.gauss not random.uniform")

    @staticmethod
    def normalize_weights(particles: List[Particle]) -> List[Particle]:
        weights = np.array([p.weight for p in particles])
        _sum = sum(weights)
        weights = weights / _sum if _sum != 0 else weights
        for p, w in zip(particles, weights):
            p.weight = w
        assert np.all(weights <= 1) and np.all(weights >= 0)
        return particles

    def resample(self, particles: List[Particle]) -> List[Particle]:
        U = np.array(sorted(self.rng.uniform(0, 1, size=len(particles))))
        weights = np.array([p.weight for p in particles])
        indices = multinomial(weights, U)
        # for indexing purpose:
        particles = np.array(particles)[indices]
        return list(particles)

    def initialize(self, initial_radius_range, initial_heading_range, **kwargs) -> List[
        Particle]:
        """
        generates self.max_particles_count particles around the current position with each particle
        having noise on its heading and position. This method is called by the object builder __init__.

        Parameters
        ----------
        initial_radius_range: [float,float]
            The mean and std of the initial distribution the radius of the particles to be spawned is drawn from.

        initial_heading_range: [float, float]
            The mean and std of the initial distribution the heading of the particles to be spawned is drawn from.

        Returns
        -------
        particles: List[Particle]
            contains self.max_particles_count Particle objects with weights 1/self.max_particles_count.
        """
        n = self.max_particles_count
        _particles = []
        for i in range(n):
            on_circle_angle = random.uniform(0, 2 * math.pi)
            radius = random.gauss(*initial_radius_range)
            angle = random.gauss(*initial_heading_range) + self.current_heading
            x = radius * np.cos(on_circle_angle) + self.current_position.x
            y = radius * np.sin(on_circle_angle) + self.current_position.y
            _particles.append(
                Particle(
                    x,
                    y,
                    angle,
                    1 / n)
            )
        return _particles

    def generate(self, n: int = None) -> List[Particle]:
        """
        generates a given "n" particles or self.max_particles_count around the current position

        Parameters
        ----------
        n : int
            number of particles to be generated. if None, will generate self.max_particles_count

        Returns
        -------
        particles: List[Particle]
            contains n Particle objects.
        """

        n = self.max_particles_count if n is None else n
        _particles = []
        for i in range(n):
            on_circle_angle = random.uniform(0, 2 * math.pi)
            radius = self.dist_law_fn(0, self.step_range[1])
            angle = self.dist_law_fn(*self.heading_range) + self.current_heading
            x = radius * np.cos(on_circle_angle) + self.current_position.x
            y = radius * np.sin(on_circle_angle) + self.current_position.y
            _particles.append(Particle(
                x,
                y,
                angle,
                1 / n)
            )
        return _particles

    def generate_global(self):  # This sparkles particles all over the map.
        map_y_range = self.map_handler.y_range
        map_x_range = self.map_handler.x_range
        map_heading_range = (-math.pi, math.pi)
        n = int(20000)
        _particles = []
        for i in range(n):
            _particles.append(Particle(
                self.dist_law_fn(*map_x_range),
                self.dist_law_fn(*map_y_range),
                self.dist_law_fn(*map_heading_range),
                1 / n)
            )
        return _particles

    def filter_light_particles(self, particles: List[Particle]):
        if self.build_grid:
            particles = list(filter(lambda p: p.weight > self.weight_threshold, particles))
        return particles

    def __str__(self):
        s = "Particle filter:\n"
        for k, v in self.__dict__.items():
            if k != "particles":
                s += f'{k} = {v}\n'
            else:
                for p in v:
                    s += f"{p}"
        return s
