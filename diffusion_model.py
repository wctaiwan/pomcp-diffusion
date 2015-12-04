import random
import numpy

class DiffusionModel:
    def __init__(self, n, m, ep, p, q):
        self.G = self.__init_graph(n, ep)
        self.infect_prob = p
        self.false_neg_prob = q
        self.infected = set()
        while len(self.infected) < m:
            x = random.randrange(n)
            if x not in self.infected:
                self.infected.add(x)
        self.resistant = set()
        self.observed_infected = self.__generate_observation()

    # Generates and returns a graph
    def __init_graph(self, n, edge_prob):
        G = numpy.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i+1, n):
                r = random.random()
                if r < edge_prob:
                    G[i][j] = 1
                    G[j][i] = 1
        return G

    # Simulates one step of the diffusion process and returns the new set of infected nodes
    # Data for infected and resistant are passed in (not instance state)
    def __simulate_infect(self, infected, resistant):
        n = len(self.G)
        new_infected = infected.copy()
        for i in infected:
            for j in range(n):
                if self.G[i][j] == 0 or j in new_infected or j in resistant:
                    continue
                r = random.random()
                if r < self.infect_prob:
                    new_infected.add(j)
        return new_infected

    # Generates and returns a set of nodes observed as infected
    def __generate_observation(self):
        observed_infected = set()
        for x in self.infected:
            r = random.random()
            if r >= self.false_neg_prob:
                observed_infected.add(x)
        return observed_infected

    # Modifies instance data
    # Adds target to the set of resistant nodes (and removes from infected set, if needed)
    # Updates the true and observed sets of infected nodes based on one step of simulation
    def step(self, target):
        if target is not None:
            if target in self.infected:
                self.infected.remove(target)
            self.resistant.add(target)
        self.infected = self.__simulate_infect(self.infected, self.resistant)
        self.observed_infected = self.__generate_observation()

    # Simulates one step of the process on the given model after target has been vaccinated
    # Returns the new set of infected nodes and the reward
    def simulate_step(self, infected, resistant, target):
        infected_, resistant_ = infected.copy(), resistant.copy()
        if target in infected_:
            infected_.remove(target)
        resistant_.add(target)
        infected_ = self.__simulate_infect(infected_, resistant_)
        # reward = -(additional nodes infected + 1), i.e. vaccinating at equilibrium has 0 reward
        return infected_, resistant_, len(infected)-len(infected_)-1

    # Returns a copy of the set of nodes observed as infected
    def get_observed_infected(self):
        return self.observed_infected.copy()

    # Returns a copy of the set of resistant nodes
    def get_resistant(self):
        return self.resistant.copy()

    # Returns whether the the diffusion process has reached equilibrium
    def has_stabilized(self, infected=None, resistant=None):
        if infected is None:
            infected = self.infected
        if resistant is None:
            resistant = self.resistant

        n = len(self.G)
        for i in infected:
            for j in range(n):
                if self.G[i][j] == 1 and j not in infected and j not in resistant:
                    return False
        return True

    # For a completed process, returns the number of infected and resistant nodes and the value
    # Note that this is equal to sum(rewards at each step)-m for a fully observable model
    def get_value(self):
        assert self.has_stabilized()
        return len(self.infected), len(self.resistant), -(len(self.infected) + len(self.resistant))
