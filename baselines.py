import sys
import random
import numpy
from diffusion_model import DiffusionModel

# Parameters
n = 30 # Number of nodes
m = 2 # Number of initially infected nodes
ep = 0.1 # Edge probability in ER model
p = 0.5 # Probability that an infected node infects a susceptible node
q = 0.5 # Probability that an infected node appears healthy
ntrials = 50 # Number of trials
strategy = 2 # 0: random, 1: random infected, 2: infected with most susceptible neighbors

# Returns the (observed) infected node with the most susceptible neighbors
def get_best_target(model, infected, resistant):
    max_count = -1
    best_target = None
    for i in infected:
        count = 0
        for j in range(n):
            if model.G[i][j] == 1 and j not in infected and j not in resistant:
                count += 1
        if count > max_count:
            max_count = count
            best_target = i
    return best_target

# Generates one model, runs the simulation using the chosen strategy and returns the results
def run_once():
    model = DiffusionModel(n, m, ep, p, q)
    while not model.has_stabilized():
        observed_infected, resistant = model.get_observed_infected(), model.get_resistant()
        if model.has_stabilized(observed_infected, resistant):
            # Observed model has stabilized; take no action for one round
            target = None
        else:
            if strategy == 1:
                target = random.choice(tuple(observed_infected))
            elif strategy == 2:
                target = get_best_target(model, observed_infected, resistant)
            else:
                target = random.choice(tuple(set(range(n)).difference(resistant)))
        model.step(target)
    return model.get_value()

def main():
    values = []
    for j in range(ntrials):
        infected_count, resistant_count, value = run_once()
        print "Infected: %d; resistant: %d; value: %d"  % (infected_count, resistant_count, value)
        values.append(value)
    print "Mean value: %f; stdev: %f" % (numpy.mean(values), numpy.std(values))

if __name__ == "__main__":
    main()
