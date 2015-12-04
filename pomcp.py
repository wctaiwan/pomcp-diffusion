import time
import math
import random
import numpy
from diffusion_model import DiffusionModel

# Parameters
n = 30 # Number of nodes
m = 2 # Number of initially infected nodes
ep = 0.1 # Edge probability in ER model
p = 0.5 # Probability that an infected node infects a susceptible node
q = 0.2 # Probability that an infected node appears healthy
tsim = 10 # Time for simulation at each step
c = 0.8 # Exploration constant
ntrials = 50 # Number of trials

MIN_VALUE = -(n+1) # minimum = -n

# Returns the search tree key for a given state
def get_key(infected, resistant):
    return (frozenset(infected), frozenset(resistant))

def search(model, search_tree):
    observed_infected, resistant = model.get_observed_infected(), model.get_resistant()

    if model.has_stabilized(observed_infected, resistant):
        # Observed model has stabilized; take no action for one round
        return None, 0

    # Perform simulations and update the search tree
    nsim = 0
    start_time = time.clock()
    while time.clock() - start_time < tsim:
        for i in range(100):
            simulate(model, observed_infected, resistant, search_tree)
        nsim += 100

    root = get_key(observed_infected, resistant)
    max_value = MIN_VALUE
    best_target = None
    for target in search_tree[root]['candidates']:
        value = search_tree[root]['candidates'][target]['value']
        if value > max_value:
            max_value = value
            best_target = target
    return best_target, nsim

def simulate(model, infected, resistant, search_tree):
    if model.has_stabilized(infected, resistant):
        return 0

    state = get_key(infected, resistant)
    candidates = set(range(n)).difference(resistant)

    if state not in search_tree:
        search_tree[state] = {'count': 0, 'candidates': {}}
        for target in candidates:
            search_tree[state]['candidates'][target] = {'count': 0, 'value': None}
        return rollout(model, infected, resistant)

    tree_node = search_tree[state]
    best_target = None
    if tree_node['count'] < len(candidates): # If there are unexplored branches, pick the first one
        for target in candidates:
            if tree_node['candidates'][target]['count'] == 0:
                best_target = target
                break
    else: # Otherwise, pick the branch with the highest score
        max_score = -1
        for target in candidates:
            target_branch = tree_node['candidates'][target]
            score = float(target_branch['value']) / len(candidates) \
                + c * math.sqrt(math.log(tree_node['count'])/target_branch['count'])
            if score > max_score:
                max_score = score
                best_target = target
    best_target_branch = tree_node['candidates'][best_target]

    infected_, resistant_, reward = model.simulate_step(infected, resistant, best_target)
    value_ = reward + simulate(model, infected_, resistant_, search_tree)

    tree_node['count'] += 1
    best_target_branch['count'] += 1
    if best_target_branch['value'] is None:
        best_target_branch['value'] = value_
    else:
        best_target_branch['value'] = best_target_branch['value'] \
            + float(value_ - best_target_branch['value']) / best_target_branch['count']
    return value_

def rollout(model, infected, resistant):
    if model.has_stabilized(infected, resistant):
        return 0
    candidates = set(range(n)).difference(resistant)
    target = random.choice(tuple(candidates))
    infected_, resistant_, reward = model.simulate_step(infected, resistant, target)
    return reward + rollout(model, infected_, resistant_)

# Generates one model, runs POMCP and returns the results
def run_once():
    model = DiffusionModel(n, m, ep, p, q)
    search_tree = {}
    nsim_total = 0
    while not model.has_stabilized():
        best_target, nsim = search(model, search_tree)
        model.step(best_target)
        nsim_total += nsim
    return model.get_value(), nsim_total

def main():
    values = []
    for j in range(ntrials):
        (infected_count, resistant_count, value), nsim_total = run_once()
        print "Simuations: %d; infected: %d; resistant: %d; value: %d"  % \
            (nsim_total, infected_count, resistant_count, value)
        values.append(value)
    print "Mean value: %f; stdev: %f" % (numpy.mean(values), numpy.std(values))

if __name__ == "__main__":
    main()
