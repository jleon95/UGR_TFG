from NonDominatedSort import NonDominatedSort
import numpy as np
from numpy.random import randint

# Returns a population of "pop_size" binary-encoded individuals whose
# active features have been selected from the interval [0,"total_features"].
# The number of ones of each individual is in the range provided by "count_range".
def InitializePopulation(pop_size, total_features, count_range):

	# Matrix of individuals.
	population = np.zeros((pop_size,total_features))
	# Get the number of ones of each individual in one go.
	active_features = randint(*count_range,size=pop_size)
	# For each individual, swap some of its zeros for ones.
	for i in range(pop_size):
		population[i][randint(total_features,size=active_features[i])] = 1

	return population

# Main procedure of this module.
# "data": a matrix of samples x features.
# "labels": class labels for the samples in "data" (same order).
# "count_range": interval for the desired number of selected features.
# "objective_funcs": a list of Python functions for fitness evaluation.
# "pop_size": the working population size of the genetic algorithm.
# "generations": how many generations the algorithm will run for.
# "seed": random seed for reproducible experiments.
def FeatureSelection(data, labels, count_range, objective_funcs, pop_size, generations, seed = 29):

	assert data.shape[0] > 0, "You need a non-empty training set"
	assert labels.shape[0] == data.shape[0], \
			"You need an equal number of labels than of samples"
	assert count_range > 0 and count_range < data.shape[1], \
			"You need a feature count between 0 and all features"
	assert len(objective_funcs) > 1, \
			"You need at least 2 objective functions."
	assert pop_size >= 10, "You need at least 10 individuals"
	assert generations >= 5, "You need at least 5 generations"
	np.random.seed(seed)