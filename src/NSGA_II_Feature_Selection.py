from NonDominatedSort import NonDominatedSort
import numpy as np
from numpy.random import randint, ranf

#-------------------- Crossover operators --------------------
# Given:
# 11111111111
# 00000000000
# Take a random section from the first and the rest from the second, as in:
# 11110000000
def SinglePointCrossover(parent1, parent2):

	pivot = randint(1,len(parent1))
	offspring = np.copy(parent1)
	offspring[:pivot] = parent2[:pivot]
	return offspring

# Given:
# 11111111111
# 00000000000
# Use two pivots to insert a section of the second into the first, as in:
# 11110000011
def TwoPointCrossover(parent1, parent2):

	pivot1 = randint(len(parent1))
	pivot2 = randint(len(parent1))
	while pivot1 == pivot2:
		pivot2 = randint(len(parent1))
	if pivot1 > pivot2:
		pivot1, pivot2 = pivot2, pivot1
	offspring = np.copy(parent1)
	offspring[pivot1:pivot2] = parent2[pivot1:pivot2]
	return offspring

# Given:
# 01100111111
# 01000000110
# Keep the matching elements and choose the rest using probabilities, as in:
# 01100010110
def UniformCrossover(parent1, parent2, prob = 0.5):

	offspring = np.copy(parent1)
	for i in range(len(parent1)):
		if parent1[i] != parent2[i]:
			offspring[i] = parent1[i] if ranf() <= prob else parent2[i]

	return offspring

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