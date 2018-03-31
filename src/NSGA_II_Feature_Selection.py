from NonDominatedSort import NonDominatedSort
import numpy as np
from numpy.random import choice, ranf

#-------------------- Crossover operators --------------------

# Given:
# 11111111111
# 00000000000
# Takes a random section from the first and the rest from the second, as in:
# 11110000000
def SinglePointCrossover(parent1, parent2, max_features):

	pivot = choice(len(parent1))
	offspring = np.copy(parent1)
	offspring[:pivot] = parent2[:pivot]
	offspring_ones = offspring[offspring > 0]
	if len(offspring_ones) > max_features: # If there are more ones than allowed
		# Swap some of them for zeros
		offspring_ones[choice(len(offspring_ones),replace=False,
						size=len(offspring_ones)-max_features)] = 0
		offspring[offspring > 0] = offspring_ones
	return offspring

# Given:
# 11111111111
# 00000000000
# Uses two pivots to insert a section of the second into the first, as in:
# 11110000011
def TwoPointCrossover(parent1, parent2, max_features):

	pivot1 = choice(len(parent1))
	pivot2 = choice(len(parent1))
	while pivot1 == pivot2:
		pivot2 = choice(len(parent1))
	if pivot1 > pivot2:
		pivot1, pivot2 = pivot2, pivot1
	offspring = np.copy(parent1)
	offspring[pivot1:pivot2] = parent2[pivot1:pivot2]
	offspring_ones = offspring[offspring > 0]
	if len(offspring_ones) > max_features:
		offspring_ones[choice(len(offspring_ones),replace=False,
						size=len(offspring_ones)-max_features)] = 0
		offspring[offspring > 0] = offspring_ones
	return offspring

# Given:
# 01100111111
# 01000000110
# Keeps the matching elements and chooses the rest using probabilities, as in:
# 01100010110
def UniformCrossover(parent1, parent2, max_features, prob = 0.5):

	offspring = np.copy(parent1)
	for i in range(len(parent1)):
		if parent1[i] != parent2[i]:
			offspring[i] = parent1[i] if ranf() <= prob else parent2[i]

	offspring_ones = offspring[offspring > 0]
	if len(offspring_ones) > max_features:
		offspring_ones[choice(len(offspring_ones),replace=False,
						size=len(offspring_ones)-max_features)] = 0
		offspring[offspring > 0] = offspring_ones
	return offspring


#-------------------- Population initialization --------------------

# Returns a population of "pop_size" binary-encoded individuals whose
# active features have been selected from the interval [0,"total_features").
# The number of ones of each individual is in the range [1,"max_features").
def InitializePopulation(pop_size, total_features, max_features):

	# Matrix of individuals.
	population = np.zeros((pop_size,total_features))
	# Get the number of ones of each individual in one go.
	active_features = choice(np.arange(1,max_features),size=pop_size)
	# For each individual, swap some of its zeros for ones.
	for i in range(pop_size):
		population[i][choice(total_features,replace=False,
							size=active_features[i])] = 1

	return population

#-------------------- Mutation operator --------------------

# Swaps len(chromosome) * prob (rounded) random bits.
# Assumes that the elements are boolean in type.
def FlipBitsMutation(chromosome, max_features, prob = 0.02):

	mutated = np.copy(chromosome)
	swap_positions = choice(len(mutated),replace=False,
							size=round(len(mutated) * prob))
	mutated[swap_positions] = np.invert(mutated[swap_positions])
	mutated_ones = mutated[mutated > 0]
	if len(mutated_ones) > max_features:
		mutated_ones[choice(len(mutated_ones),replace=False,
						size=len(mutated_ones)-max_features)] = 0
		mutated[mutated > 0] = mutated_ones
	return mutated

# Main procedure of this module.
# "data": a matrix of samples x features.
# "labels": class labels for the samples in "data" (same order).
# "max_features": upper bound for the number of selected features.
# "objective_funcs": a list of Python functions for fitness evaluation.
# "pop_size": the working population size of the genetic algorithm.
# "generations": how many generations the algorithm will run for.
# "seed": random seed for reproducible experiments.
def FeatureSelection(data, labels, max_features, objective_funcs, pop_size, generations, seed = 29):

	assert data.shape[0] > 0, "You need a non-empty training set"
	assert labels.shape[0] == data.shape[0], \
			"You need an equal number of labels than of samples"
	assert max_features > 0 and max_features < data.shape[1], \
			"You need a feature count between 0 and all features"
	assert len(objective_funcs) > 1, \
			"You need at least 2 objective functions."
	assert pop_size >= 10, "You need at least 10 individuals"
	assert generations >= 5, "You need at least 5 generations"
	np.random.seed(seed)