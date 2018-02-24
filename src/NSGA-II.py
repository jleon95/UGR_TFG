import numpy as np

def InitializeFirstFront(population, n_features, n_objectives):

	# "individuals" contains pairs in the format:
	# 1st element: number of individuals that dominate this one
	# 2nd element: list of individuals dominated by this one
	individuals = []
	fronts = np.zeros(population.shape[0])

	for i in range(population.shape[0]): # For each individual

		individuals.append([0,[]])

		for j in range(population.shape[0]): # For every other individual

			dom_less = 0
			dom_equal = 0
			dom_more = 0

			for k in range(n_objectives): # For every objective function value

				# Remember that smaller values are better, the optimal being 0
				if population[i,n_features+k] < population[j,n_features+k]: 
					dom_less += 1
				elif population[i,n_features+k] == population[j,n_features+k]:
					dom_equal += 1
				else
					dom_more += 1

			if dom_less == 0 && dom_equal != n_objectives: # i is farther from the origin than j
				individuals[-1][0] += 1 # Acknowledge that j dominates i
			elif dom_more == 0 && dom_equal != n_objectives # i is closer to the origin than j
				individuals[-1][1].append(j) # Acknowledge that i dominates j (index-based)

		if individuals[-1][0] == 0: # If no other individual dominates i, it belongs to front 1
			fronts[i] = 1

	return fronts

# The population consists of N individuals and their scores for the M objectives
# attached at the end.
# In this case, the individuals are neural networks and their features are the
# number of neurons in each layer.
def NonDominatedSort(population, n_objectives):
	n_individuals = population.shape[0]
	n_features = population.shape[1]
	# Pairs of [front, crowding distance] for each individual
	# in the original order of appearance in "population".
	sort_scores = np.zeros((population.shape[0],2))
	sort_scores[:,0] = InitializeFirstFront(population,n_features,n_objectives)