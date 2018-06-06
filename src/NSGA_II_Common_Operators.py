# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#	By Javier León Palomares, University of Granada, 2018   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import multiprocessing
import numpy as np
from numpy.random import choice
from sklearn.externals.joblib import Parallel, delayed

# Common operators for every NSGA-II algorithm. These don't need
# any particularization for the problem at hand.

#-------------------- Population evaluation --------------------

# Evaluates every individual of the given population using the
# metrics contained in "objective_funcs".
# "objective_funcs" is a list of tuples (function, [arguments])
# meant to provide optional arguments to the evaluation functions
# if needed (for example, data for model training).
# It can also attempt to decrease computation time by means of
# parallelism ("n_cores"), but if the metrics are too simple
# it can lead to worse performance by overhead.
def EvaluatePopulation(population, objective_funcs, n_cores = 1):

	results = np.empty((population.shape[0],len(objective_funcs)))

	if n_cores > multiprocessing.cpu_count():
		n_cores = -1 # -1 means all cores for joblib.

	with Parallel(n_jobs=n_cores) as parallel:

		for o, (f, args) in enumerate(objective_funcs):

			results[:,o] = parallel(delayed(f)(population[i,:],*args)
							for i in range(population.shape[0]))

	return results

#-------------------- Selection process --------------------

# Binary tournament. Takes as many random individuals as 2 * "pool_size"
# and outputs "pool_size" winners as individuals selected for crossover.
# For each pair, we choose the individual with the lower rank; if there's
# a draw, we favor the one with greater crowding distance.
# "sort_scores" contains (front, crowding distance) for each individual.
# Returns an array of individuals.
def TournamentSelection(population, sort_scores, pool_size):

	selected = np.zeros((pool_size,population.shape[1]))
	for i in range(selected.shape[0]):

		candidates = choice(sort_scores.shape[0],replace=False,size=2)

		if sort_scores[candidates[0]][0] != sort_scores[candidates[1]][0]:
			best_front_pos = candidates[np.argmin(sort_scores[candidates,0])]
			selected[i] = population[best_front_pos]
		else: # If both fronts are the same, choose the one with the most distance.
			max_distance_pos = candidates[np.argmax(sort_scores[candidates,1])]
			selected[i] = population[max_distance_pos]

	return selected
