# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#	By Javier León Palomares, University of Granada, 2018   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from NonDominatedSort import NonDominatedSortScores, IndirectSort
from NSGA_II_Common_Operators import *
import numpy as np
from numpy.random import choice, ranf
from sklearn.model_selection import cross_val_score
from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn import linear_model

#-------------------- Population initialization --------------------

# Returns a population of "pop_size" binary-encoded individuals whose
# active features have been selected from the interval [0,"total_features").
# The number of ones of each individual is in the range [1,"max_features").
def InitializePopulation(pop_size, total_features, max_features):

	# Matrix of individuals.
	population = np.zeros((pop_size,total_features),dtype=bool)
	# Get the number of ones of each individual in one go.
	active_features = choice(np.arange(1,max_features),size=pop_size)
	# For each individual, swap some of its zeros for ones.
	for i in range(pop_size):
		population[i][choice(total_features,replace=False,
							size=active_features[i])] = 1

	return population

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
	elif len(offspring_ones) < 1: # For this rather unlikely situation,
		offspring = np.copy(parent2) # just avoid increasing the running time.
	return offspring

# Given:
# 11111111111
# 00000000000
# Uses two pivots to insert a section of the second into the first, as in:
# 11110000011
def TwoPointCrossover(parent1, parent2, max_features):

	pivot1, pivot2 = choice(len(parent1),replace=False,size=2)
	if pivot1 > pivot2:
		pivot1, pivot2 = pivot2, pivot1
	offspring = np.copy(parent1)
	offspring[pivot1:pivot2] = parent2[pivot1:pivot2]
	offspring_ones = offspring[offspring > 0]
	if len(offspring_ones) > max_features:
		offspring_ones[choice(len(offspring_ones),replace=False,
						size=len(offspring_ones)-max_features)] = 0
		offspring[offspring > 0] = offspring_ones
	elif len(offspring_ones) < 1: # For this rather unlikely situation,
		offspring = np.copy(parent2) # just avoid increasing the running time.
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
	elif len(offspring_ones) < 1: # For this rather unlikely situation,
		offspring = np.copy(parent2) # just avoid increasing the running time.
	return offspring

#-------------------- Mutation operator --------------------

# Swaps "swaps" random bits.
# Assumes that the elements are boolean in type.
def FlipBitsMutation(chromosome, max_features, swaps = 1):

	mutated = np.copy(chromosome)
	swap_positions = choice(len(mutated),replace=False,
							size=swaps)
	mutated[swap_positions] = np.invert(mutated[swap_positions])
	mutated_ones = mutated[mutated > 0]
	if len(mutated_ones) > max_features:
		mutated_ones[choice(len(mutated_ones),replace=False,
						size=len(mutated_ones)-max_features)] = 0
		mutated[mutated > 0] = mutated_ones
	elif len(mutated_ones) < 1: # Probably won't happen too often.
		mutated = np.copy(chromosome)
	return mutated

#-------------------- Offspring generation --------------------

# Using already selected individuals, creates as many offspring as
# the number of parents * "crossover_prob" (rounded) using the
# crossover operator contained in "crossover".
# "max_features" is used to prevent offspring with more features
# than desired.
def CreateOffspring(parents, crossover, mutation, max_features,
			 crossover_prob = 0.9, mutation_prob = 0.1):

	n_crossovers = round(parents.shape[0] * crossover_prob)
	offspring = np.empty((n_crossovers,parents.shape[1]),dtype=bool)
	for n in range(n_crossovers):
		p1, p2 = choice(parents.shape[0],replace=False,size=2)
		offspring[n] = crossover(parents[p1],parents[p2],max_features)
		if ranf() <= mutation_prob:
			offspring[n] = mutation(offspring[n],max_features)

	return offspring

#-------------------- Fitness metrics --------------------

# Values closer to zero imply better fitness. Thus, in multiobjective
# optimization, the point (0,0,...,0) is the theoretical optimum.

# Measures the simplicity of a feature set by counting the active
# features. A lower count results in a better score (closer to 0).
def Simplicity(individual, *_):

	return np.count_nonzero(individual)

# Assesses the agreement between the test labels and a classifier
# trained using the active features of "individual", taking chance
# into account.
# "data" and "labels" are two dictionaries whose keys 'train' and
# 'test' contain the corresponding samples or class labels.
# The returned value is 1 - Kappa coefficient.
def KappaLoss(individual, data, labels, *_):

	log_reg = linear_model.LogisticRegression()
	log_reg.fit(data['train'][:,individual],labels['train'])
	predictions = log_reg.predict(data['test'][:,individual])
	return 1 - cohen_kappa_score(predictions,labels['test'])

# Assesses the agreement between the test labels and a classifier
# trained using the active features of "individual".
# "data" and "labels" are two dictionaries whose keys 'train' and
# 'test' contain the corresponding samples or class labels.
# The returned value is 1 - accuracy.
def SimpleLoss(individual, data, labels, *_):

	log_reg = linear_model.LogisticRegression()
	log_reg.fit(data['train'][:,individual],labels['train'])
	predictions = log_reg.predict(data['test'][:,individual])
	return 1 - accuracy_score(predictions,labels['test'])

# Returns the k-fold cross-validation accuracy loss using
# "individual" to choose active features and "rounds" as k.
# "data" and "labels" are two dictionaries whose keys 'train' and
# 'test' contain the corresponding samples or class labels.
# The returned value is 1 - cross-validation accuracy.
def CrossValidationLoss(individual, data, labels, rounds = 5):

	log_reg = linear_model.LogisticRegression()
	scores = cross_val_score(log_reg,data['train'][:,individual],
							labels['train'],cv=rounds)
	return 1 - scores.mean()

#-------------------- NSGA-II Algorithm --------------------

# Main procedure of this module.
# "data": a dictionary with two matrices of samples x features (train and test).
# "labels": corresponding class labels for the samples in "data" (same order).
# "max_features": upper bound for the number of selected features.
# "objective_funcs": a list of Python functions for fitness evaluation.
# "pop_size": the working population size of the genetic algorithm.
# "generations": how many generations the algorithm will run for.
# "seed": random seed for reproducible experiments.
# "crossover_prob": in practice, size(parents) x this = size(offspring).
# "crossover_func": crossover method. Built-in Uniform Crossover by default.
# "mutation_prob": probability of a mutation after a successful crossover.
# "mutation_func": mutation method. Built-in Flip Bits Mutation by default.
# "pool_fraction": proportion of parent pool size with respect to "pop_size".
# "n_cores": number of processor cores used in the evaluation step.
def FeatureSelection(data, labels, max_features, objective_funcs, pop_size, generations, 
		seed = 29, crossover_prob = 0.9, crossover_func = UniformCrossover, 
		mutation_prob = 0.8, mutation_func = FlipBitsMutation, pool_fraction = 0.5, 
		n_cores = 1, show_metrics = False):

	assert data['train'].shape[0] > 0, \
			"You need a non-empty training set"
	assert labels['train'].shape[0] == data['train'].shape[0], \
			"You need an equal number of labels than of samples"
	assert max_features > 0 and max_features < data['train'].shape[1], \
			"You need a feature count between 1 and all features"
	assert len(objective_funcs) > 1, \
			"You need at least 2 objective functions"
	assert pop_size >= 10, "You need at least 10 individuals"
	assert generations >= 5, "You need at least 5 generations"
	np.random.seed(seed)

	if show_metrics:
		print("Population size: "+str(pop_size))
		print("Generations: "+str(generations))
		print("Seed: "+str(seed))
		print("Max features: "+str(max_features))
		print("Crossover probability: "+str(crossover_prob))
		print("Mutation probability: "+str(mutation_prob))
		print("Pool proportion (to population size): "+str(pool_fraction))

	# As some evaluation functions need more arguments,
	# put them together in a generic way for simplicity.
	funcs_with_args = []
	for f in objective_funcs:
		funcs_with_args.append((f,[data,labels]))

	# Preallocate intermediate population array (for allocation efficiency).
	intermediate_pop = np.empty((pop_size+round(pop_size*pool_fraction*crossover_prob),
										data['train'].shape[1]),dtype=bool)
	# Initial population.
	population = InitializePopulation(pop_size,data['train'].shape[1],max_features)
	# Initial evaluation using objective_funcs.
	evaluation = EvaluatePopulation(population,funcs_with_args,n_cores=n_cores)
	# Initial non-domination scores [front, crowding_distance].
	nds_scores = NonDominatedSortScores(evaluation)
	# Pool size for parent selection.
	pool_size = round(pop_size * pool_fraction)

	if show_metrics:
		print("Mean fitness values of each generation")
		print(np.mean(evaluation,axis=0))

	for gen in range(generations):

		# Parents pool
		parents = TournamentSelection(population,nds_scores[:pop_size],pool_size)
		# Fill the intermediate population with previous generation + offspring.
		intermediate_pop[:pop_size,:] = population
		intermediate_pop[pop_size:,:] = CreateOffspring(parents,crossover_func,
											mutation_func,max_features,crossover_prob,
											mutation_prob)
		# Apply evaluation and non-dominated sort to the joint population.
		evaluation = EvaluatePopulation(intermediate_pop,funcs_with_args,n_cores=n_cores)
		nds_scores = NonDominatedSortScores(evaluation)

		# Sort population and scores based on front and crowding distance,
		# then keep the best "pop_size" of them for the next generation.
		nds_indices = IndirectSort(nds_scores)
		population = intermediate_pop[nds_indices][:pop_size,:]
		nds_scores = nds_scores[nds_indices][:pop_size,:]

		if show_metrics:
			print(np.mean(evaluation[nds_indices][:pop_size],axis=0))
	
	return population, nds_scores, evaluation[nds_indices][:pop_size,:]

if __name__ == '__main__':
	
	# Pathnames assume that the script is called from parent directory
	# and not from 'src' (data not included in the repository).

	data_list = [{'train': np.load("data/data_training_104.npy"),
				  'test': np.load("data/data_test_104.npy")},
				 {'train': np.load("data/data_training_107.npy"),
				  'test': np.load("data/data_test_107.npy")},
				 {'train': np.load("data/data_training_110.npy"),
				  'test': np.load("data/data_test_110.npy")}]

	labels_list = [{'train': np.load("data/labels_training_104.npy"),
				   'test': np.load("data/labels_test_104.npy")},
				   {'train': np.load("data/labels_training_107.npy"),
				   'test': np.load("data/labels_test_107.npy")},
				   {'train': np.load("data/labels_training_110.npy"),
				   'test': np.load("data/labels_test_110.npy")}]

	for data, labels in zip(data_list,labels_list):

		population, sort_scores, evaluation = \
				FeatureSelection(data=data,labels=labels,max_features=40,
				objective_funcs=[Simplicity,CrossValidationLoss],
				pop_size=50,generations=30,n_cores=2)

		print(evaluation)
