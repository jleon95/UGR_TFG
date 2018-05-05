from NonDominatedSort import NonDominatedSortScores, IndirectSort
import multiprocessing
import numpy as np
from numpy.random import choice, ranf, randint
from sklearn.externals.joblib import Parallel, delayed
from sklearn.model_selection import cross_val_score
from sklearn.metrics import cohen_kappa_score, accuracy_score

#-------------------- Population initialization --------------------

# Returns a population of "pop_size" neural networks with "n_hidden"
# hidden layers. The number of units at each layer is decided using 
# "input_size" such that the network has a diamond-like structure
# (the width is maximal at the middle and decreases towards both ends).
def InitializePopulation(pop_size, input_size, n_hidden):

	population = np.zeros((pop_size,n_hidden),dtype=np.uint16)
	population[:,0] = randint(input_size,int(input_size*1.75),
							size=population.shape[0],dtype=np.uint16)

	if n_hidden > 1:

		for ind in population:
			middle = int(len(ind)/2.0)
			for i in range(1,middle):
				ind[i] = int(ind[i-1] + ind[i-1] * ranf())
			for i in range(middle,len(ind)):
				ind[i] = max(int(ind[0]/2.0),int(ind[i-1] - ind[i-1] * ranf()))

	return population	


#-------------------- Population evaluation --------------------

#-------------------- Selection process --------------------

#-------------------- Crossover operators --------------------

#-------------------- Mutation operator --------------------

#-------------------- Offspring generation --------------------

#-------------------- Fitness metrics --------------------

#-------------------- NSGA-II Algorithm --------------------

# Main procedure of this module.
# "data": a dictionary with two matrices of samples x features (train and test).
# "labels": corresponding class labels for the samples in "data" (same order).
# "n_hidden": number of hidden layers.
# "objective_funcs": a list of Python functions for fitness evaluation.
# "activation": a string with the Keras name of the activation function.
# "pop_size": the working population size of the genetic algorithm.
# "generations": how many generations the algorithm will run for.
# "seed": random seed for reproducible experiments.
# "crossover_prob": in practice, size(parents) x this = size(offspring).
# "crossover_func": crossover method.
# "mutation_prob": probability of a mutation after a successful crossover.
# "mutation_func": mutation method.
# "pool_fraction": proportion of parent pool size with respect to "pop_size".
# "n_cores": number of processor cores used in the evaluation step.
def NNOptimization(data, labels, n_hidden, objective_funcs, activation,
		pop_size, generations, seed = 29,
		crossover_prob = 0.9, crossover_func = None, 
		mutation_prob = 0.8, mutation_func = None, 
		pool_fraction = 0.5, n_cores = 1):

	assert data['train'].shape[0] > 0, \
			"You need a non-empty training set"
	assert labels['train'].shape[0] == data['train'].shape[0], \
			"You need an equal number of labels than of samples"
	assert n_hidden > 0, \
			"You need at least 1 hidden layer"
	assert len(objective_funcs) > 1, \
			"You need at least 2 objective functions."
	assert pop_size >= 10, "You need at least 10 individuals"
	assert generations >= 5, "You need at least 5 generations"
	np.random.seed(seed)


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
