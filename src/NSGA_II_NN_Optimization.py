from NonDominatedSort import NonDominatedSortScores, IndirectSort
import multiprocessing
import numpy as np
from numpy.random import choice, ranf, randint
from sklearn.externals.joblib import Parallel, delayed
from sklearn.model_selection import cross_val_score
from sklearn.metrics import cohen_kappa_score, accuracy_score
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.utils import to_categorical

#-------------------- Population initialization --------------------

# Returns a population of "pop_size" neural networks with "n_hidden"
# hidden layers. The number of units at each layer is decided using 
# "input_size" such that the network has a diamond-like structure
# (the width is maximal at the middle and decreases towards both ends).
def InitializePopulation(pop_size, input_size, n_hidden):

	population = np.zeros((pop_size,n_hidden),dtype=np.int32)
	population[:,0] = randint(input_size,int(input_size*1.75),
							size=population.shape[0],dtype=np.int32)

	if n_hidden > 1:

		for ind in population:
			middle = int(len(ind)/2.0)
			for i in range(1,middle):
				ind[i] = int(ind[i-1] + ind[i-1] * ranf())
			for i in range(middle,len(ind)):
				ind[i] = max(int(ind[0]/2.0),int(ind[i-1] - ind[i-1] * ranf()))

	return population	

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
		best_front_pos = candidates[np.argmin(sort_scores[candidates,0])]

		# If both fronts are the same, we can't use "best_front_pos".
		if sort_scores[candidates[0]][0] != sort_scores[candidates[1]][0]:
			selected[i] = population[best_front_pos]
		else:
			max_distance_pos = candidates[np.argmax(sort_scores[candidates,1])]
			selected[i] = population[max_distance_pos]

	return selected

#-------------------- Crossover operators --------------------

#-------------------- Mutation operator --------------------

#-------------------- Offspring generation --------------------

#-------------------- Fitness metrics --------------------

# Values closer to zero imply better fitness. Thus, in multiobjective
# optimization, the point (0,0,...,0) is the theoretical optimum.

# Creates a Keras model compatible with the scikit-learn API.
# "input_size": number of input features.
# "output_size": number of different classes (2 or more).
# "layers": a list or numpy array of neurons per layer.
# "activation": Keras name of the activation function.
# "lr": learning rate.
# "dropout": dropout rate, if used (> 0). Recommended (0.2-0.5).
def CreateNeuralNetwork(input_size, output_size, layers, activation,
	lr = 0.0, dropout = 0.0):

	model = Sequential()
	model.add(Dense(layers[0],activation=activation,input_dim=input_size))

	for layer in layers[1:]:

		if dropout > 0.0:
			model.add(Dropout(dropout))
		model.add(Dense(layer,activation=activation))

	sgd = SGD(lr=lr if lr > 0.0 else 0.1)
	if output_size > 2:
		model.add(Dense(output_size,activation='softmax'))
		model.compile(optimizer=sgd,loss='categorical_crossentropy',
              metrics=['accuracy'])
	else:
		model.add(Dense(output_size,activation='sigmoid'))
		model.compile(optimizer=sgd,loss='binary_crossentropy',
              metrics=['accuracy'])

	return model

# Measures the simplicity of a model by counting the total amount of
# hidden units. A lower count results in a better score (closer to 0).
def Simplicity(individual, *_):

	return np.sum(individual)

# Assesses the agreement between the test labels and a classifier
# trained using the layers described by "individual", taking chance
# into account.
# "data" and "labels" are two dictionaries whose keys 'train' and
# 'test' contain the corresponding samples or class labels.
# The returned value is 1 - Kappa coefficient.
def KappaLoss(individual, data, labels, activation, dropout = 0.0, *_):

	network = KerasClassifier(build_fn=CreateNeuralNetwork, 
		input_size=data['train'].shape[1],
		output_size=2 if len(labels['train'].shape) < 2 else labels['train'].shape[1],
		layers=individual,activation=activation,dropout=dropout,epochs=300,
		verbose=0)
	network.fit(data['train'],labels['train'])
	predictions = network.predict(data['test'])
	return 1 - cohen_kappa_score(predictions,labels['test'])

# Assesses the agreement between the test labels and a classifier
# trained using the layers described by "individual".
# "data" and "labels" are two dictionaries whose keys 'train' and
# 'test' contain the corresponding samples or class labels.
# The returned value is 1 - accuracy.
def SimpleLoss(individual, data, labels, activation, dropout = 0.0, *_):

	network = KerasClassifier(build_fn=CreateNeuralNetwork, 
		input_size=data['train'].shape[1],
		output_size=2 if len(labels['train'].shape) < 2 else labels['train'].shape[1],
		layers=individual,activation=activation,dropout=dropout,epochs=300,
		verbose=0)
	network.fit(data['train'],labels['train'])
	score = network.score(data['test'],labels['test'])
	return 1 - score

# Returns the k-fold cross-validation accuracy loss using
# "individual" to build the hidden layers and "rounds" as k.
# "data" and "labels" are two dictionaries whose keys 'train' and
# 'test' contain the corresponding samples or class labels.
# The returned value is 1 - cross-validation accuracy.
def CrossValidationLoss(individual, data, labels, activation,
	dropout = 0.0, rounds = 5):

	network = KerasClassifier(build_fn=CreateNeuralNetwork, 
		input_size=data['train'].shape[1],
		output_size=2 if len(labels['train'].shape) < 2 else labels['train'].shape[1],
		layers=individual,activation=activation,dropout=dropout,epochs=300,
		verbose=0)
	scores = cross_val_score(network,data['train'],
							labels['train'],cv=rounds)
	return 1 - scores.mean()

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