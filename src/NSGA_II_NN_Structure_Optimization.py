# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#	By Javier León Palomares, University of Granada, 2018   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from NonDominatedSort import NonDominatedSortScores, IndirectSort
from NSGA_II_Common_Operators import *
import numpy as np
from numpy.random import choice, ranf, randint, standard_normal
from sklearn.model_selection import cross_val_score
from sklearn.metrics import cohen_kappa_score, accuracy_score
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras import backend as K
import warnings
import gc
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=DeprecationWarning)

#-------------------- Population initialization --------------------

# Returns a population of "pop_size" neural network structures 
# whose number of hidden layers are inside the interval [1,"max_hidden"].
# The number of units at each layer is decided using "input_size" 
# such that the network has a diamond-like structure(the width is maximal
# at the middle and decreases towards both ends).
def InitializePopulation(pop_size, input_size, max_hidden):

	# Matrix of individuals
	pop = np.zeros((pop_size,max_hidden),dtype=np.int32)
	# Initialize their first layer (used as a base for subsequent ones).
	pop[:,0] = randint(input_size,int(input_size*1.75),
							size=pop.shape[0],dtype=np.int32)
	# Choose how many layers each individual will have.
	active_layers = choice(np.arange(1,max_hidden+1),size=pop_size)

	for i in range(pop_size):

		if active_layers[i] > 1:
			middle = int(active_layers[i]/2.0)+1 # Middle layer is the biggest.
			for j in range(1,middle): # Go from smaller to bigger layers.
				pop[i,j] = pop[i,j-1] + pop[i,j-1] * ranf()
			for j in range(middle,active_layers[i]): # From bigger to smaller.
				pop[i,j] = max(int(pop[i,0]/2.0),
							   int(pop[i,j-1] - pop[i,j-1] * ranf()))

	return pop

#-------------------- Crossover operators --------------------

# Let N and M be the biggest layers of "parent1" and "parent2",
# assuming a structure somewhat resembling of a diamond.
# The offspring is created as follows:
# - "parent1" contributes its layers until N (excluding it).
# - "parent2" contributes M and its subsequent layers.
# The excess layers (size > allowed amount of layers) are discarded
# from the end of the array.
def SinglePointCrossover(parent1, parent2):

	# Make sure the splice doesn't overflow the temporal array.
	offspring = np.zeros(len(parent1)*2,dtype=np.int32)
	# Find the biggest layers (ideally around the middle).
	biggest_layer_p1 = np.argmax(parent1)
	biggest_layer_p2 = np.argmax(parent2)
	# last layer - biggest layer yields the number of layers
	# that "parent2" contributes to the offspring.
	last_layers_p2 = (np.argmax(parent2 == 0) or len(parent2)) - \
						 biggest_layer_p2
	# Assemble the offspring.
	offspring[:biggest_layer_p1] = parent1[:biggest_layer_p1]
	offspring[biggest_layer_p1:biggest_layer_p1+last_layers_p2] = \
		parent2[biggest_layer_p2:biggest_layer_p2+last_layers_p2]
	return offspring[:len(parent1)]

#-------------------- Mutation operators --------------------

# Chooses a layer and changes its size using a magnitude modifier
# and a sample from a normal distribution (to add randomness).
# Afterwards, it compensates the overall unit count of the network
# by approximately offsetting the change in the other layers-
# Therefore, it is a shape change rather than a size change.
def SingleLayerMutation(individual, magnitude = 0.3):

	mutated = np.copy(individual)
	valid_layers = np.argmax(mutated == 0) or len(mutated)
	layer = choice(valid_layers)
	change = int(mutated[layer] * magnitude * standard_normal())
	mutated[:valid_layers] -= int(change//(valid_layers))
	mutated[layer] += change
	# If there are invalid layer sizes (<= 0), avoid a crash by
	# making them the mean of the valid elements.
	if len(mutated[:valid_layers][mutated[:valid_layers] <= 0]) > 0:
		mean = np.mean(mutated[:valid_layers][mutated[:valid_layers] > 0])
		mutated[:valid_layers][mutated[:valid_layers] <= 0] = int(mean)
	return mutated

# Scales the whole network evenly using a magnitude modifier. The
# sign of the change is decided randomly.
# It is a size change rather than a shape change.
def ScaleMutation(individual, magnitude = 0.1):

	mutated = np.copy(individual)
	valid_layers = np.argmax(mutated == 0) or len(mutated)
	sign = 1 if ranf() > 0.5 else -1
	mutated[:valid_layers] += sign * int((np.sum(mutated) * magnitude) // valid_layers)
	# If there are invalid layer sizes (<= 0), avoid a crash by
	# making them the mean of the valid elements.
	if len(mutated[:valid_layers][mutated[:valid_layers] <= 0]) > 0:
		mean = np.mean(mutated[:valid_layers][mutated[:valid_layers] > 0])
		mutated[:valid_layers][mutated[:valid_layers] <= 0] = int(mean)
	return mutated

#-------------------- Offspring generation --------------------

# Using already selected individuals, creates as many offspring as
# the number of parents * "crossover_prob" (rounded) +
# the number of parents * "mutation_prob" (rounded) 
# using the crossover operator contained in "crossover" and the
# mutation operators contained in "mutations".
def CreateOffspring(parents, crossover, mutations, crossover_prob = 0.2,
			mutation_prob = 0.8):

	n_crossovers = round(parents.shape[0] * crossover_prob)
	n_mutations = round(parents.shape[0] * mutation_prob)
	offspring = np.empty((n_crossovers+n_mutations,parents.shape[1]),
					dtype=np.int32)
	for n in range(n_crossovers):
		p1, p2 = choice(parents.shape[0],replace=False,size=2)
		offspring[n] = crossover(parents[p1],parents[p2])

	for n in range(n_crossovers,offspring.shape[0]):
		p = choice(parents.shape[0])
		m = choice(len(mutations))
		offspring[n] = mutations[m](parents[p])

	return offspring

#-------------------- Fitness metrics --------------------

# Values closer to zero imply better fitness. Thus, in multiobjective
# optimization, the point (0,0,...,0) is the theoretical optimum.

# Since Keras internally works with one-hot encodings for training
# labels (and sometimes for test labels too), it is important
# to have all labels in one-hot encoding to unify the workflow in
# this respect. It can be done with keras.utils.to_categorical().

# Creates a Keras model compatible with the scikit-learn API.
# "input_size": number of input features.
# "output_size": number of different classes (2 or more).
# "layers": a numpy array of neurons per layer.
# "activation": Keras name of the activation function.
# "lr": learning rate.
# "dropout": dropout rate, if used (> 0). Recommended (0.2-0.5).
def CreateNeuralNetwork(input_size, output_size, layers, activation,
	lr = 0.0, dropout = 0.0):
	
	K.clear_session()
	model = Sequential()
	if dropout > 0.0:
			model.add(Dropout(dropout,input_shape=(input_size,)))
	model.add(Dense(layers[0],activation=activation,input_dim=input_size))

	# Since "layers" has a fixed size equal to the maximum layer count allowed,
	# networks with less layers have 0s starting at some point in the array.
	valid_layers = np.argmax(layers == 0) or len(layers)
	for layer in layers[1:valid_layers]:

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
		layers=individual,activation=activation,dropout=dropout,epochs=25,
		verbose=0)
	network.fit(data['train'],labels['train'])
	predictions = network.predict(data['test'])
	score = cohen_kappa_score(predictions,labels['test'].argmax(1))
	return 1 - score

# Assesses the agreement between the test labels and a classifier
# trained using the layers described by "individual".
# "data" and "labels" are two dictionaries whose keys 'train' and
# 'test' contain the corresponding samples or class labels.
# The returned value is 1 - accuracy.
def SimpleLoss(individual, data, labels, activation, dropout = 0.0, *_):

	network = KerasClassifier(build_fn=CreateNeuralNetwork, 
		input_size=data['train'].shape[1],
		output_size=2 if len(labels['train'].shape) < 2 else labels['train'].shape[1],
		layers=individual,activation=activation,dropout=dropout,epochs=25,
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
		layers=individual,activation=activation,dropout=dropout,epochs=25,
		verbose=0)
	scores = cross_val_score(network,data['train'],
							labels['train'],cv=rounds)
	return 1 - scores.mean()

#-------------------- NSGA-II Algorithm --------------------

# Main procedure of this module.
# "data": a dictionary with two matrices of samples x features (train and test).
# "labels": corresponding class labels for the samples in "data" (same order).
# "max_hidden": number of hidden layers.
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
def StructureOptimization(data, labels, max_hidden, objective_funcs, activation,
		pop_size, generations, seed = 29,
		crossover_prob = 0.2, crossover_func = SinglePointCrossover, 
		mutation_prob = 0.8, mutation_funcs = [SingleLayerMutation], 
		pool_fraction = 0.5, n_cores = 1, show_metrics = False):

	assert data['train'].shape[0] > 0, \
			"You need a non-empty training set"
	assert labels['train'].shape[0] == data['train'].shape[0], \
			"You need an equal number of labels than of samples"
	assert max_hidden > 0, \
			"You need at least 1 hidden layer"
	assert len(objective_funcs) > 1, \
			"You need at least 2 objective functions"
	assert pop_size >= 10, "You need at least 10 individuals"
	assert generations >= 5, "You need at least 5 generations"
	np.random.seed(seed)
	K.tf.set_random_seed(seed)

	if show_metrics:
		print("Population size: "+str(pop_size))
		print("Generations: "+str(generations))
		print("Seed: "+str(seed))
		print("Max hidden layers: "+str(max_hidden))
		print("Crossover probability: "+str(crossover_prob))
		print("Mutation probability: "+str(mutation_prob))
		print("Pool proportion (to population size): "+str(pool_fraction))

	# As some evaluation functions need more arguments,
	# put them together in a generic way for simplicity.
	funcs_with_args = []
	for f in objective_funcs:
		funcs_with_args.append((f,[data,labels,activation]))

	# Pool size for parent selection.
	pool_size = round(pop_size * pool_fraction)
	offspring_size = round(pool_size*crossover_prob)+round(pool_size*mutation_prob)
	# Preallocate intermediate population array (for allocation efficiency).
	intermediate_pop = np.empty((pop_size+offspring_size,max_hidden),dtype=np.int32)
	# Initial population.
	population = InitializePopulation(pop_size,data['train'].shape[1],max_hidden)
	# Initial evaluation using objective_funcs.
	evaluation = EvaluatePopulation(population,funcs_with_args,n_cores=n_cores)
	K.clear_session()
	# Initial non-domination scores [front, crowding_distance].
	nds_scores = NonDominatedSortScores(evaluation)

	if show_metrics:
		print("Mean fitness values of each generation")
		print(np.mean(evaluation,axis=0))

	for gen in range(generations):

		# Parents pool
		parents = TournamentSelection(population,nds_scores[:pop_size],pool_size)
		# Fill the intermediate population with previous generation + offspring.
		intermediate_pop[:pop_size,:] = population
		intermediate_pop[pop_size:,:] = CreateOffspring(parents,crossover_func,
											mutation_funcs,crossover_prob,
											mutation_prob)
		# Apply evaluation and non-dominated sort to the joint population.
		evaluation = EvaluatePopulation(intermediate_pop,funcs_with_args,n_cores=n_cores)
		K.clear_session()
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

	features_list = [np.load("data/example_features_104.npy"),
					 np.load("data/example_features_107.npy"),
					 np.load("data/example_features_110.npy")]

	for data, labels, features in zip(data_list,labels_list, features_list):

		data['train'] = data['train'][:,features]
		data['test'] = data['test'][:,features]
		labels['train'] = to_categorical(labels['train'])
		labels['test'] = to_categorical(labels['test'])

		population, sort_scores, evaluation = \
				StructureOptimization(data=data,labels=labels,max_hidden=3,
				objective_funcs=[KappaLoss,CrossValidationLoss],
				activation="elu",pop_size=10,generations=5,seed = 29,
				crossover_prob = 0.2, crossover_func = SinglePointCrossover, 
				mutation_prob = 0.8, mutation_funcs = [SingleLayerMutation,ScaleMutation], 
				pool_fraction = 0.5, n_cores = 2, show_metrics = True)
