from NonDominatedSort import NonDominatedSortScores, IndirectSort
from NSGA_II_Common_Operators import *
import numpy as np
from numpy.random import choice, ranf, randint, normal, uniform
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
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=DeprecationWarning)

#-------------------- Population initialization --------------------

# Returns a population of "pop_size" neural network hyperparameter
# configurations in the form [epochs, learning rate, dropout rate]. 
# The range of possible epochs is given by [1,"max_epochs"].
# The range of learning rate values is given by "lr_range".
# The range of dropout values is given by "dropout_range".
def InitializePopulation(pop_size, max_epochs, lr_range, dropout_range):

	pop = np.zeros((pop_size,3))
	pop[:,0] = choice(max_epochs,size=pop_size)+1
	pop[:,1] = uniform(*lr_range,size=pop_size)
	pop[:,2] = choice(np.arange(*dropout_range,step=0.05),size=pop_size)
	return pop

#-------------------- Crossover operators --------------------

# Returns a child whose elements are the arithmetic mean
# of the ones from its parents.
def ArithmeticMeanCrossover(parent1, parent2):

	return (parent1 + parent2) / 2.0

# Returns a child which is a direct mix of its parents, with
# no numerical alterations.
def SinglePointCrossover(parent1, parent2):

	offspring = np.copy(parent1)
	offspring[1:] = parent2[1:]
	return offspring

#-------------------- Mutation operators --------------------

# Currently they have more parameters than they use for code simplicity
# in CreateOffspring().

# Uses a Gaussian distribution centered in 1 to alter the epochs and
# learning rate values of an individual by multiplication.
def GaussianMutation(individual, max_epochs, lr_range, dropout_range, std = 0.25):

	mutated = np.copy(individual)
	coefs = normal(1.0,std,size=len(individual))
	mutated[0] = mutated[0]*coefs[0] if 0 < mutated[0]*coefs[0] <= max_epochs \
									 else mutated[0]
	mutated[1] = mutated[1]*coefs[1] if lr_range[0] <= mutated[1]*coefs[1] <= lr_range[1] \
									 else mutated[1]
	return mutated

# Alters the dropout rate of an individual in a controlled way
# (steps of +- 0.05 points).
def DropoutMutation(individual, max_epochs, lr_range, dropout_range):

	mutated = np.copy(individual)
	new_value = mutated[2] + choice([-1,1])*0.05
	mutated[2] = new_value if dropout_range[0] <= new_value <= dropout_range[1] \
						   else mutated[2]
	return mutated

#-------------------- Offspring generation --------------------

# Using already selected individuals, creates as many offspring as
# the number of parents * "crossover_prob" (rounded) +
# the number of parents * "mutation_prob" (rounded) 
# using the crossover operator contained in "crossover" and the
# mutation operators contained in "mutations".
def CreateOffspring(parents, crossover, mutations, max_epochs, lr_range,
			dropout_range, crossover_prob = 0.2, mutation_prob = 0.8):

	n_crossovers = round(parents.shape[0] * crossover_prob)
	n_mutations = round(parents.shape[0] * mutation_prob)
	offspring = np.empty((n_crossovers+n_mutations,parents.shape[1]))
	for n in range(n_crossovers):
		p1, p2 = choice(parents.shape[0],replace=False,size=2)
		offspring[n] = crossover(parents[p1],parents[p2])

	for n in range(n_crossovers,offspring.shape[0]):
		p = choice(parents.shape[0])
		m = choice(len(mutations))
		offspring[n] = mutations[m](parents[p],max_epochs,lr_range, dropout_range)

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

# Assesses the agreement between the test labels and a classifier
# trained using the epochs, learning rate and dropout rate described
# by "individual", taking chance into account. A fixed structure,
# "layers", is used.
# "data" and "labels" are two dictionaries whose keys 'train' and
# 'test' contain the corresponding samples or class labels.
# The returned value is 1 - Kappa coefficient.
def KappaLoss(individual, data, labels, layers, activation, *_):

	network = KerasClassifier(build_fn=CreateNeuralNetwork, 
		input_size=data['train'].shape[1],
		output_size=2 if len(labels['train'].shape) < 2 else labels['train'].shape[1],
		layers=layers,activation=activation,lr=individual[1],
		dropout=individual[2],epochs=int(individual[0]),verbose=0)
	network.fit(data['train'],labels['train'])
	predictions = network.predict(data['test'])
	score = cohen_kappa_score(predictions,labels['test'].argmax(1))
	return 1 - score

# Assesses the agreement between the test labels and a classifier
# trained using the epochs, learning rate and dropout rate described
# by "individual". A fixed structure, "layers", is used.
# "data" and "labels" are two dictionaries whose keys 'train' and
# 'test' contain the corresponding samples or class labels.
# The returned value is 1 - accuracy.
def SimpleLoss(individual, data, labels, layers, activation, *_):

	network = KerasClassifier(build_fn=CreateNeuralNetwork, 
		input_size=data['train'].shape[1],
		output_size=2 if len(labels['train'].shape) < 2 else labels['train'].shape[1],
		layers=layers,activation=activation,lr=individual[1],
		dropout=individual[2],epochs=int(individual[0]),verbose=0)
	network.fit(data['train'],labels['train'])
	score = network.score(data['test'],labels['test'])
	return 1 - score

# Returns the k-fold cross-validation accuracy loss using the epochs,
# learning rate and dropout rate described by "individual", using
# "layers" to build the hidden layers and "rounds" as k.
# "data" and "labels" are two dictionaries whose keys 'train' and
# 'test' contain the corresponding samples or class labels.
# The returned value is 1 - cross-validation accuracy.
def CrossValidationLoss(individual, data, labels, layers, activation, rounds = 5):

	network = KerasClassifier(build_fn=CreateNeuralNetwork, 
		input_size=data['train'].shape[1],
		output_size=2 if len(labels['train'].shape) < 2 else labels['train'].shape[1],
		layers=layers,activation=activation,lr=individual[1],
		dropout=individual[2],epochs=int(individual[0]),verbose=0)
	scores = cross_val_score(network,data['train'],
							labels['train'],cv=rounds)
	return 1 - scores.mean()

#-------------------- NSGA-II Algorithm --------------------

# Main procedure of this module.
# "data": a dictionary with two matrices of samples x features (train and test).
# "labels": corresponding class labels for the samples in "data" (same order).
# "max_epochs": maximum amount of training epochs.
# "lr_range": range of values for learning rates.
# "dropout_range": range of values for dropout.
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
def LearningOptimization(data, labels, max_epochs, lr_range, dropout_range,
		layers,	objective_funcs, activation, pop_size, generations, seed = 29,
		crossover_prob = 0.2, crossover_func = ArithmeticMeanCrossover, 
		mutation_prob = 0.8, mutation_funcs = [GaussianMutation], 
		pool_fraction = 0.5, n_cores = 1, show_metrics = False):

	assert data['train'].shape[0] > 0, \
			"You need a non-empty training set"
	assert labels['train'].shape[0] == data['train'].shape[0], \
			"You need an equal number of labels than of samples"
	assert max_epochs > 0, \
			"You need at least 1 training epoch"
	assert len(lr_range) == 2 and 0 < lr_range[0] <= lr_range[1], \
			"You need a valid learning rate range"
	assert len(dropout_range) == 2 and 0 <= dropout_range[0] <= dropout_range[1], \
			"You need a valid dropout range"
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
		print("Max epochs: "+str(max_epochs))
		print("Learning rate range: "+str(lr_range))
		print("Dropout range: "+str(dropout_range))
		print("Structure: "+str(layers))
		print("Crossover probability: "+str(crossover_prob))
		print("Mutation probability: "+str(mutation_prob))
		print("Pool proportion (to population size): "+str(pool_fraction))

	# As some evaluation functions need more arguments,
	# put them together in a generic way for simplicity.
	funcs_with_args = []
	for f in objective_funcs:
		funcs_with_args.append((f,[data,labels,layers,activation]))

	# Pool size for parent selection.
	pool_size = round(pop_size * pool_fraction)
	offspring_size = round(pool_size*crossover_prob)+round(pool_size*mutation_prob)
	# Preallocate intermediate population array (for allocation efficiency).
	intermediate_pop = np.empty((pop_size+offspring_size,3))
	# Initial population.
	population = InitializePopulation(pop_size,max_epochs,lr_range,dropout_range)
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
											mutation_funcs,max_epochs,lr_range,
											dropout_range,crossover_prob,
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
		labels['train'] -= 1
		labels['test'] -= 1
		labels['train'] = to_categorical(labels['train'])
		labels['test'] = to_categorical(labels['test'])

		population, sort_scores, evaluation = \
				LearningOptimization(data=data,labels=labels,max_epochs=300,
				lr_range=[0.005,0.3],dropout_range=[0.0,0.51],
				layers=np.asarray([76]),
				objective_funcs=[KappaLoss,SimpleLoss],
				activation="elu",pop_size=10,generations=5,seed=29,
				crossover_prob=0.5, crossover_func=SinglePointCrossover, 
				mutation_prob=0.5, mutation_funcs=[GaussianMutation,DropoutMutation], 
				pool_fraction=0.5, n_cores=1, show_metrics=True)