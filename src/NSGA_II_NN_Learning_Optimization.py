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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=DeprecationWarning)

#-------------------- Population initialization --------------------

# Returns a population of "pop_size" neural network hyperparameter
# configurations in the form [epochs, learning rate]. 
# The range of possible epochs is given by [1,"max_epochs"].
# The range of learning rate values is given by "lr_range".
def InitializePopulation(pop_size, max_epochs, lr_range):

	pop = np.zeros((pop_size,2))
	pop[:,0] = choice(max_epochs,size=pop_size)+1
	pop[:,1] = uniform(*lr_range,size=pop_size)
	return pop

#-------------------- Crossover operators --------------------

# Returns a child whose elements are the arithmetic mean
# of the ones from its parents.
def ArithmeticMeanCrossover(parent1, parent2):

	return (parent1 + parent2) / 2.0

#-------------------- Mutation operators --------------------

# Uses a Gaussian distribution centered in 1 to alter the values
# of an individual by multiplication.
def GaussianMutation(individual, std = 0.25):

	mutated = np.copy(individual)
	coefs = normal(1.0,std,size=len(individual))
	for i in range(len(individual)):
		mutated[i] = mutated[i]*coefs[i] if coefs[i] > 0 else mutated[i]
	return mutated


#-------------------- Offspring generation --------------------

# Using already selected individuals, creates as many offspring as
# the number of parents * "crossover_prob" (rounded) +
# the number of parents * "mutation_prob" (rounded) 
# using the crossover operator contained in "crossover" and the
# mutation operator contained in "mutation".
def CreateOffspring(parents, crossover, mutation, crossover_prob = 0.2,
			mutation_prob = 0.8):

	n_crossovers = round(parents.shape[0] * crossover_prob)
	n_mutations = round(parents.shape[0] * mutation_prob)
	offspring = np.empty((n_crossovers+n_mutations,parents.shape[1]))
	for n in range(n_crossovers):
		p1, p2 = choice(parents.shape[0],replace=False,size=2)
		offspring[n] = crossover(parents[p1],parents[p2])

	for n in range(n_crossovers,offspring.shape[0]):
		p = choice(parents.shape[0])
		offspring[n] = mutation(parents[p])

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
# trained using the epochs and learning rate described by "individual",
# taking chance into account. A fixed structure, "layers", is used.
# "data" and "labels" are two dictionaries whose keys 'train' and
# 'test' contain the corresponding samples or class labels.
# The returned value is 1 - Kappa coefficient.
def KappaLoss(individual, data, labels, layers, activation, dropout = 0.0, *_):

	network = KerasClassifier(build_fn=CreateNeuralNetwork, 
		input_size=data['train'].shape[1],
		output_size=2 if len(labels['train'].shape) < 2 else labels['train'].shape[1],
		layers=layers,activation=activation,lr=individual[1],
		dropout=dropout,epochs=int(individual[0]),verbose=0)
	network.fit(data['train'],labels['train'])
	predictions = network.predict(data['test'])
	score = cohen_kappa_score(predictions,labels['test'].argmax(1))
	return 1 - score

# Assesses the agreement between the test labels and a classifier
# trained using the epochs and learning rate described by "individual".
# A fixed structure, "layers", is used.
# "data" and "labels" are two dictionaries whose keys 'train' and
# 'test' contain the corresponding samples or class labels.
# The returned value is 1 - accuracy.
def SimpleLoss(individual, data, labels, layers, activation, dropout = 0.0, *_):

	network = KerasClassifier(build_fn=CreateNeuralNetwork, 
		input_size=data['train'].shape[1],
		output_size=2 if len(labels['train'].shape) < 2 else labels['train'].shape[1],
		layers=layers,activation=activation,lr=individual[1],
		dropout=dropout,epochs=int(individual[0]),verbose=0)
	network.fit(data['train'],labels['train'])
	score = network.score(data['test'],labels['test'])
	return 1 - score

# Returns the k-fold cross-validation accuracy loss using
# the epochs and learning rate described by "individual",
# "layers" to build the hidden layers and "rounds" as k.
# "data" and "labels" are two dictionaries whose keys 'train' and
# 'test' contain the corresponding samples or class labels.
# The returned value is 1 - cross-validation accuracy.
def CrossValidationLoss(individual, data, labels, layers, activation,
	dropout = 0.0, rounds = 5):

	network = KerasClassifier(build_fn=CreateNeuralNetwork, 
		input_size=data['train'].shape[1],
		output_size=2 if len(labels['train'].shape) < 2 else labels['train'].shape[1],
		layers=layers,activation=activation,lr=individual[1],
		dropout=dropout,epochs=int(individual[0]),verbose=0)
	scores = cross_val_score(network,data['train'],
							labels['train'],cv=rounds)
	return 1 - scores.mean()

#-------------------- NSGA-II Algorithm --------------------