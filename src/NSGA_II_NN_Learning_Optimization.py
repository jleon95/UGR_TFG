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

#-------------------- NSGA-II Algorithm --------------------