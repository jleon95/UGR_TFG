# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#	By Javier León Palomares, University of Granada, 2018   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from NSGA_II_Feature_SelectionSVM import *
import numpy as np
import time

if __name__ == '__main__':

	data = {'train': np.load("../data/data_training_104.npy"),
			'test': np.load("../data/data_test_104.npy")}

	labels = {'train': np.load("../data/labels_training_104.npy"),
			  'test': np.load("../data/labels_test_104.npy")}

	seeds = [29,28,20,11,26]
	
	time_sequential = 0
	time_parallel = 0

	for seed in seeds:

		start_sequential = time.time()

		population, sort_scores, evaluation = \
			FeatureSelection(data=data,labels=labels,max_features=50,
				objective_funcs=[KappaLoss,CrossValidationLoss],
				pop_size=300,generations=150,seed=seed,crossover_prob=0.9,
				crossover_func=UniformCrossover,mutation_prob=1.0,
				mutation_func=FlipBitsMutation,pool_fraction=0.5,
				n_cores=1,show_metrics=False)

		time_sequential += time.time() - start_sequential

		start_parallel = time.time()

		population, sort_scores, evaluation = \
			FeatureSelection(data=data,labels=labels,max_features=50,
				objective_funcs=[KappaLoss,CrossValidationLoss],
				pop_size=300,generations=150,seed=seed,crossover_prob=0.9,
				crossover_func=UniformCrossover,mutation_prob=1.0,
				mutation_func=FlipBitsMutation,pool_fraction=0.5,
				n_cores=-1,show_metrics=False)

		time_parallel += time.time() - start_parallel

	print("Average of %d trials" % len(seeds))
	print("Sequential:")
	print(1.0 * time_sequential / len(seeds))
	print("Parallel:")
	print(1.0 * time_parallel / len(seeds))