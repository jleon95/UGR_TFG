# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#	By Javier León Palomares, University of Granada, 2018   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from NSGA_II_Feature_SelectionSVM import *
import numpy as np
import time
import multiprocessing

if __name__ == '__main__':

	data = {'train': np.load("../data/data_training_104.npy"),
			'test': np.load("../data/data_test_104.npy")}

	labels = {'train': np.load("../data/labels_training_104.npy"),
			  'test': np.load("../data/labels_test_104.npy")}

	cores = [x for x in range(2,multiprocessing.cpu_count()+1)]

	start_sequential = time.time()

	population, sort_scores, evaluation = \
		FeatureSelection(data=data,labels=labels,max_features=50,
			objective_funcs=[KappaLoss,CrossValidationLoss],
			pop_size=100,generations=50,seed=29,crossover_prob=0.9,
			crossover_func=UniformCrossover,mutation_prob=1.0,
			mutation_func=FlipBitsMutation,pool_fraction=0.5,
			n_cores=1,show_metrics=False)

	print("Sequential:")
	print(time.time() - start_sequential)
	print("Parallel:")

	for n_core in cores:

		start_parallel = time.time()

		population, sort_scores, evaluation = \
			FeatureSelection(data=data,labels=labels,max_features=50,
				objective_funcs=[KappaLoss,CrossValidationLoss],
				pop_size=100,generations=50,seed=29,crossover_prob=0.9,
				crossover_func=UniformCrossover,mutation_prob=1.0,
				mutation_func=FlipBitsMutation,pool_fraction=0.5,
				n_cores=n_core,show_metrics=False)

		print("%d %f" % (n_core,time.time() - start_parallel))
