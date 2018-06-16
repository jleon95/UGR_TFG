# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#	By Javier León Palomares, University of Granada, 2018   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import NSGA_II_Feature_Selection as logreg
import NSGA_II_Feature_SelectionSVM as svm
import numpy as np
import time

if __name__ == '__main__':

	data = {'train': np.load("../data/data_training_104.npy"),
			'test': np.load("../data/data_test_104.npy")}

	labels = {'train': np.load("../data/labels_training_104.npy"),
			  'test': np.load("../data/labels_test_104.npy")}

	seeds = [29,28,20,11,26]
	
	time_logreg = 0
	time_svm = 0

	for seed in seeds:

		start_logreg = time.time()

		population, sort_scores, evaluation = \
			logreg.FeatureSelection(data=data,labels=labels,max_features=50,
			objective_funcs=[logreg.KappaLoss,logreg.CrossValidationLoss],
			pop_size=800,generations=200,seed=seed,crossover_prob=0.9,
			crossover_func=logreg.UniformCrossover,mutation_prob=1.0,
			mutation_func=logreg.FlipBitsMutation,pool_fraction=0.5,
			n_cores=1,show_metrics=False)

		time_logreg += time.time() - start_logreg

		start_svm = time.time()

		population, sort_scores, evaluation = \
			svm.FeatureSelection(data=data,labels=labels,max_features=50,
			objective_funcs=[svm.KappaLoss,svm.CrossValidationLoss],
			pop_size=800,generations=200,seed=seed,crossover_prob=0.9,
			crossover_func=svm.UniformCrossover,mutation_prob=1.0,
			mutation_func=svm.FlipBitsMutation,pool_fraction=0.5,
			n_cores=11,show_metrics=False)

		time_svm += time.time() - start_svm

	print("Average of %d trials" % len(seeds))
	print("LogReg:")
	print(1.0 * time_logreg / len(seeds))
	print("SVM:")
	print(1.0 * time_svm / len(seeds))