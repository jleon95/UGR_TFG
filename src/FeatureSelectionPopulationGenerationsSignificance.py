# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#	By Javier León Palomares, University of Granada, 2018   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from NSGA_II_Feature_Selection import *
import numpy as np

if __name__ == '__main__':

	data_list = [{'train': np.load("../data/data_training_104.npy"),
				  'test': np.load("../data/data_test_104.npy")},
				 {'train': np.load("../data/data_training_107.npy"),
				  'test': np.load("../data/data_test_107.npy")},
				 {'train': np.load("../data/data_training_110.npy"),
				  'test': np.load("../data/data_test_110.npy")}]

	labels_list = [{'train': np.load("../data/labels_training_104.npy"),
				   'test': np.load("../data/labels_test_104.npy")},
				   {'train': np.load("../data/labels_training_107.npy"),
				   'test': np.load("../data/labels_test_107.npy")},
				   {'train': np.load("../data/labels_training_110.npy"),
				   'test': np.load("../data/labels_test_110.npy")}]

	filenames = ["104", "107", "110"]

	seeds = [29,28,20,11,26,30,23,7,42,111,293,287,311,420,1023]

	# 300 individuals and 150 generations
	for j in range(len(seeds)):

		for i, data, labels in zip(filenames,data_list,labels_list):

			population, sort_scores, evaluation = \
					FeatureSelection(data=data,labels=labels,max_features=40,
					objective_funcs=[KappaLoss,CrossValidationLoss],
					pop_size=300,generations=150,seed=seeds[j],crossover_prob=0.9,
					crossover_func=UniformCrossover,mutation_prob=1.0,
					mutation_func=FlipBitsMutation,pool_fraction=0.5,
					n_cores=-1,show_metrics=False)

			best_kappa = evaluation[np.argmin(evaluation[:,0]),0]
			crossover_array = np.load("../results/feature_selection/significance_tests/population_generations/population_generations_best_kappa_"+i+".npy")
			crossover_array[j,0] = best_kappa
			np.save("../results/feature_selection/significance_tests/population_generations/population_generations_best_kappa_"+i,crossover_array)
			print(crossover_array)

	# 500 individuals and 100 generations
	for j in range(len(seeds)):

		for i, data, labels in zip(filenames,data_list,labels_list):

			population, sort_scores, evaluation = \
					FeatureSelection(data=data,labels=labels,max_features=40,
					objective_funcs=[KappaLoss,CrossValidationLoss],
					pop_size=500,generations=100,seed=seeds[j],crossover_prob=0.9,
					crossover_func=UniformCrossover,mutation_prob=1.0,
					mutation_func=FlipBitsMutation,pool_fraction=0.5,
					n_cores=-1,show_metrics=False)

			best_kappa = evaluation[np.argmin(evaluation[:,0]),0]
			crossover_array = np.load("../results/feature_selection/significance_tests/population_generations/population_generations_best_kappa_"+i+".npy")
			crossover_array[j,1] = best_kappa
			np.save("../results/feature_selection/significance_tests/population_generations/population_generations_best_kappa_"+i,crossover_array)
			print(crossover_array)

	# 800 individuals and 200 generations
	for j in range(len(seeds)):

		for i, data, labels in zip(filenames,data_list,labels_list):

			population, sort_scores, evaluation = \
					FeatureSelection(data=data,labels=labels,max_features=40,
					objective_funcs=[KappaLoss,CrossValidationLoss],
					pop_size=800,generations=200,seed=seeds[j],crossover_prob=0.9,
					crossover_func=UniformCrossover,mutation_prob=1.0,
					mutation_func=FlipBitsMutation,pool_fraction=0.5,
					n_cores=-1,show_metrics=False)

			best_kappa = evaluation[np.argmin(evaluation[:,0]),0]
			crossover_array = np.load("../results/feature_selection/significance_tests/population_generations/population_generations_best_kappa_"+i+".npy")
			crossover_array[j,2] = best_kappa
			np.save("../results/feature_selection/significance_tests/population_generations/population_generations_best_kappa_"+i,crossover_array)
			print(crossover_array)