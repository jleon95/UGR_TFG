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

	print("Fitness metrics: KappaLoss, CrossValidationLoss\n")

	for i, (data, labels) in enumerate(zip(data_list,labels_list)):

		print("Individual "+filenames[i])
		population, sort_scores, evaluation = \
				FeatureSelection(data=data,labels=labels,max_features=40,
				objective_funcs=[KappaLoss,CrossValidationLoss],
				pop_size=70,generations=70,seed=29,crossover_prob=0.9,
				crossover_func=UniformCrossover,mutation_prob=1.0,
				mutation_func=FlipBitsMutation,pool_fraction=0.5,
				n_cores=3,show_metrics=True)

		np.save("../results/feature_selection_population_"+filenames[i],population)
		np.save("../results/feature_selection_sort_scores_"+filenames[i],sort_scores)
		np.save("../results/feature_selection_evaluation_"+filenames[i],evaluation)
		print("\n\n")
