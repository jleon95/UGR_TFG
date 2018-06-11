# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#	By Javier León Palomares, University of Granada, 2018   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import NSGA_II_Feature_Selection as logreg
import NSGA_II_Feature_SelectionSVM as svm
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

	# Logistic Regression
	for j in range(len(seeds)):

		for i, data, labels in zip(filenames,data_list,labels_list):

			population, sort_scores, evaluation = \
					logreg.FeatureSelection(data=data,labels=labels,max_features=50,
					objective_funcs=[logreg.KappaLoss,logreg.CrossValidationLoss],
					pop_size=800,generations=200,seed=seeds[j],crossover_prob=0.9,
					crossover_func=logreg.UniformCrossover,mutation_prob=1.0,
					mutation_func=logreg.FlipBitsMutation,pool_fraction=0.5,
					n_cores=-1,show_metrics=False)

			best_kappa = evaluation[np.argmin(evaluation[:,0]),0]
			model_array = np.load("../results/feature_selection/significance_tests/logreg_svm/logreg_svm_best_kappa_"+i+".npy")
			model_array[j,0] = best_kappa
			np.save("../results/feature_selection/significance_tests/logreg_svm/logreg_svm_best_kappa_"+i,model_array)
			print(model_array)

	# Support Vector Machine
	for j in range(len(seeds)):

		for i, data, labels in zip(filenames,data_list,labels_list):

			population, sort_scores, evaluation = \
					svm.FeatureSelection(data=data,labels=labels,max_features=50,
					objective_funcs=[svm.KappaLoss,svm.CrossValidationLoss],
					pop_size=800,generations=200,seed=seeds[j],crossover_prob=0.9,
					crossover_func=svm.UniformCrossover,mutation_prob=1.0,
					mutation_func=svm.FlipBitsMutation,pool_fraction=0.5,
					n_cores=-1,show_metrics=False)

			best_kappa = evaluation[np.argmin(evaluation[:,0]),0]
			model_array = np.load("../results/feature_selection/significance_tests/logreg_svm/logreg_svm_best_kappa_"+i+".npy")
			model_array[j,1] = best_kappa
			np.save("../results/feature_selection/significance_tests/logreg_svm/logreg_svm_best_kappa_"+i,model_array)
			print(model_array)