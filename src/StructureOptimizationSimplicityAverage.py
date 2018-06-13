# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#	By Javier León Palomares, University of Granada, 2018   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from NSGA_II_NN_Structure_Optimization import *
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


	features_list = [np.load("../data/features/best_features_800i_200g_9c_10m_50f_K_CV_SVM_104.npy"),
					 np.load("../data/features/best_features_800i_200g_9c_10m_50f_K_CV_SVM_107.npy"),
					 np.load("../data/features/best_features_800i_200g_9c_10m_50f_K_CV_SVM_110.npy")]

	filenames = ["104", "107", "110"]

	for data, labels, features in zip(data_list,labels_list,features_list):

		data['train'] = data['train'][:,features]
		data['test'] = data['test'][:,features]
		labels['train'] = to_categorical(labels['train'])
		labels['test'] = to_categorical(labels['test'])

	seeds = [29,28,20,11,26,30,23,7,42,111]

	print("Fitness metrics: KappaLoss, Simplicity\n")

	for j in range(len(seeds)):

		for i, data, labels, features in zip(filenames,data_list,labels_list,features_list):

			print("Individual "+i)
			population, sort_scores, evaluation = \
					StructureOptimization(data=data,labels=labels,max_hidden=4,
					objective_funcs=[KappaLoss,Simplicity],
					activation="elu",pop_size=60,generations=15,seed=seeds[j],
					crossover_prob=0.1,crossover_func=SinglePointCrossover, 
					mutation_prob=0.9,mutation_funcs=[SingleLayerMutation,ScaleMutation], 
					pool_fraction=0.5,n_cores=1,show_metrics=True)

			best_kappa = evaluation[np.argmin(evaluation[:,0]),0]
			simplicity_array = np.load("../results/structure_optimization/simplicity_test/simplicity_best_kappa_"+i+".npy")
			simplicity_array[j] = best_kappa
			np.save("../results/structure_optimization/simplicity_test/simplicity_best_kappa_"+i,simplicity_array)
			print(simplicity_array)