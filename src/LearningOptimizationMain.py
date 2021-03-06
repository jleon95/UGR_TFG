# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#	By Javier León Palomares, University of Granada, 2018   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from NSGA_II_NN_Learning_Optimization import *
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


	features_list = [np.load("../data/features/features_800i_200g_9c_10m_50f_K_CV_LogReg_104.npy"),
					 np.load("../data/features/features_800i_200g_9c_10m_50f_K_CV_LogReg_107.npy"),
					 np.load("../data/features/features_800i_200g_9c_10m_50f_K_CV_LogReg_110.npy")]

	structures_list = [np.load("../data/structures/structure_30i_20g_1c_9m_2hl_K_CV_104.npy.npy"),
					np.load("../data/structures/structure_30i_20g_1c_9m_2hl_K_CV_107.npy.npy"),
					np.load("../data/structures/structure_30i_20g_1c_9m_2hl_K_CV_110.npy.npy")]

	filenames = ["104", "107", "110"]

	print("Fitness metrics: KappaLoss, CrossValidationLoss\n")

	for i, data, labels, features, structure in zip(filenames,data_list,labels_list,features_list,structures_list):

		data['train'] = data['train'][:,features]
		data['test'] = data['test'][:,features]
		labels['train'] = to_categorical(labels['train'])
		labels['test'] = to_categorical(labels['test'])

		print("Individual "+i)
		population, sort_scores, evaluation = \
				LearningOptimization(data=data,labels=labels,max_epochs=400,
				lr_range=[0.005,0.3],dropout_range=[0.0,0.51],
				layers=structure,
				objective_funcs=[KappaLoss,CrossValidationLoss],
				activation="elu",pop_size=30,generations=20,seed=29,
				crossover_prob=0.2, crossover_func=SinglePointCrossover, 
				mutation_prob=0.8, mutation_funcs=[GaussianMutation,DropoutMutation], 
				pool_fraction=0.5, n_cores=1, show_metrics=True)

		np.save("../results/learning_optimization_population_"+i,population)
		np.save("../results/learning_optimization_sort_scores_"+i,sort_scores)
		np.save("../results/learning_optimization_evaluation_"+i,evaluation)
		print("\n\n")
