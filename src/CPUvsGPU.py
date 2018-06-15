# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#	By Javier León Palomares, University of Granada, 2018   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
import time
import warnings
import os
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.utils import to_categorical
from NSGA_II_NN_Learning_Optimization import CreateNeuralNetwork
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Measures training time in GPU
def GPUTrainingTime(data, labels, features, structure):

	start = time.time()
	with tf.device("/gpu:0"):

		network = KerasClassifier(build_fn=CreateNeuralNetwork,
			input_size=data['train'].shape[1],
			output_size=2 if len(labels['train'].shape) < 2 else labels['train'].shape[1],
			layers=structure,activation="elu",lr=0.1,
			dropout=0.0,epochs=25,verbose=0)
		network.fit(data['train'],labels['train'])

	return time.time() - start

# Measures training time in CPU
def CPUTrainingTime(data, labels, features, structure):

	start = time.time()
	with tf.device("/cpu:0"):

		network = KerasClassifier(build_fn=CreateNeuralNetwork,
			input_size=data['train'].shape[1],
			output_size=2 if len(labels['train'].shape) < 2 else labels['train'].shape[1],
			layers=structure,activation="elu",lr=0.1,
			dropout=0.0,epochs=25,verbose=0)
		network.fit(data['train'],labels['train'])

	return time.time() - start

if __name__ == '__main__':
	
	# Pathnames assume that the script is called from parent directory
	# and not from 'src' (data not included in the repository).

	data = {'train': np.load("data/data_training_104.npy"),
		 	'test': np.load("data/data_test_104.npy")}

	labels = {'train': np.load("data/labels_training_104.npy"),
			  'test': np.load("data/labels_test_104.npy")}

	features = np.load("data/features/best_features_800i_200g_9c_10m_50f_K_CV_SVM_104.npy")

	data['train'] = data['train'][:,features]
	data['test'] = data['test'][:,features]

	labels['train'] = to_categorical(labels['train'])
	labels['test'] = to_categorical(labels['test'])

	structures = [[50],[100],[150],[200],[250],[300],[400],[500],[600],[800],[1000],[1250],[1500],[2000],[3000],[4000],[6000],[8000],[10000],[20000],[30000],[40000],[50000],[70000],[100000]]

	times_gpu = [GPUTrainingTime(data,labels,features,structure) for structure in structures]
	#times_cpu = [CPUTrainingTime(data,labels,features,structure) for structure in structures]

	print("GPU:")
	print(times_gpu)
	#print("\n\n")
	#print("CPU:")
	#print(times_cpu)
