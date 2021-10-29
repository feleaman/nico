#++++++++++++++++++++++ IMPORT MODULES AND FUNCTIONS +++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import Tk

import sys
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, './lib')
# from m_open_extension import *
# from m_fft import *
# from m_demodulation import *
# from m_denois import *
# from m_det_features import *
# from m_processing import *
# from decimal import Decimal
Inputs = ['mode']
InputsOpt_Defaults = {'hola':'chao'}

def main(argv):
	print('caca')
	# tf.logging.set_verbosity(tf.logging.ERROR)
	
	model = keras.Sequential()
	# Adds a densely-connected layer with 64 units to the model:
	# model.add(keras.layers.Dense(64, activation='relu', input_shape=(1000,32)))
	model.add(keras.layers.Dense(64, activation='relu'))

	# Add another:
	model.add(keras.layers.Dense(64, activation='relu'))
	# Add a softmax layer with 10 output units:
	model.add(keras.layers.Dense(10, activation='softmax'))
	
	
	
	# # Create a sigmoid layer:
	# layers.Dense(64, activation='sigmoid')
	# # Or:
	# layers.Dense(64, activation=tf.sigmoid)

	# # A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
	# layers.Dense(64, kernel_regularizer=keras.regularizers.l1(0.01))
	# # A linear layer with L2 regularization of factor 0.01 applied to the bias vector:
	# layers.Dense(64, bias_regularizer=keras.regularizers.l2(0.01))

	# # A linear layer with a kernel initialized to a random orthogonal matrix:
	# layers.Dense(64, kernel_initializer='orthogonal')
	# # A linear layer with a bias vector initialized to 2.0s:
	# layers.Dense(64, bias_initializer=keras.initializers.constant(2.0))

	
	model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
	
	
	data = np.random.random((1000, 32))
	print(data)
	labels = np.random.random((1000, 10))

	model.fit(data, labels, epochs=10, batch_size=32)


	
	
	return

# plt.show()
def read_parser(argv, Inputs, InputsOpt_Defaults):
	Inputs_opt = [key for key in InputsOpt_Defaults]
	Defaults = [InputsOpt_Defaults[key] for key in InputsOpt_Defaults]
	parser = ArgumentParser()
	for element in (Inputs + Inputs_opt):
		print(element)
		if element == 'no_element':
			parser.add_argument('--' + element, nargs='+')
		else:
			parser.add_argument('--' + element, nargs='?')
	
	args = parser.parse_args()
	config = {}
	for element in Inputs:
		if getattr(args, element) != None:
			config[element] = getattr(args, element)
		else:
			print('Required:', element)
			sys.exit()

	for element, value in zip(Inputs_opt, Defaults):
		if getattr(args, element) != None:
			config[element] = getattr(args, element)
		else:
			print('Default ' + element + ' = ', value)
			config[element] = value

	return config


if __name__ == '__main__':
	main(sys.argv)
