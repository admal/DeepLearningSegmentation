import tensorflow as tf
from config import *

import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, upsample_2d
from tflearn.layers.estimator import regression


class Model:
	_model_graph = None
	_model = None
	model_dir = "C:\\Users\\ASUS\\Documents\\PW\\SieciNeuronowe\\Projekt2\\Model"

	def __init__(self, model_dir=None):
		"""
        Initializes model
        :param model_dir: directory with model, it can be trained and used to predict rotation
        or with checkpoints to continue training
        """
		if model_dir is None:
			"C:\\Users\\ASUS\\Documents\\PW\\SieciNeuronowe\\Projekt2\\Model"
		else:
			self.model_dir = model_dir

	def get_model(self):
		if self._model_graph is None:
			self._model_graph = self._build_model()
			self._model = tflearn.DNN(self._model_graph,
			                          tensorboard_dir=MODEL_DIRECTORY,
			                          tensorboard_verbose=3)

		return self._model

	def _build_model(self):
		network = input_data(shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
		# VGG Model
		network = conv_2d(network, 64, 3, padding="valid", activation="relu")
		network = conv_2d(network, 64, 3, padding="valid", activation="relu")
		network = max_pool_2d(network, 2)
		network = conv_2d(network, 128, 3, padding="valid", activation="relu")
		network = conv_2d(network, 128, 3, padding="valid", activation="relu")
		network = max_pool_2d(network, 2)
		network = conv_2d(network, 256, 3, padding="valid", activation="relu")
		network = conv_2d(network, 256, 3, padding="valid", activation="relu")
		network = conv_2d(network, 256, 3, padding="valid", activation="relu")
		network = max_pool_2d(network, 2)
		network = conv_2d(network, 512, 3, padding="valid", activation="relu")
		network = conv_2d(network, 512, 3, padding="valid", activation="relu")
		network = conv_2d(network, 512, 3, padding="valid", activation="relu")
		network = max_pool_2d(network, 2)
		network = conv_2d(network, 512, 3, padding="valid", activation="relu")
		network = conv_2d(network, 512, 3, padding="valid", activation="relu")
		network = conv_2d(network, 512, 3, padding="valid", activation="relu")
		network = max_pool_2d(network, 2)
		# fully connected layers to convolution layers
		# network = fully_connected(network, 4096)
		# network = fully_connected(network, 4096)
		# network = fully_connected(network, 4096)
		network = conv_2d(network, 1024, 5)
		network = conv_2d(network, 1024, 5)
		network = conv_2d(network, 1024, 5)
		network = conv_2d(network, NO_CLASSES, 1)
		# network = upsample_2d(network, 2)
		network = regression(
			network,
			learning_rate=LEARNING_RATE,
			loss='categorical_crossentropy',
			name='targets',
			metric='accuracy'
		)

		return network

	def load(self):
		self.get_model().load(self.model_dir)
