import os
import tensorflow as tf
import numpy as np

from config import NO_CLASSES, IMAGE_HEIGHT, IMAGE_WIDTH, LEARNING_RATE


def resize_bilinear(images):
	return tf.image.resize_bilinear(images, [IMAGE_HEIGHT, IMAGE_WIDTH])


def bilinear_upsample_weights(factor, number_of_classes):
	filter_size = factor * 2 - factor % 2
	factor = (filter_size + 1) // 2
	if filter_size % 2 == 1:
		center = factor - 1
	else:
		center = factor - 0.5
	og = np.ogrid[:filter_size, :filter_size]
	upsample_kernel = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
	weights = np.zeros((filter_size, filter_size, number_of_classes, number_of_classes),
	                   dtype=np.float32)
	for i in range(number_of_classes):
		weights[:, :, i, i] = upsample_kernel
	return weights


class VggnetFCN:
	_model_graph = None
	_model = None
	model_dir = r"C:\Users\wpiot\PycharmProjects\DeepLearningSegmentation"

	def __init__(self, model_dir):
		self.model_dir = model_dir

	@staticmethod
	def load_model():
		if not os.path.isfile('model.json'):
			return None

		json_file = open('model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = tf.keras.models.model_from_json(loaded_model_json)
		# load weights into new model
		loaded_model.load_weights("model.h5")
		print("Loaded model from disk")
		print(loaded_model.summary())
		return loaded_model

	@staticmethod
	def predict_on_model(model, image):
		ret = model.predict(image)
		return ret

	def get_model(self):
		return self._model()

	def _model(self):
		inputs = tf.keras.Input(shape=(None, None, 3))
		vgg16 = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_tensor=inputs)
		x = tf.keras.layers.Conv2D(filters=NO_CLASSES,
		                           kernel_size=(1, 1))(vgg16.output)
		x = tf.keras.layers.Conv2DTranspose(filters=NO_CLASSES,
		                                    kernel_size=(64, 64),
		                                    strides=(32, 32),
		                                    padding='same',
		                                    activation='sigmoid')(x)
		model = tf.keras.Model(inputs=inputs, outputs=x)
		for layer in model.layers[:15]:
			layer.trainable = False

		return model

	@staticmethod
	def compile_model(model):
		adam = tf.keras.optimizers.Adam(LEARNING_RATE)
		sgd = tf.keras.optimizers.SGD(lr=0.01)
		model.compile(loss="binary_crossentropy", optimizer=sgd)
		return model

	@staticmethod
	def fit_model(model, x, y, x_val, y_val):
		model.fit(x, y, batch_size=8, epochs=2, verbose=1, validation_data=(x_val, y_val))

		model_json = model.to_json()
		with open("model.json", "w") as json_file:
			json_file.write(model_json)
		model.save_weights("model.h5")
		return model