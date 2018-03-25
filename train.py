import logging
import os
import cv2
import random
import numpy as np
from FNN.TrainModel import TrainModel
from config import *

logging.basicConfig(filename=TRAIN_LOG_FILE, level=logging.DEBUG)

def normalize_image(image):
	return image / 255

def normalized2rgb(normalized_image):
	return normalized_image * 255

def load_data():
	x = []
	y = []
	x_v = []
	y_v = []
	skip = 10
	for raw_image_path, labaled_image_path, count in zip(os.listdir(RAW_IMAGES_PATH),
	                                                     os.listdir(LABELED_IMAGES_PATH),
	                                                     range(1, MAX_LOAD_IMAGES + 1)):
		print("{} {} {}".format(raw_image_path, labaled_image_path, count))

		raw_image = cv2.imread(os.path.join(RAW_IMAGES_PATH, raw_image_path))
		labeled_image = cv2.imread(os.path.join(LABELED_IMAGES_PATH, labaled_image_path))

		# TODO: add image resizing

		raw_image = normalize_image(raw_image)
		labeled_image = normalize_image(labeled_image)

		if count % skip == 0:
			x.append(raw_image)
			y.append(labeled_image)
		else:
			x_v.append(raw_image)
			y_v.append(labeled_image)

	indices = list(range(len(x)))
	print(len(indices))
	random.shuffle(indices)
	x = [x[i] for i in indices]
	y = [y[i] for i in indices]

	indices = list(range(len(x_v)))
	random.shuffle(indices)
	x_v = [x_v[i] for i in indices]
	y_v = [y_v[i] for i in indices]
	return x, y, x_v, y_v


def main():
	x, y, x_v, y_v = load_data()
	print(len(x))
	print(len(y))
	print(len(x_v))
	print(len(y_v))
	logging.info("START")
	x = np.reshape(x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
	x_v = np.reshape(x_v, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])

	y = np.reshape(x_v, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
	y_v = np.reshape(y_v, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
	model = TrainModel(MODEL_DIRECTORY + '\\trained-model')
	model.train(x, y, x_v, y_v)
	logging.info("FINISH")

if __name__ == '__main__':
	main()
