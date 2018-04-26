import logging
import os
import cv2
import numpy as np
from FNN.TrainModel import TrainModel
from config import *
from class_mappings import *
import time
import datetime
logging.basicConfig(filename=TRAIN_LOG_FILE, level=logging.DEBUG)


def normalize_image(image):
	return image / 255


def normalized2rgb(normalized_image):
	return normalized_image * 255


def get_classes_from_rgb(rgb):
	tuple_rgb = tuple(rgb)
	classes = np.zeros(NO_CLASSES)
	if tuple_rgb in rgb2class:
		classes[rgb2class[tuple_rgb]] = 1
	else:
		classes[30] = 1 #TMP! TODO: do sth with unprocessed pixels
	return classes


def rgb_image2class_image(rgb_image):
	new_image = []
	for x in range(rgb_image.shape[0]):
		new_image.append([])
		for y in range(rgb_image.shape[1]):
			rgb = rgb_image[x, y]
			classes = get_classes_from_rgb(tuple(rgb))
			new_image[x].append(list(classes))
	return np.array(new_image)


def load_data(max_load, curr_idx):
	x = []
	y = []
	x_v = []
	y_v = []
	skip = 10
	for raw_image_path, labaled_image_path, count in zip(os.listdir(RAW_IMAGES_PATH),
	                                                     os.listdir(LABELED_IMAGES_PATH),
	                                                     range(1, MAX_LOAD_IMAGES + 1)):
		if count <= curr_idx: #I know it is shitty solution, TODO: fix it ~AM
			continue
		print("{} {} {}".format(raw_image_path, labaled_image_path, count))

		raw_image = cv2.imread(os.path.join(RAW_IMAGES_PATH, raw_image_path))
		labeled_image = cv2.imread(os.path.join(LABELED_IMAGES_PATH, labaled_image_path))

		raw_image = cv2.resize(raw_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
		labeled_image = cv2.resize(labeled_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
		raw_image = normalize_image(raw_image)
		labeled_image = rgb_image2class_image(labeled_image)

		if count % skip == 0:
			x_v.append(raw_image)
			y_v.append(labeled_image)
		else:
			x.append(raw_image)
			y.append(labeled_image)

		if count >= curr_idx + max_load: #I know it is shitty solution, TODO: fix it ~AM
			break

	return np.asarray(x), np.asarray(y), np.asarray(x_v), np.asarray(y_v)


def main():
	iterations = 1
	load_by_iter = int(MAX_LOAD_IMAGES / iterations)

	for iteration in range(iterations):
		logging.info("START ITER {}".format(iteration + 1))
		x, y, x_v, y_v = load_data(load_by_iter, iteration * load_by_iter)
		print(x.dtype)
		logging.info("[{}]".format(datetime.datetime.now()))
		logging.info("Lengths: x: {}, y: {}, x_v: {}, y_v: {}".format(len(x), len(y), len(x_v), len(y_v)))
		logging.info("Trainig data shape: x: {}, y: {}".format(x.shape, y.shape))
		logging.info("Validation data shape: x_v: {}, y_v: {}".format(x_v.shape, y_v.shape))

		model = TrainModel(MODEL_DIRECTORY + '\\trained-model')
		# print("Input shape: {}".format(x.shape))
		model.train(x, y, x_v, y_v)
		del x
		del y
		del x_v
		del y_v

	logging.info("FINISH")


if __name__ == '__main__':
	main()
