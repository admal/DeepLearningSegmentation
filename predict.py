import cv2
import os
import numpy as np

from FNN.PredictModel import PredictModel
from config import *
from class_mappings import *


def get_rgb_from_classes(classes):
	max_arg = np.argmax(classes, 0)
	rgb = list(class2rgb[max_arg])
	return np.array(rgb)


def labeled_image2rgb_image(labeled_image):
	new_image = []
	for x in range(labeled_image.shape[0]):
		new_image.append([])
		for y in range(labeled_image.shape[1]):
			classes = labeled_image[x, y]
			rgb = get_rgb_from_classes(classes)
			new_image[x].append(rgb)
	return np.array(new_image)


def main():
	raw_image_path = "C:\\Users\\Adam\\Documents\\Materialy_pw\\PW\\ARDO\\Data\\RawImages\\0006R0_f01290.png"
	image = cv2.imread(os.path.join(RAW_IMAGES_PATH, raw_image_path))
	image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

	model = PredictModel()
	ret = model.predict([image])
	labeled_image = ret[0]
	rgb_image = labeled_image2rgb_image(labeled_image)
	cv2.imshow("Ret", rgb_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
