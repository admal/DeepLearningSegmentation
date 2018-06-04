import cv2
import datetime
import numpy as np

from FNN2.VggnetFCN import VggnetFCN
from class_mappings import *
from config import *
import tensorflow as tf


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
	return np.array(new_image, np.uint8)


def main():
	# raw_image_path = r"C:\Users\Adam\Documents\Materialy_pw\PW\ARDO\Data\test\Seq05VD_f02280.png"
	# raw_image_path = r"C:\Users\Adam\Documents\Materialy_pw\PW\ARDO\Data\test\Seq05VD_f00060.png"
	raw_image_path = r"C:\Users\Adam\Documents\Materialy_pw\PW\ARDO\Data\test\Seq05VD_f02520.png"
	image = cv2.imread(raw_image_path)
	raw_image = cv2.imread(raw_image_path)

	image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	model = VggnetFCN(MODEL_DIRECTORY)
	image = np.array(image, dtype=np.float32)
	image = np.expand_dims(image, axis=0)
	image = tf.keras.applications.vgg16.preprocess_input(image)
	loaded_model = model.load_model()
	if loaded_model is None:
		loaded_model = model.get_model()

	loaded_model = model.compile_model(loaded_model)
	ret = model.predict_on_model(loaded_model, image)
	labeled_image = ret[0]
	rgb_image = labeled_image2rgb_image(labeled_image)
	rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

	cv2.imwrite(r'C:\Users\Adam\Documents\Materialy_pw\PW\ARDO\Results\result-{}.png'.format(
		datetime.datetime.now().timestamp()), rgb_image)

	cv2.imshow("Ret", rgb_image)
	cv2.imshow("Original", raw_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
