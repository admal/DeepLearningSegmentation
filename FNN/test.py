import cv2
import numpy as np
import tensorflow as tf

from FNN.VggnetFCN import VggnetFCN
from config import *
from FNN.predict import labeled_image2rgb_image


def main():
	raw_images_path = r"C:\Users\Adam\Documents\Materialy_pw\PW\ARDO\Data\test\\"
	results_path = r"C:\Users\Adam\Documents\Materialy_pw\PW\ARDO\Data\test\Results\\"
	images = ["0001TP_009990", "0006R0_f01770", "0006R0_f02280", "0016E5_01710", "0016E5_05940", "0016E5_08019",
	          "0016E5_08107", "Seq05VD_f00060", "Seq05VD_f01170", "Seq05VD_f02520", "Seq05VD_f02670",
	          "Seq05VD_f03540", "Seq05VD_f03720", "Seq05VD_f05100"]

	for image_name in images:
		print("START: {}".format(image_name))
		image = cv2.imread(raw_images_path + image_name + ".png")
		raw_image = cv2.imread(raw_images_path + image_name + ".png")
		raw_image = cv2.resize(raw_image, (IMAGE_WIDTH, IMAGE_HEIGHT))

		labeled = cv2.imread(SEGMENTED_IMAGES_PATH + "\\" + image_name + "_L.png")
		labeled = cv2.resize(labeled, (IMAGE_WIDTH, IMAGE_HEIGHT))

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

		cv2.imwrite(results_path + image_name + r"result.png", rgb_image)
		cv2.imwrite(results_path + image_name + r"original.png", raw_image)
		cv2.imwrite(results_path + image_name + r"labeled.png", labeled)


if __name__ == '__main__':
	main()
