import cv2
import numpy as np
from FNN.Model import Model
from FNN.PredictModel import PredictModel
from class_mappings import *
from config import *

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
    raw_image_path = r"C:\Users\wpiot\PycharmProjects\DeepLearningSegmentation\test\38.png"
    image = cv2.imread(raw_image_path)

    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.expand_dims(image, axis=2)
    model = PredictModel(MODEL_DIRECTORY)
    ret_model = Model.load_model()
    ret = model.predict_on_model(ret_model, image)
    labeled_image = ret[0]
    rgb_image = labeled_image2rgb_image(labeled_image)

    cv2.imwrite('01.png', rgb_image)

    cv2.imshow("Ret", rgb_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
