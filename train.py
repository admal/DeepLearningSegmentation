import logging
import os
import pickle
import re

import tensorflow as tf
import cv2
import numpy as np
from keras import optimizers

from FNN.MyModel import MyModel
from FNN.PredictModel import PredictMyModel
from FNN.TrainModel import TrainMyModel
from config import *
from class_mappings import *

import time
import datetime

from predict import labeled_image2rgb_image

logging.basicConfig(filename=TRAIN_LOG_FILE, level=logging.DEBUG)


def normalize_image(image):
    return image / 255


def normalized2rgb(normalized_image):
    return normalized_image * 255


def get_classes_from_rgb(rgb):
    tuple_rgb = tuple(rgb)
    classes = np.zeros(NO_CLASSES, dtype=np.uint8)
    if tuple_rgb in rgb2class:
        classes[rgb2class[tuple_rgb]] = 1
    else:
        print("unknown class")
        classes[30] = 1  # TMP! TODO: do sth with unprocessed pixels
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


def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]


def preprocess_segmented():
    count = 1
    all_paths = os.listdir(LABELED_IMAGES_PATH)
    all_paths.sort(key=natural_keys)
    all_len = len(all_paths)
    pickle_dir = r"C:\Users\wpiot\PycharmProjects\DeepLearningSegmentation\test_segmented\\"
    for labaled_image_path in all_paths:
        print("{} out of {}".format(count, all_len))
        count += 1
        serializefile = pickle_dir + labaled_image_path
        if os.path.isfile(serializefile):
            continue

        labeled_image = cv2.imread(os.path.join(SEGMENTED_IMAGES_PATH, labaled_image_path))
        labeled_image = cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB)
        labeled_image = rgb_image2class_image(labeled_image)

        with open(serializefile, 'wb') as handle:
            pickle.dump(labeled_image, handle, protocol=pickle.HIGHEST_PROTOCOL)


def preapare_arr(arr):
    return np.asarray(arr, dtype='float32')


def load_data(start, end):
    x = []
    y = []
    x_v = []
    y_v = []
    skip = 5
    all_raw_paths = os.listdir(RAW_IMAGES_PATH)
    all_raw_paths.sort(key=natural_keys)

    all_labeled_paths = os.listdir(LABELED_IMAGES_PATH)
    all_labeled_paths.sort(key=natural_keys)
    all_over = False
    if end >= len(all_raw_paths):
        end = len(all_raw_paths) - 1
        all_over = True

    raw_paths = all_raw_paths[start:end]
    segmented_paths = all_labeled_paths[start:end]
    print("from {} to {}".format(start, end))
    count = 0
    for raw_image_path, labaled_image_path in zip(raw_paths, segmented_paths):

        # print("{} {} {}".format(raw_image_path, labaled_image_path, count))
        count += 1

        raw_image = cv2.imread(os.path.join(RAW_IMAGES_PATH, raw_image_path))
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        raw_image = cv2.resize(raw_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        raw_image = normalize_image(raw_image)

        # raw_image = np.expand_dims(raw_image, axis=2)

        serializefile = LABELED_IMAGES_PATH + "\\" + labaled_image_path

        with open(serializefile, 'rb') as handle:
            labeled_image = pickle.load(handle)

        labeled_image = cv2.resize(labeled_image, (IMAGE_WIDTH, IMAGE_HEIGHT))

        if count % skip == 0:
            x_v.append(raw_image)
            y_v.append(labeled_image)
        else:
            x.append(raw_image)
            y.append(labeled_image)

    return preapare_arr(x), preapare_arr(y), preapare_arr(x_v), preapare_arr(y_v), all_over


def main():
    model = TrainMyModel(MODEL_DIRECTORY)
    load_model = model.load_model()
    if load_model:
        load_model.compile(loss="categorical_crossentropy", optimizer=optimizers.adam(LEARNING_RATE), metrics=['accuracy'])
    # print("Input shape: {}".format(x.shape))

    count = 0
    from_num = 0

    to_num = MAX_LOAD_IMAGES
    while count < 1000:
        x, y, x_v, y_v, all_over = load_data(from_num, to_num)
        if len(x_v) == 0 or len(y_v) == 0:
            from_num = 0
            to_num = MAX_LOAD_IMAGES
            continue
        logging.info("[{}]".format(datetime.datetime.now()))
        logging.info("Lengths: x: {}, y: {}, x_v: {}, y_v: {}".format(len(x), len(y), len(x_v), len(y_v)))
        logging.info("Training data shape: x: {}, y: {}".format(x.shape, y.shape))
        logging.info("Validation data shape: x_v: {}, y_v: {}".format(x_v.shape, y_v.shape))

        load_model = model.train(x, y, x_v, y_v, load_model)
        raw_image_path = r"C:\Users\wpiot\PycharmProjects\DeepLearningSegmentation\test\197.png"
        image = cv2.imread(raw_image_path)
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = np.expand_dims(image, axis=2)
        ret = PredictMyModel.predict_on_model(load_model, image)
        rgb_image = labeled_image2rgb_image(ret[0])
        cv2.imwrite('C:\\Users\\wpiot\\PycharmProjects\\DeepLearningSegmentation\\results\\{}.png'.format(count),rgb_image )
        count += 1
        if all_over:
            from_num = 0
            to_num = MAX_LOAD_IMAGES
        else:
            from_num = to_num + 1
            to_num += MAX_LOAD_IMAGES

    logging.info("FINISH")


if __name__ == '__main__':
    main()
